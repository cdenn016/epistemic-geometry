"""
generalized_diagnostics_metrics.py â€” Phase 2 (ctx+cache, uses your stack)

Implements (objective-faithful):
  â€¢ Self:     Î±_belief Â· KL(q || Î¦Ìƒ Â· p)
  â€¢ Feedback: Î»        Â· KL(p || Î¦  Â· q)
  â€¢ Mass:     Ï‰_mass_* Â· ||Ï†||Â² over mask
  â€¢ Curvature:Ï‰_curv_* Â· ||F||Â²  (uses your omega/transport_cache)

All configuration is read from ctx.config (no global config usage).
"""

from __future__ import annotations
from typing import Any, Mapping, Sequence, Dict, Optional, Tuple
import numpy as np

from transport.transport_cache import Phi as tc_Phi  # type: ignore
from transport.transport_cache import plaquette_product as tc_plaquette  # type: ignore
from core.gaussian_core import kl_gaussian, push_gaussian  # type: ignore
from core.numerical_utils import sanitize_sigma  # type: ignore

from transport.transport_cache import Omega as tc_Omega  # type: ignore
from core.gaussian_core import kl_gaussian as kl_divergence  # type: ignore






def _cfg_req(ctx, key: str) -> float:
    cfg = getattr(ctx, "config", None)
    if not isinstance(cfg, dict) or key not in cfg:
        raise KeyError(f"[diagnostics] missing required config key '{key}' in ctx.config")
    return float(cfg[key])

def _mask_bool(mask, tau: float):
    m = np.asarray(mask, np.float32)
    if m.ndim != 2:
        raise ValueError(f"mask must be (H,W); got {m.shape}")
    return m > tau




def _guess_generators(agent, which: str) -> np.ndarray:
    """
    Return a (K, K, 3) stack of Lie algebra generators for the requested fiber.
    Accepts several common attribute names on the agent; picks the first valid one.
    Validates shape: square matrices with a trailing axis of size 3.
    """
    which = which.lower()
    cand_map = {
        "q": ("generators_q", "G_q", "rho_q_generators", "generators", "basis_q"),
        "p": ("generators_p", "G_p", "rho_p_generators", "generators_model", "basis_p"),
    }
    if which not in cand_map:
        raise ValueError(f"_guess_generators: unknown fiber '{which}' (expected 'q' or 'p')")

    for name in cand_map[which]:
        G = getattr(agent, name, None)
        if G is None:
            continue
        G = np.asarray(G)
        if G.ndim == 3 and G.shape[0] == G.shape[1] and G.shape[2] == 3:
            return G.astype(np.float32, copy=False)

    # Nothing matched; build a helpful error message with what we actually found
    found = {name: getattr(agent, name).__class__.__name__
             for name in cand_map[which] if getattr(agent, name, None) is not None}
    raise AttributeError(
        f"_guess_generators: no suitable generators found for fiber '{which}'. "
        f"Tried {cand_map[which]}; found={found or 'none'}. "
        "Expected shape (K,K,3) with square K."
    )




def _active_px(mask_bool: np.ndarray) -> int:
    return int(np.asarray(mask_bool, bool).sum())

def _spd_fallback(S: np.ndarray, eps: float) -> np.ndarray:
    S = 0.5 * (S + np.swapaxes(S, -1, -2))
    eye = np.eye(S.shape[-1], dtype=S.dtype)
    return S + eps * eye

def _spd(S: np.ndarray, eps: float) -> np.ndarray:
    if sanitize_sigma is None:
        return _spd_fallback(np.asarray(S, np.float32), eps)
    return sanitize_sigma(np.asarray(S, np.float32), eps=eps)  # type: ignore

def _reduce(field: np.ndarray, mask: np.ndarray, kind: str) -> float:
    f = np.asarray(field, np.float32)
    m = np.asarray(mask, bool)
    if f.shape != m.shape:
        raise ValueError(f"reduce mismatch: field {f.shape} vs mask {m.shape}")
    if not m.any():
        return 0.0
    if kind == "sum":
        return float(f[m].sum())
    if kind == "mean":
        return float(f[m].mean())
    raise ValueError(f"unknown reduce kind {kind}")

def _failfast(*xs: np.ndarray, where: str = "") -> None:
    for x in xs:
        if x is None:
            raise ValueError(f"{where}: None input")
        if not np.isfinite(np.asarray(x)).all():
            raise FloatingPointError(f"{where}: non-finite values")



def _materialize_morphism(M_full, H, W, Kout, Kin, r, c):
    """
    Normalize a morphism representation to either:
      - (Kout, Kin)  (global matrix), or
      - (M, Kout, Kin)  (per-active-pixel batch)
    Accepts inputs shaped:
      - (Kout, Kin)
      - (M, Kout, Kin)         # already sliced by caller
      - (H, W, Kout, Kin)      # per-pixel field -> we index with (r,c)
    """
    import numpy as np
    M = np.asarray(M_full, np.float32)
    if M.ndim == 2 and M.shape == (Kout, Kin):
        return M  # global morphism
    if M.ndim == 3 and M.shape[1:] == (Kout, Kin):
        # assume already per-item (M == len(r))
        if M.shape[0] != len(r):
            raise ValueError(f"morphism batch length {M.shape[0]} != active pixels {len(r)}")
        return M
    if M.ndim == 4 and M.shape[:2] == (H, W) and M.shape[2:] == (Kout, Kin):
        return M[r, c]  # (M, Kout, Kin)
    raise ValueError(f"unexpected morphism shape {M.shape}; expected (Kout,Kin), (M,Kout,Kin), or (H,W,Kout,Kin)")



def _as_f32(x):
    return np.asarray(x, np.float32)





def _fiber_views(A, which: str):
    """
    Return (mu, Sigma, phi, generators, K) for requested fiber.
    Accepts common attribute spellings; validates shapes.
    """
    which = which.lower()
    H, W = A.mask.shape

    if which == "q":
        mu  = getattr(A, "mu_q_field",  None)
        Sig = getattr(A, "sigma_q_field", None)
        phi = getattr(A, "phi_field",   getattr(A, "phi", None))
        G   = getattr(A, "generators_q", getattr(A, "G_q", None))
    elif which == "p":
        mu  = getattr(A, "mu_p_field",  None)
        Sig = getattr(A, "sigma_p_field", None)
        phi = getattr(A, "phi_model_field", getattr(A, "phi_model", None))
        G   = getattr(A, "generators_p", getattr(A, "G_p", None))
    else:
        raise ValueError("which must be 'q' or 'p'")

    if mu is None or Sig is None or phi is None:
        raise AttributeError(f"agent missing fields for fiber '{which}'")

    mu  = _as_f32(mu)
    Sig = _as_f32(Sig)
    phi = _as_f32(phi)

    if mu.shape[:2] != (H, W):
        raise ValueError(f"mu shape {mu.shape} incompatible with mask {(H,W)}")
    if Sig.shape[:2] != (H, W) or Sig.shape[-1] != Sig.shape[-2] or mu.shape[-1] != Sig.shape[-1]:
        raise ValueError(f"Sigma shape {Sig.shape} incompatible with mu {mu.shape}")

    K = int(mu.shape[-1])
    return mu, Sig, phi, (None if G is None else _as_f32(G)), K


def compute_alignment_field(
    agents, i, *, which="q", ctx=None,
    eps=None, log_eps=None, kl_cap=None, neighbor_tau=None, min_pair_px=None,
):
    """
    Energy-faithful alignment field for agent i:
      out(y,x) = sum_{j in N(i)} KL( N_i || Î©_ij â–· N_j ) over overlapping neighbors,
    with NO averaging and NO caps. Suitable to integrate (sum) directly into the
    total variational energy after multiplying by the term weight.

    Notes:
      - Uses ctx.config for all thresholds/eps (no globals).
      - Î© pulled from transport cache (per-pixel (H,W,K,K)); sliced on overlap.
      - SPD is sanitized minimally (epsÂ·I + symmetrize) via your sanitizer.
      - If A.neighbors exists (list of dicts with {"id": j}), itâ€™s honored;
        otherwise we pick neighbors by overlap heuristic (deterministic).
    """
    if ctx is None:
        raise ValueError("compute_alignment_field requires ctx for transport/cache")

    cfg = getattr(ctx, "config", {}) or {}
    A = agents[i]

    # unified epsilon for energy terms (tiny, fixed)
    eps_val = float(cfg.get("energy_eps", 1e-6) if eps is None else eps)

    # neighbor selection (heuristic only; does not affect integration weights)
    neighbor_tau = float(cfg.get("overlap_eps", 0.20) if neighbor_tau is None else neighbor_tau)
    min_pair_px  = int(cfg.get("min_pair_px", 8) if min_pair_px is None else min_pair_px)

    # integration domain = support via threshold
    support_tau = float(cfg.get("support_cutoff_eps", 1e-3))
    mask_i = _mask_bool(getattr(A, "mask", None), support_tau)
    if mask_i is None or not np.any(mask_i):
        # return shape-compatible zeros
        base = getattr(A, "mask", None)
        return np.zeros_like(_as_f32(base) if base is not None else np.zeros((1, 1), np.float32), np.float32)

    # neighbors (prefer precomputed list)
    nb_list = getattr(A, "neighbors", None)
    if not nb_list:
        nb_list = []
        Ai_m = _as_f32(A.mask)
        for j, B in enumerate(agents):
            if j == i:
                continue
            m_j = _mask_bool(getattr(B, "mask", None), support_tau)
            if m_j is None or not np.any(m_j):
                continue
            inter = (Ai_m > neighbor_tau) & (_as_f32(B.mask) > neighbor_tau)
            if int(inter.sum()) >= min_pair_px:
                nb_list.append({"id": j})

    if not nb_list:
        return np.zeros_like(_as_f32(A.mask), np.float32)

    # choose fiber views
    mu_i_full, S_i_full, _phi_i_full, _G_i, K = _fiber_views(A, which)
    H, W = A.mask.shape

    # accumulate SUM over neighbors (no averaging)
    kl_sum = np.zeros((H, W), np.float32)

    for nb in nb_list:
        j = int(nb["id"])
        B = agents[j]
        mask_j =_mask_bool(getattr(B, "mask", None), support_tau)
        if mask_j is None or not np.any(mask_j):
            continue

        # overlap = integration domain for this pair
        overlap = mask_i & mask_j
        if not np.any(overlap):
            continue

        mu_j_full, S_j_full, _phi_j_full, _G_j, _Kj = _fiber_views(B, which)
        if _Kj != K:
            # Itâ€™s valid for q/p to differ across fibers, but same 'which' must match K
            raise ValueError(f"fiber dim mismatch for which='{which}': Ai K={K}, Aj K={_Kj}")

        # Slice to overlap (M pixels)
        mu_i = _as_f32(mu_i_full)[overlap]                             # (M,K)
        S_i  = sanitize_sigma(_as_f32(S_i_full)[overlap], eps=eps_val) # (M,K,K)
        mu_j = _as_f32(mu_j_full)[overlap]
        S_j  = sanitize_sigma(_as_f32(S_j_full)[overlap], eps=eps_val)

        # Î© from transport cache (per-pixel) â†’ slice to overlap (M,K,K)
        Om_full = tc_Omega(ctx, A, B, which=which)                     # (H,W,K,K) or broadcastable
        Omega   = _as_f32(Om_full)[overlap]                            # (M,K,K)

        # push N_j by Î© into iâ€™s fiber; exact push; SPD sanitize only
        mu_j_t, S_j_t = push_gaussian(
            mu=mu_j, Sigma=S_j, M=Omega,
            eps=eps_val, symmetrize=True, sanitize=True, approx="exact"
        )

        # KL at overlap pixels; floor to [0,âˆž)
        kl_vals = kl_divergence(mu_i, S_i, mu_j_t, S_j_t, eps=eps_val)
        kl_vals = np.where(np.isfinite(kl_vals), kl_vals, 0.0)
        kl_vals = np.maximum(kl_vals, 0.0).astype(np.float32, copy=False)

        # scatter-add into SUM (no averaging)
        idx = np.argwhere(overlap)
        r, c = idx[:, 0], idx[:, 1]
        np.add.at(kl_sum, (r, c), kl_vals)

    # zero outside Ai support; return SUM over neighbors
    return kl_sum * mask_i.astype(np.float32, copy=False)


def _energy_alignment(ctx, agents):
    """
    Returns (align_q_sum, align_q_mean, align_p_sum, align_p_mean), already weight-scaled.
    Uses:
      - beta       (q-side coupling)
      - beta_model (p-side coupling)
    """
    if compute_alignment_field is None:
        return 0.0, 0.0, 0.0, 0.0

    tau = _cfg_req(ctx, "support_cutoff_eps")
    wq  = _cfg_req(ctx, "beta")
    wp  = _cfg_req(ctx, "beta_model")

    if wq == 0.0 and wp == 0.0:
        return 0.0, 0.0, 0.0, 0.0

    sum_q = 0.0
    sum_p = 0.0
    S_tot = 0

    for i, A in enumerate(agents):
        mask = _mask_bool(A.mask, tau)
        if not mask.any():
            continue
        S_tot += int(mask.sum())

        if wq != 0.0:
            fq = np.asarray(compute_alignment_field(agents, i, which="q", ctx=ctx), np.float32)
            if fq.shape != mask.shape:
                raise ValueError(f"align(q) shape {fq.shape} != mask {mask.shape}")
            sum_q += float(fq[mask].sum())

        if wp != 0.0:
            fp = np.asarray(compute_alignment_field(agents, i, which="p", ctx=ctx), np.float32)
            if fp.shape != mask.shape:
                raise ValueError(f"align(p) shape {fp.shape} != mask {mask.shape}")
            sum_p += float(fp[mask].sum())

    if S_tot <= 0:
        return 0.0, 0.0, 0.0, 0.0

    return (
        wq * sum_q,
        wq * (sum_q / S_tot),
        wp * sum_p,
        wp * (sum_p / S_tot),
    )


import numpy as np



def _energy_self_feedback(ctx: Any, agent: Any) -> Tuple[float, float, float, float]:
    """
    Returns (self_sum, self_mean, fb_sum, fb_mean), weight-scaled.
    Robust to Φ̃≈0 by adding a self-term covariance floor and norm-guarding the mean push.
    """
    import numpy as np
    from transport.transport_cache import Phi as tc_Phi

    eps    = _cfg_req(ctx, "energy_eps")
    tau    = _cfg_req(ctx, "support_cutoff_eps")
    w_self = _cfg_req(ctx, "alpha")
    w_fb   = _cfg_req(ctx, "feedback_weight")

    # Optional config knobs (safe defaults if absent)
    cfg       = getattr(ctx, "config", {}) or {}
    _get      = (cfg.get if hasattr(cfg, "get") else lambda k, d: getattr(cfg, k, d))
    sigma_rel = float(_get("self_loss_floor_rel", 1.0))   # try 1.0 to keep self small baseline
    sigma_abs = float(_get("self_loss_floor_abs", 1e-10))
    norm_tol  = float(_get("self_morphism_norm_tol", 1e-3))

    mask = _mask_bool(agent.mask, tau)
    if not mask.any():
        return 0.0, 0.0, 0.0, 0.0

    mu_q = np.asarray(agent.mu_q_field,    np.float32)   # (H,W,Kq)
    S_q  = np.asarray(agent.sigma_q_field, np.float32)   # (H,W,Kq,Kq)
    mu_p = np.asarray(agent.mu_p_field,    np.float32)   # (H,W,Kp)
    S_p  = np.asarray(agent.sigma_p_field, np.float32)   # (H,W,Kp,Kp)
    H, W, Kq = mu_q.shape
    _, _, Kp = mu_p.shape
    if S_q.shape != (H, W, Kq, Kq) or S_p.shape != (H, W, Kp, Kp):
        raise ValueError("Sigma shapes do not match mu shapes")

    # Morphisms (global or per-pixel)
    Phi_q_to_p = tc_Phi(ctx, agent, kind="q_to_p")   # (Kp,Kq) or (H,W,Kp,Kq)
    Phi_p_to_q = tc_Phi(ctx, agent, kind="p_to_q")   # (Kq,Kp) or (H,W,Kq,Kp)

    # Active pixels (gather)
    rrcc = np.argwhere(mask)
    r, c = rrcc[:, 0], rrcc[:, 1]
    mu_q_m, S_q_m = mu_q[r, c], _spd(S_q[r, c], eps)   # (M,Kq), (M,Kq,Kq)
    mu_p_m, S_p_m = mu_p[r, c], _spd(S_p[r, c], eps)   # (M,Kp), (M,Kp,Kp)
    M = mu_q_m.shape[0]

    # Materialize morphisms over active pixels -> (M,Kout,Kin)
    M_p_to_q = _materialize_morphism(Phi_p_to_q, H, W, Kq, Kp, r, c)
    M_q_to_p = _materialize_morphism(Phi_q_to_p, H, W, Kp, Kq, r, c)

    # -------------------- SELF: KL(q || Φ̃·p) with floor + norm-guard --------------------
    # Push p→q (mean & cov)
    mu_pt, S_pt = push_gaussian(mu_p_m, S_p_m, M_p_to_q, eps=eps)   # (M,Kq), (M,Kq,Kq)

    # Norm of Φ̃ per pixel; if Φ̃ is effectively zero, zero the mean push (prevents tiny residual quad)
    if M_p_to_q.ndim == 3:
        Mnorm = np.linalg.norm(M_p_to_q, axis=(-2, -1))            # (M,)
    else:  # (Kq,Kp) global
        Mnorm = np.full((M,), np.linalg.norm(M_p_to_q), dtype=np.float32)
    tiny = (Mnorm < norm_tol)
    if tiny.any():
        mu_pt[tiny] = 0.0

    # Loss floor on pushed covariance: S_pt ← S_pt + σ² I, with σ² scaled to var(Σ_q) per pixel
    var_q  = np.trace(S_q_m, axis1=-2, axis2=-1) / float(Kq)       # (M,)
    sigma2 = (sigma_rel * var_q + sigma_abs).astype(np.float32)    # (M,)
    Iq     = np.eye(Kq, dtype=np.float32)
    S_pt   = S_pt + sigma2[:, None, None] * Iq
    S_pt   = _spd(S_pt, eps)

    kl_self   = kl_gaussian(mu_q_m, S_q_m, mu_pt, S_pt, eps=eps)   # (M,)
    self_sum  = w_self * float(kl_self.sum())
    self_mean = w_self * (float(kl_self.mean()) if kl_self.size else 0.0)

    # -------------------- FEEDBACK: KL(p || Φ·q) (unchanged; skipped if weight=0) --------
    if w_fb > 0.0:
        mu_qt, S_qt = push_gaussian(mu_q_m, S_q_m, M_q_to_p, eps=eps)
        kl_fb       = kl_gaussian(mu_p_m, S_p_m, mu_qt, S_qt, eps=eps)
        fb_sum      = w_fb * float(kl_fb.sum())
        fb_mean     = w_fb * (float(kl_fb.mean()) if kl_fb.size else 0.0)
    else:
        fb_sum, fb_mean = 0.0, 0.0

    return self_sum, self_mean, fb_sum, fb_mean



def _spd_sym(A, eps):
    # Symmetrize then eigen-floor to guarantee SPD
    A = 0.5 * (A + np.swapaxes(A, -1, -2))
    try:
        w, V = np.linalg.eigh(A)                  # (...,K)
        w = np.clip(w, eps, None)
        return (V * w[..., None, :]) @ np.swapaxes(V, -1, -2)
    except Exception:
        # very rare: fallback to diagonal jitter
        K = A.shape[-1]
        return A + eps * np.eye(K, dtype=A.dtype)

def _chol_prec_logdet(S, eps):
    """Return (Prec, logdet) using robust Cholesky, sanitizing if needed."""
    try:
        L = np.linalg.cholesky(S)  # (...,K,K)
    except np.linalg.LinAlgError:
        S = _spd_sym(S, eps)
        L = np.linalg.cholesky(S)
    Linv = np.linalg.inv(L)
    Prec = Linv.swapaxes(-1, -2) @ Linv
    # logdet = 2 * sum(log(diag(L)))
    diagL = np.diagonal(L, axis1=-2, axis2=-1)
    logdet = 2.0 * np.log(np.clip(diagL, eps, None)).sum(axis=-1)
    return Prec, logdet

def _gaussian_kl_components(mu_a, Sa, mu_b, Sb, eps=1e-8):
    """
    KL( N_a || N_b ) components with SPD-safe handling.
    Returns dict with quad, trace, logdet, dim, kl.
    """
    # Ensure symmetry & SPD tolerance before factorization
    Sa = _spd_sym(Sa, eps)
    Sb = _spd_sym(Sb, eps)

    Prec_b, logdet_b = _chol_prec_logdet(Sb, eps)
    _,      logdet_a = _chol_prec_logdet(Sa, eps)

    dmu = (mu_b - mu_a)[..., None]                           # (...,K,1)
    quad = np.squeeze(np.swapaxes(dmu, -1, -2) @ Prec_b @ dmu, axis=(-2, -1))
    trace = np.einsum("...ij,...ji->...", Prec_b, Sa, optimize=True)
    K = Sa.shape[-1]
    logdet = (logdet_b - logdet_a)
    kl = 0.5 * (trace + quad - K + logdet)

    return {
        "quad":   quad.astype(np.float32),
        "trace":  trace.astype(np.float32),
        "logdet": logdet.astype(np.float32),
        "dim":    np.full_like(quad, K, dtype=np.float32),
        "kl":     kl.astype(np.float32),
    }


def compute_self_feedback_maps(ctx, agent, *, return_components=False):
    """
    Per-pixel maps for self and feedback terms, with robust SPD handling and
    a loss floor on the self term to prevent trace blow-ups when Φ̃≈0.

    New config knobs (optional; defaults shown):
      - self_loss_floor_rel: 1e-2     # relative to per-pixel var(Σ_q)
      - self_loss_floor_abs: 1e-8     # absolute floor
      - self_morphism_norm_tol: 1e-3  # treat Φ̃ as "tiny" if ‖Φ̃‖_F < tol at pixel
    """
    import numpy as np
    from transport.transport_cache import Phi as tc_Phi

    eps    = _cfg_req(ctx, "energy_eps")
    tau    = _cfg_req(ctx, "support_cutoff_eps")
    w_self = _cfg_req(ctx, "alpha")
    w_fb   = _cfg_req(ctx, "feedback_weight")

    # Optional config with safe fallbacks whether ctx.config is a dict or object
    cfg = getattr(ctx, "config", {}) or {}
    _get = (cfg.get if hasattr(cfg, "get") else lambda k, d: getattr(cfg, k, d))
    sigma_rel = float(_get("self_loss_floor_rel", 1e-2))
    sigma_abs = float(_get("self_loss_floor_abs", 1e-8))
    norm_tol  = float(_get("self_morphism_norm_tol", 1e-3))

    # Active support
    mask = _mask_bool(agent.mask, tau)
    H, W = mask.shape
    if not mask.any():
        z = np.zeros((H, W), np.float32)
        out = {"self": z, "feedback": z}
        if return_components:
            out["self_components"]    = {k: z for k in ("quad", "trace", "logdet")}
            out["feedback_components"] = {k: z for k in ("quad", "trace", "logdet")}
        return out

    # Fields
    mu_q = np.asarray(agent.mu_q_field,    np.float32)  # (H,W,Kq)
    S_q  = np.asarray(agent.sigma_q_field, np.float32)  # (H,W,Kq,Kq)
    mu_p = np.asarray(agent.mu_p_field,    np.float32)  # (H,W,Kp)
    S_p  = np.asarray(agent.sigma_p_field, np.float32)  # (H,W,Kp,Kp)

    H_, W_, Kq = mu_q.shape
    _,  _,  Kp = mu_p.shape
    if S_q.shape != (H_, W_, Kq, Kq) or S_p.shape != (H_, W_, Kp, Kp):
        raise ValueError(f"Sigma shapes do not match mu shapes: "
                         f"S_q={S_q.shape}, mu_q={mu_q.shape}; S_p={S_p.shape}, mu_p={mu_p.shape}")

    # Morphisms (global or per-pixel)
    Phi_qp = tc_Phi(ctx, agent, kind="q_to_p")  # q→p  (Kp,Kq) or (H,W,Kp,Kq)
    Phi_pq = tc_Phi(ctx, agent, kind="p_to_q")  # p→q  (Kq,Kp) or (H,W,Kq,Kp)

    # Gather active pixels
    rrcc = np.argwhere(mask)
    r, c = rrcc[:, 0], rrcc[:, 1]

    # Slice & sanitize base covariances
    mu_q_m = mu_q[r, c]                                # (M,Kq)
    S_q_m  = _spd_sym(np.asarray(S_q[r, c], np.float32), eps)  # (M,Kq,Kq)
    mu_p_m = mu_p[r, c]                                # (M,Kp)
    S_p_m  = _spd_sym(np.asarray(S_p[r, c], np.float32), eps)  # (M,Kp,Kp)

    # Materialize morphisms for these pixels -> (M,Kout,Kin)
    M_qp = _materialize_morphism(Phi_qp, H, W, Kp, Kq, r, c)   # q→p
    M_pq = _materialize_morphism(Phi_pq, H, W, Kq, Kp, r, c)   # p→q

    # ===================== SELF: KL(q || Φ̃·p) =====================
    # Mean + covariance push
    mu_p_push_q = np.einsum("...ik,...k->...i", M_pq, mu_p_m, optimize=True)            # (M,Kq)
    S_p_push_q  = np.einsum("...ik,...kl,...jl->...ij", M_pq, S_p_m, M_pq, optimize=True)  # (M,Kq,Kq)

    # Rank/norm guard for Φ̃: if ‖Φ̃‖_F is tiny, zero the mean push (prevents quad noise)
    Mnorm = np.linalg.norm(M_pq, axis=(-2, -1))  # (M,)
    tiny  = Mnorm < norm_tol
    if tiny.any():
        mu_p_push_q[tiny] = 0.0

    # Loss floor R = σ^2 I on the pushed covariance (well-posed even if Φ̃≈0)
    # Scale σ^2 by per-pixel variance of Σ_q to keep units sane
    var_q = np.trace(S_q_m, axis1=-2, axis2=-1) / float(Kq)   # (M,)
    sigma2 = (sigma_rel * var_q + sigma_abs).astype(np.float32)  # (M,)
    Iq = np.eye(Kq, dtype=np.float32)
    S_p_push_q = S_p_push_q + sigma2[:, None, None] * Iq
    S_p_push_q = _spd_sym(S_p_push_q, eps)

    # KL components
    comp_self = _gaussian_kl_components(mu_q_m, S_q_m, mu_p_push_q, S_p_push_q, eps=eps)
    self_vals = (w_self * comp_self["kl"]).astype(np.float32)

    # Scatter into (H,W)
    self_map = np.zeros((H, W), np.float32)
    self_map[r, c] = self_vals

    out = {"self": self_map}
    if return_components:
        self_comp = {k: np.zeros((H, W), np.float32) for k in ("quad", "trace", "logdet")}
        for k in ("quad", "trace", "logdet"):
            self_comp[k][r, c] = (w_self * comp_self[k]).astype(np.float32)
        out["self_components"] = self_comp

    # ===================== FEEDBACK: KL(p || Φ·q) =====================
    if w_fb <= 0.0:
        fb_map = np.zeros((H, W), np.float32)
        out["feedback"] = fb_map
        if return_components:
            out["feedback_components"] = {k: fb_map for k in ("quad", "trace", "logdet")}
        return out

    mu_q_push_p = np.einsum("...ik,...k->...i", M_qp, mu_q_m, optimize=True)            # (M,Kp)
    S_q_push_p  = np.einsum("...ik,...kl,...jl->...ij", M_qp, S_q_m, M_qp, optimize=True)  # (M,Kp,Kp)
    S_q_push_p  = _spd_sym(S_q_push_p, eps)

    comp_fb = _gaussian_kl_components(mu_p_m, S_p_m, mu_q_push_p, S_q_push_p, eps=eps)
    fb_vals = (w_fb * comp_fb["kl"]).astype(np.float32)

    fb_map = np.zeros((H, W), np.float32)
    fb_map[r, c] = fb_vals
    out["feedback"] = fb_map

    if return_components:
        fb_comp = {k: np.zeros((H, W), np.float32) for k in ("quad", "trace", "logdet")}
        for k in ("quad", "trace", "logdet"):
            fb_comp[k][r, c] = (w_fb * comp_fb[k]).astype(np.float32)
        out["feedback_components"] = fb_comp

    return out



import numpy as np
from transport.transport_cache import Phi as tc_Phi

def _eigvals_sym(A):
    A = 0.5 * (A + A.swapaxes(-1, -2))
    try:
        w = np.linalg.eigvalsh(A)
    except np.linalg.LinAlgError:
        w = np.linalg.eigvals(A).real
    return w

def _cond_sym(A, eps=1e-30):
    w = _eigvals_sym(A)
    w = np.clip(w, eps, None)
    return float(w.max() / w.min())

def _at(agent, r, c):
    mu_q = np.asarray(agent.mu_q_field,  np.float32)[r, c]
    S_q  = np.asarray(agent.sigma_q_field, np.float32)[r, c]
    mu_p = np.asarray(agent.mu_p_field,  np.float32)[r, c]
    S_p  = np.asarray(agent.sigma_p_field, np.float32)[r, c]
    return mu_q, S_q, mu_p, S_p

def _materialize_single(M, H, W, r, c):
    # Accepts either global (Kout,Kin) or per-pixel (H,W,Kout,Kin)
    if M.ndim == 2:
        return M
    return np.asarray(M[r, c], np.float32)

def inspect_self_hotspots(ctx, agent, *, topk=5):
    """Print detailed diagnostics for the top-k self-energy pixels of one agent."""
    maps = compute_self_feedback_maps(ctx, agent, return_components=True)
    Smap = maps["self"]
    tau  = _cfg_req(ctx, "support_cutoff_eps")
    mask = _mask_bool(agent.mask, tau)
    if not mask.any():
        print(f"[inspect] aid={getattr(agent,'id','?')} no active pixels")
        return

    rrcc = np.argwhere(mask)
    vals = Smap[mask]
    k = min(topk, vals.size)
    idx = np.argpartition(vals, -k)[-k:]
    idx = idx[np.argsort(-vals[idx])]
    print(f"[inspect] aid={getattr(agent,'id','?')} top{ k } self-energy pixels:")

    # Get morphism (q<-p) used by the self term
    H, W = mask.shape
    M_pq_full = tc_Phi(ctx, agent, kind="p_to_q")  # (Kq,Kp) or (H,W,Kq,Kp)

    for j in idx:
        r, c = rrcc[j]
        mu_q, S_q, mu_p, S_p = _at(agent, r, c)

        M_pq = _materialize_single(M_pq_full, H, W, r, c)  # Kq x Kp
        mu_p_push_q = M_pq @ mu_p
        S_p_push_q  = M_pq @ S_p @ M_pq.T

        # Components already computed earlier; pull them for this pixel
        comps = maps["self_components"]
        quad   = float(comps["quad"][r, c])
        trace  = float(comps["trace"][r, c])
        logdet = float(comps["logdet"][r, c])
        tot    = float(Smap[r, c])

        cn_Sq  = _cond_sym(S_q)
        cn_Spq = _cond_sym(S_p_push_q)
        mu_q_n = float(np.linalg.norm(mu_q))
        mu_pn  = float(np.linalg.norm(mu_p_push_q))
        dmun   = float(np.linalg.norm(mu_p_push_q - mu_q))
        Mn     = float(np.linalg.norm(M_pq))

        print(f"  ({int(r)}, {int(c)}): self={tot:.4e} | quad={quad:.4e} trace={trace:.4e} logdet={logdet:.4e}")
        print(f"              ‖μ_q‖={mu_q_n:.3e} ‖Φ̃μ_p‖={mu_pn:.3e} ‖Δμ‖={dmun:.3e}  ‖Φ̃‖_F={Mn:.3e}")
        print(f"              cond(Σ_q)={cn_Sq:.3e}  cond(Φ̃Σ_pΦ̃ᵀ)={cn_Spq:.3e}")

        # Row/col swap probe (quick litmus for indexing/reflection bugs)
        if 0 <= c < H and 0 <= r < W and (r != c):
            mu_q_rc, S_q_rc, mu_p_rc, S_p_rc = _at(agent, c, r)
            M_pq_rc = _materialize_single(M_pq_full, H, W, c, r)
            mu_p_push_q_rc = M_pq_rc @ mu_p_rc
            S_p_push_q_rc  = M_pq_rc @ S_p_rc @ M_pq_rc.T

            # Very rough KL approximation: only the quadratic piece (cheap sanity)
            try:
                from numpy.linalg import inv
                quad_rc = float((mu_p_push_q_rc - mu_q_rc).T @ inv(S_p_push_q_rc) @ (mu_p_push_q_rc - mu_q_rc))
                print(f"              swap-probe (c,r)=({int(c)},{int(r)}) quad≈{quad_rc:.3e}")
            except Exception:
                print(f"              swap-probe (c,r)=({int(c)},{int(r)}) quad≈<failed>")

def correlate_self_with_phi(ctx, agent):
    """Pearson r between self map and ‖phi‖ at same pixels; helps spot gauge-driven spikes."""
    tau  = _cfg_req(ctx, "support_cutoff_eps")
    mask = _mask_bool(agent.mask, tau)
    maps = compute_self_feedback_maps(ctx, agent, return_components=False)
    S = maps["self"][mask].astype(np.float64)

    phi = getattr(agent, "phi_field", None) or getattr(agent, "phi", None)
    if phi is None:
        print("[corr] no phi field on agent")
        return
    phin = np.linalg.norm(np.asarray(phi, np.float64), axis=-1)[mask]
    if S.size < 2 or phin.std() == 0 or S.std() == 0:
        print("[corr] insufficient variance for correlation")
        return
    r = float(np.corrcoef(S, phin)[0,1])
    print(f"[corr] aid={getattr(agent,'id','?')} corr(self, ‖phi‖) = {r:.3f}")


def log_self_hotspots(ctx, agents, *, topk=10, percentile=(50,90,99), save_npz=False, outdir="checkpoints"):
    """
    Print concise diagnostics showing whether giant energy comes from a handful
    of pixels or from a broad field. Optionally saves NPZ maps per agent.
    """
    def _stats(v):
        v = v[np.isfinite(v)]
        if v.size == 0:
            return {}
        q = np.percentile(v, np.array(percentile, dtype=np.float32))
        return {f"p{p}": float(qi) for p, qi in zip(percentile, q)} | {"max": float(v.max()), "sum": float(v.sum())}

    for a in agents:
        maps = compute_self_feedback_maps(ctx, a, return_components=True)
        S = maps["self"]; F = maps["feedback"]
        mask = _mask_bool(a.mask, _cfg_req(ctx, "support_cutoff_eps"))
        vals = S[mask]

        if not vals.size:
            print(f"[hotspots] aid={getattr(a,'id','?')} no active pixels")
            continue

        flat_idx = np.argpartition(vals, -min(topk, vals.size))[-min(topk, vals.size):]
        rrcc = np.argwhere(mask)
        rows, cols = rrcc[flat_idx, 0], rrcc[flat_idx, 1]
        tops = vals[flat_idx]
        order = np.argsort(-tops)
        rows, cols, tops = rows[order], cols[order], tops[order]

        s_stats = _stats(vals)
        f_stats = _stats(F[mask])
        share = float(tops.sum() / (vals.sum() + 1e-30))

        print(f"[hotspots] aid={getattr(a,'id','?')} | self_sum={s_stats.get('sum',0):.4e} "
              f"| top{len(tops)} share={share:.3f} | "
              + " ".join([f"{k}={v:.3e}" for k,v in s_stats.items() if k!='sum']))

        for i in range(len(tops)):
            print(f"  -> ({int(rows[i])}, {int(cols[i])}) self={tops[i]:.4e}")

        if save_npz:
            import os
            os.makedirs(outdir, exist_ok=True)
            np.savez_compressed(
                os.path.join(outdir, f"self_maps_a{getattr(a,'id','x')}_step{int(getattr(ctx,'global_step',-1))}.npz"),
                self=S, feedback=F,
                self_quad=maps["self_components"]["quad"],
                self_trace=maps["self_components"]["trace"],
                self_logdet=maps["self_components"]["logdet"],
            )

















def _energy_mass(ctx: Any, agent: Any) -> Tuple[float, float, float, float]:
    """Returns (mass_q_sum, mass_q_mean, mass_p_sum, mass_p_mean)."""
    tau = _cfg_req(ctx, "support_cutoff_eps")
    wq  = _cfg_req(ctx, "belief_mass")
    wp  = _cfg_req(ctx, "model_mass")
    
    mask = _mask_bool(agent.mask, tau)
    if not mask.any():
        return 0.0, 0.0, 0.0, 0.0

    phi_q = np.asarray(agent.phi, np.float32)       # (H,W,3)
    phi_p = np.asarray(agent.phi_model, np.float32) # (H,W,3)
    if phi_q.ndim != 3 or phi_q.shape[-1] != 3 or phi_p.ndim != 3 or phi_p.shape[-1] != 3:
        raise ValueError("phi_field/phi_model_field must be (H,W,3)")

    m_q = (phi_q ** 2).sum(axis=-1)  # ||Ï†||Â²
    m_p = (phi_p ** 2).sum(axis=-1)
    return (
        wq * _reduce(m_q, mask, "sum"),
        wq * _reduce(m_q, mask, "mean"),
        wp * _reduce(m_p, mask, "sum"),
        wp * _reduce(m_p, mask, "mean"),
    )

def _pick_generators(agent, names):
    for nm in names:
        G = getattr(agent, nm, None)
        if G is not None:
            return G
    return None

def _energy_curvature(ctx: Any, agent: Any) -> Tuple[float, float, float, float]:
    """
    Returns (curv_q_sum, curv_q_mean, curv_p_sum, curv_p_mean).

    Backend: lattice plaquette via transport_cache.plaquette_product(...)
      P ∈ ℝ^{H×W×K×K}, site magnitude = ||P - I||_F^2
    """
    import numpy as np

    tau = _cfg_req(ctx, "support_cutoff_eps")
    wq  = _cfg_req(ctx, "curvature_weight")
    wp  = _cfg_req(ctx, "model_curvature_weight")

    mask = _mask_bool(agent.mask, tau)
    if not mask.any():
        return 0.0, 0.0, 0.0, 0.0

    if tc_plaquette is None:
        raise ImportError("transport_cache.plaquette_product is unavailable")

    # ---- belief side (q-fiber) ----
    Gq = _pick_generators(agent, ("generators_q", "G_q"))
    if Gq is None:
        Gq = _guess_generators(agent, "q")
    Gq = np.asarray(Gq, np.float32)

    Pq = np.asarray(
        tc_plaquette(ctx, agent.phi, Gq, boundary="periodic", split_edge=True),
        np.float32,
    )  # (H,W,Kq,Kq)

    if Pq.ndim != 4 or Pq.shape[-1] != Pq.shape[-2]:
        raise ValueError(f"Invalid plaquette (q) shape {Pq.shape}")

    Kq = int(Pq.shape[-1])
    Iq = np.eye(Kq, dtype=np.float32)
    fq = np.linalg.norm(Pq - Iq, ord="fro", axis=(-2, -1)) ** 2  # (H,W)

    # ---- model side (p-fiber) ----
    Gp = _pick_generators(agent, ("generators_p", "G_p"))
    if Gp is None:
        Gp = _guess_generators(agent, "p")
    Gp = np.asarray(Gp, np.float32)

    Pp = np.asarray(
        tc_plaquette(ctx, agent.phi_model, Gp, boundary="periodic", split_edge=True),
        np.float32,
    )  # (H,W,Kp,Kp)

    if Pp.ndim != 4 or Pp.shape[-1] != Pp.shape[-2]:
        raise ValueError(f"Invalid plaquette (p) shape {Pp.shape}")

    Kp = int(Pp.shape[-1])
    Ip = np.eye(Kp, dtype=np.float32)
    fp = np.linalg.norm(Pp - Ip, ord="fro", axis=(-2, -1)) ** 2  # (H,W)

    return (
        wq * _reduce(fq, mask, "sum"),
        wq * _reduce(fq, mask, "mean"),
        wp * _reduce(fp, mask, "sum"),
        wp * _reduce(fp, mask, "mean"),
    )


# ——— Variational energy API ————————————————————————————————————————————————
# Single-source-of-truth scalar objective + optional breakdown.
# Uses the same internals as your printed metrics.



def total_energy(ctx: Any, agents: Any, *, return_breakdown: bool = False
                 ) -> float | Tuple[float, Dict[str, float]]:
    """
    Variational energy minimized by the suite.
    Returns:
      - float E (if return_breakdown=False)
      - (E, breakdown_dict) otherwise
    """
    # compute_global_metrics aggregates:
    #   self_sum, feedback_sum, mass_q_sum, mass_p_sum,
    #   curv_q_sum, curv_p_sum, align_q_sum, align_p_sum,
    # and sets E_total = sum of those terms.
    M = compute_global_metrics(ctx, agents)  # ← already defined above
    E = float(M.get("E_total", 0.0))

    if not return_breakdown:
        return E

    # Keep a stable, documented breakdown
    breakdown = {
        "self":      float(M.get("self_sum",     0.0)),
        "feedback":  float(M.get("feedback_sum", 0.0)),
        "align_q":   float(M.get("align_q_sum",  0.0)),
        "align_p":   float(M.get("align_p_sum",  0.0)),
        "mass_q":    float(M.get("mass_q_sum",   0.0)),
        "mass_p":    float(M.get("mass_p_sum",   0.0)),
        "curv_q":    float(M.get("curv_q_sum",   0.0)),
        "curv_p":    float(M.get("curv_p_sum",   0.0)),
        "mean_total":float(M.get("mean_total",   0.0)),  # optional convenience
    }
    return E, breakdown


def assert_energy_finite(ctx: Any, agents: Any, *, allow_zero: bool = True) -> None:
    """
    Safety check for CI/runs: energy must be finite (and non-negative unless disabled).
    """
    E, terms = total_energy(ctx, agents, return_breakdown=True)
    if not np.isfinite(E):
        raise ValueError(f"total_energy is not finite: {E}")
    if (not allow_zero) and E <= 0.0:
        raise ValueError(f"total_energy must be positive, got {E}")
    for name, val in terms.items():
        if name == "mean_total": 
            continue
        if not np.isfinite(val):
            raise ValueError(f"energy term '{name}' is not finite: {val}")
        if val < 0.0:
            # Most terms are ≥0 by construction; relax here if you ever add signed terms.
            print(f"[warn] energy term '{name}' < 0 ({val}); verify this is expected.")



def compute_agent_metrics(ctx: Any, agent: Any) -> Dict[str, float]:
    self_sum, self_mean, fb_sum, fb_mean = _energy_self_feedback(ctx, agent)
    mass_q_sum, mass_q_mean, mass_p_sum, mass_p_mean = _energy_mass(ctx, agent)
    curv_q_sum, curv_q_mean, curv_p_sum, curv_p_mean = _energy_curvature(ctx, agent)
    return {
        "self_sum": self_sum, "self_mean": self_mean,
        "feedback_sum": fb_sum, "feedback_mean": fb_mean,
        "mass_q_sum": mass_q_sum, "mass_q_mean": mass_q_mean,
        "mass_p_sum": mass_p_sum, "mass_p_mean": mass_p_mean,
        "curv_q_sum": curv_q_sum, "curv_q_mean": curv_q_mean,
        "curv_p_sum": curv_p_sum, "curv_p_mean": curv_p_mean,
    }


def compute_global_metrics(ctx: Any, agents: Sequence[Any]) -> Dict[str, float]:
    tau = _cfg_req(ctx, "support_cutoff_eps")
    totals = {
        "self_sum": 0.0, "feedback_sum": 0.0,
        "mass_q_sum": 0.0, "mass_p_sum": 0.0,
        "curv_q_sum": 0.0, "curv_p_sum": 0.0,
        "_S": 0,
    }
    for a in agents:
        m = _mask_bool(a.mask, tau)
        S = _active_px(m)
        if S <= 0:
            continue
        d = compute_agent_metrics(ctx, a)
        for k in ("self_sum", "feedback_sum", "mass_q_sum", "mass_p_sum", "curv_q_sum", "curv_p_sum"):
            totals[k] += float(d[k])
        totals["_S"] += S

        # alignment (computed once across agents; uses neighbors)
    align_q_sum, align_q_mean, align_p_sum, align_p_mean = _energy_alignment(ctx, agents)

    S_tot = max(totals["_S"], 1)
    out = dict(totals)

    # base means
    out["self_mean"]     = out["self_sum"]     / S_tot
    out["feedback_mean"] = out["feedback_sum"] / S_tot
    out["mass_q_mean"]   = out["mass_q_sum"]   / S_tot
    out["mass_p_mean"]   = out["mass_p_sum"]   / S_tot
    out["curv_q_mean"]   = out["curv_q_sum"]   / S_tot
    out["curv_p_mean"]   = out["curv_p_sum"]   / S_tot

    # alignment terms
    out["align_q_sum"]  = align_q_sum
    out["align_q_mean"] = align_q_mean
    out["align_p_sum"]  = align_p_sum
    out["align_p_mean"] = align_p_mean

    # totals including alignment
    out["E_total"] = float(
        out["self_sum"] + out["feedback_sum"] +
        out["mass_q_sum"] + out["mass_p_sum"] +
        out["curv_q_sum"] + out["curv_p_sum"] +
        out["align_q_sum"] + out["align_p_sum"]
    )
    out["mean_total"] = float(
        out["self_mean"] + out["feedback_mean"] +
        out["mass_q_mean"] + out["mass_p_mean"] +
        out["curv_q_mean"] + out["curv_p_mean"] +
        out["align_q_mean"] + out["align_p_mean"]
    )
    return out



def print_energy_breakdown(ctx, metrics_global, *, stream=None) -> None:
    step = int(getattr(ctx, "global_step", -1))
    w_self = _cfg_req(ctx, "alpha")
    w_fb   = _cfg_req(ctx, "feedback_weight") if "feedback_weight" in ctx.config else 0.0
    w_mq   = _cfg_req(ctx, "belief_mass")
    w_mp   = _cfg_req(ctx, "model_mass")
    w_cq   = _cfg_req(ctx, "curvature_weight")
    w_cp   = _cfg_req(ctx, "model_curvature_weight")
    w_aq   = _cfg_req(ctx, "beta")         # alignment q
    w_ap   = _cfg_req(ctx, "beta_model")   # alignment p
    lines = []
    lines.append(f"[Energy Totals @ step {step}]")
    lines.append(f"  E_total = {metrics_global.get('E_total', 0.0):.6e}  |  mean_total = {metrics_global.get('mean_total', 0.0):.6e}")
    lines.append("  ----------------------------------------------------")
    lines.append("  term         weighted_sum            mean_per_px        weight")
    lines.append(f"  self         {metrics_global.get('self_sum', 0.0):>14.6e}    {metrics_global.get('self_mean', 0.0):>14.6e}    {w_self:g}")
    lines.append(f"  align_q      {metrics_global.get('align_q_sum', 0.0):>14.6e}    {metrics_global.get('align_q_mean', 0.0):>14.6e}    {w_aq:g}")
    lines.append(f"  align_p      {metrics_global.get('align_p_sum', 0.0):>14.6e}    {metrics_global.get('align_p_mean', 0.0):>14.6e}    {w_ap:g}")
    lines.append(f"  feedback     {metrics_global.get('feedback_sum', 0.0):>14.6e}    {metrics_global.get('feedback_mean', 0.0):>14.6e}    {w_fb:g}")
    lines.append(f"  mass_q       {metrics_global.get('mass_q_sum', 0.0):>14.6e}    {metrics_global.get('mass_q_mean', 0.0):>14.6e}    {w_mq:g}")
    lines.append(f"  mass_p       {metrics_global.get('mass_p_sum', 0.0):>14.6e}    {metrics_global.get('mass_p_mean', 0.0):>14.6e}    {w_mp:g}")
    lines.append(f"  curv_q       {metrics_global.get('curv_q_sum', 0.0):>14.6e}    {metrics_global.get('curv_q_mean', 0.0):>14.6e}    {w_cq:g}")
    lines.append(f"  curv_p       {metrics_global.get('curv_p_sum', 0.0):>14.6e}    {metrics_global.get('curv_p_mean', 0.0):>14.6e}    {w_cp:g}")
    
    
    
    txt = "\n".join(lines)
    if callable(getattr(ctx, "log", None)): ctx.log(txt)
    else: print(txt, file=stream or None)



def snapshot_scalars(
    ctx: Any,
    agents: Sequence[Any],
    outdir: str | None,
    metrics_global: Mapping[str, float],
    *,
    step_tag: Optional[int] = None,
) -> Optional[None]:
    # Intentionally a no-op in Phase 2 (we keep it light).
    return None
