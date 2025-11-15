# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 10:07:56 2025

@author: chris and christine
"""
import numpy as np
from core.numerical_utils import sanitize_sigma, safe_inv, safe_logdet
from typing import Optional, Tuple

import core.config
from core.numerical_utils import sanitize_sigma, safe_inv, safe_logdet
from typing import Optional
from core.omega import compute_field_strength  
from transport.transport_cache import omega_child_to_parents_batched, Omega
from core.omega import exp_lie_algebra_irrep


# -*- coding: utf-8 -*-
"""
Linear pushforward of Gaussian parameters under a matrix M:
  μ' = M μ
  Σ' = M Σ Mᵀ

One canonical entry-point:
  push_gaussian(mu, Sigma, M, *, approx="auto"| "exact" | "first_order", ...)

- Cache-agnostic; pure math.
- Works for square (Ω) and rectangular (Φ) maps.
- Broadcast-safe over leading dims.

Use this from transport/updates/energies; DO NOT re-implement locally.
"""

def _as_float(x):
    return np.asarray(x, dtype=float)


def _expand_diag(diag_or_full: np.ndarray, *, assume_stddev: bool) -> np.ndarray:
    """Convert (...,K) or (...,K,K) to full SPD (...,K,K)."""
    S = np.asarray(diag_or_full)
    if S.ndim >= 2 and S.shape[-2] == S.shape[-1]:
        return S
    if S.ndim == 1:
        S = S[None]
    diag = S
    if assume_stddev:
        diag = diag * diag
    K = int(diag.shape[-1])
    I = np.eye(K, dtype=diag.dtype)
    return np.einsum("...k,ij->...ij", diag, I, optimize=True)


def kl_gaussian(
    mu1: np.ndarray,
    S1: np.ndarray,
    mu2: np.ndarray,
    S2: np.ndarray,
    *,
    eps: float = 1e-6,
    allow_diag: bool = True,
    diag_is_stddev: bool = True,
    sanitize: bool = True,
) -> np.ndarray:
    """KL(N(μ1,Σ1) || N(μ2,Σ2)) in closed form; broadcast-safe.

    Accepts diagonal covariances if allow_diag=True (interprets as stddevs if
    diag_is_stddev=True), otherwise requires full SPD.
    """
    mu1 = _as_float(mu1)
    mu2 = _as_float(mu2)
    S1 = np.asarray(S1)
    S2 = np.asarray(S2)

    if allow_diag:
        S1 = _expand_diag(S1, assume_stddev=diag_is_stddev)
        S2 = _expand_diag(S2, assume_stddev=diag_is_stddev)

    if sanitize:
        S1 = sanitize_sigma(0.5 * (S1 + np.swapaxes(S1, -1, -2)), eps=eps)
        S2 = sanitize_sigma(0.5 * (S2 + np.swapaxes(S2, -1, -2)), eps=eps)

    K = int(mu1.shape[-1])
    S2_inv = safe_inv(S2, eps=eps)

    # tr(S2^{-1} S1)
    term_trace = np.einsum("...ij,...ji->...", S2_inv, S1, optimize=True)
    # (μ2-μ1)^T S2^{-1} (μ2-μ1)
    dmu = mu2 - mu1
    term_quad = np.einsum("...i,...ij,...j->...", dmu, S2_inv, dmu, optimize=True)

    # log|Σ2|/|Σ1|
    sign1, logdet1 = np.linalg.slogdet(S1)
    sign2, logdet2 = np.linalg.slogdet(S2)
    # If sanitize_sigma succeeded, signs should be +1; fall back to safe_logdet
    bad = (sign1 < 0) | (sign2 < 0)
    if np.any(bad):
        logdet1 = safe_logdet(S1, eps=eps)
        logdet2 = safe_logdet(S2, eps=eps)
    logdet_ratio = logdet2 - logdet1

    kl = 0.5 * (term_trace + term_quad - K + logdet_ratio)
    return np.clip(np.nan_to_num(kl, nan=0.0, posinf=1e6, neginf=0.0), 0.0, 1e6)


# Back-compat wrapper (keeps "mask" arg and N×K convention)

def kl_divergence(mu1, sigma1, mu2, sigma2, mask: Optional[np.ndarray] = None, eps: float = 1e-6):
    """Legacy name. Returns KL per batch; optional elementwise mask multiply."""
    vals = kl_gaussian(mu1, sigma1, mu2, sigma2, eps=eps)
    if mask is not None:
        m = np.asarray(mask)
        if m.ndim == 2 and m.shape[-1] == 1:
            m = np.squeeze(m, axis=-1)
        vals = vals * m
    if np.any(~np.isfinite(vals)):
        raise ValueError("[kl_divergence] NaN/Inf in output")
    return vals




def _validate_dims(mu: np.ndarray, Sigma: np.ndarray, M: np.ndarray) -> Tuple[int,int,int]:
    mu = np.asarray(mu); Sigma = np.asarray(Sigma); M = np.asarray(M)
    K_in = int(mu.shape[-1])
    if Sigma.shape[-2:] != (K_in, K_in):
        raise ValueError(f"[push_gaussian] Σ dims {Sigma.shape[-2:]} incompatible with μ dim {K_in}")
    if M.shape[-1] != K_in:
        raise ValueError(f"[push_gaussian] M last dim {M.shape[-1]} != μ dim {K_in}")
    K_out = int(M.shape[-2])
    return K_in, K_out, mu.ndim

def _near_identity_ok(M: np.ndarray, tau: float) -> bool:
    # Only meaningful if square; Frobenius norm check on (M - I)
    M = np.asarray(M)
    if M.shape[-2] != M.shape[-1]:
        return False
    K = M.shape[-1]
    I = np.eye(K, dtype=M.dtype)
    A = M - I
    # Compute per-batch Frobenius with broadcasting
    Anorm = np.sqrt(np.square(A).sum(axis=(-2, -1)))
    # “All” batches near identity: keep the rule strict to avoid mixed modes
    return bool(np.all(Anorm <= tau))

def push_gaussian(
    mu: np.ndarray,
    Sigma: np.ndarray,
    M: np.ndarray,
    *,
    eps: float = 1e-8,
    symmetrize: bool = True,
    sanitize: bool = True,
    approx: str = "auto",     # "auto" | "exact" | "first_order"
    near_identity_tau: Optional[float] = None,
    return_inv: bool = False,
):
    """
    Canonical Gaussian pushforward under a linear map.

    Args
    ----
    mu, Sigma : (..., K_in), (..., K_in, K_in)
    M        : (..., K_out, K_in)  (square for Ω; rectangular for Φ)
    eps      : jitter to keep Σ SPD
    symmetrize: force symmetry guards (recommended)
    sanitize : run sanitize_sigma at the end
    approx   : "exact", "first_order", or "auto"
    near_identity_tau: threshold for "auto" mode (Frobenius ||M-I|| ≤ tau) when square
    return_inv : also return Σ'^{-1} (exact path or first-order inverse if used)

    Returns
    -------
    (mu_out, Sigma_out) or (mu_out, Sigma_out, Sigma_out_inv)
    """
    mu = np.asarray(mu); Sigma = np.asarray(Sigma); M = np.asarray(M)
    K_in, K_out, _ = _validate_dims(mu, Sigma, M)

    # Symmetrize & pre-jitter once
    if symmetrize:
        Sigma = 0.5 * (Sigma + np.swapaxes(Sigma, -1, -2))
    Sigma = Sigma + eps * np.eye(K_in, dtype=Sigma.dtype)

    # Decide mode
    mode = approx
    if approx == "auto":
        tau = 5e-2 if near_identity_tau is None else float(near_identity_tau)
        mode = "first_order" if _near_identity_ok(M, tau) else "exact"
    elif approx not in ("exact", "first_order"):
        raise ValueError(f"[push_gaussian] unknown approx='{approx}'")

    # ---- FIRST-ORDER (only valid for square maps) ----------------------------
    if mode == "first_order":
        if K_in != K_out:
            # Can't do first-order on rectangular maps safely; fall back to exact
            mode = "exact"
        else:
            I = np.eye(K_in, dtype=M.dtype)
            A = M - I
            mu_out = np.einsum("...ik,...k->...i", A, mu, optimize=True) + mu
            A_S   = np.einsum("...ik,...kl->...il", A, Sigma, optimize=True)
            S_AT  = np.einsum("...ik,...jk->...ij", Sigma, A, optimize=True)  # ΣAᵀ
            Sigma_out = Sigma + A_S + S_AT
            if symmetrize:
                Sigma_out = 0.5 * (Sigma_out + np.swapaxes(Sigma_out, -1, -2))
            Sigma_out = Sigma_out + eps * np.eye(K_in, dtype=Sigma_out.dtype)
            if sanitize:
                Sigma_out = sanitize_sigma(Sigma_out, eps=eps)
            if return_inv:
                # First-order inverse: (Σ + AΣ + ΣAᵀ)^{-1} ≈ Σ^{-1} - Σ^{-1}A - AᵀΣ^{-1}
                Sigma_inv = safe_inv(Sigma, eps=1e-8)
                left  = np.einsum("...ij,...jk->...ik", Sigma_inv, A, optimize=True)
                right = np.einsum("...ij,...jk->...ik", np.swapaxes(A, -1, -2), Sigma_inv, optimize=True)
                Sigma_out_inv = Sigma_inv - left - right
                return mu_out.astype(mu.dtype), Sigma_out.astype(Sigma.dtype), Sigma_out_inv.astype(Sigma.dtype)
            return mu_out.astype(mu.dtype), Sigma_out.astype(Sigma.dtype)

    # ---- EXACT ---------------------------------------------------------------
    mu_out = np.einsum("...ik,...k->...i", M, mu, optimize=True)
    Sigma_out = np.einsum("...ik,...kl,...jl->...ij", M, Sigma, M, optimize=True)
    if symmetrize:
        Sigma_out = 0.5 * (Sigma_out + np.swapaxes(Sigma_out, -1, -2))
    Sigma_out = Sigma_out + eps * np.eye(K_out, dtype=Sigma_out.dtype)
    if sanitize:
        Sigma_out = sanitize_sigma(Sigma_out, eps=eps)
    if return_inv:
        Sigma_out_inv = safe_inv(Sigma_out, eps=1e-8)
        return mu_out.astype(mu.dtype), Sigma_out.astype(Sigma.dtype), Sigma_out_inv.astype(Sigma.dtype)
    return mu_out.astype(mu.dtype), Sigma_out.astype(Sigma.dtype)




# --- unified threshold resolver ---------------------------------------------------
def resolve_support_tau(ctx=None, params=None, default=1e-3, name="support_tau"):
    """
    Resolve a single support threshold used everywhere (detect/spawn/CG/viz).
    Order: params[name] > config.name > default. Persist into ctx for this step.
    """
    import core.config as CFG
    val = None
    if params and (name in params):
        val = float(params[name])
    elif hasattr(CFG, name):
        val = float(getattr(CFG, name))
    else:
        val = float(default)
    if ctx is not None:
        setattr(ctx, "_resolved_"+name, val)
    return val





def _overlap_pairs(children, H, W, *, tau=0.10, min_pair_px=8):
    """
    Return list of (i,j, overlap_bool_mask) for child pairs whose supports overlap.
    Uses (mask > tau) as support for overlap detection.
    """
    Ms_bool = [(np.asarray(c.mask, np.float32) > float(tau)) for c in children]
    pairs = []
    n = len(children)
    for i in range(n):
        mi = Ms_bool[i]
        if not mi.any(): continue
        for j in range(i+1, n):
            ov = (mi & Ms_bool[j])
            if int(ov.sum()) >= int(min_pair_px):
                pairs.append((i, j, ov))
    return pairs


def curvature_boltzmann_from_children(
    children, H, W, *, gamma=0.0, fiber="q", curvature_map=None,
    ctx: Optional[object] = None, G: Optional[np.ndarray] = None
):
    """
    Build BF(x) = exp(-gamma * F(x)).
    If curvature_map given, use it. Else compute F from compute_field_strength(phi, G).
    """
    if float(gamma) <= 0.0:
        return None

    if curvature_map is not None:
        F = np.asarray(curvature_map, np.float32)
        if F.shape == (H, W):
            return np.exp(-float(gamma) * np.clip(F, 0.0, np.finfo(np.float32).max)).astype(np.float32)



    # Require generators if we need to compute F; don't read from agents
    if G is None:
        return None

    num = np.zeros((H, W), np.float32)
    den = np.zeros((H, W), np.float32)
    phi_name = "phi" if str(fiber) == "q" else "phi_model"

    for c in children:
        m = np.asarray(getattr(c, "mask", 0.0), np.float32)
        if m.shape[:2] != (H, W) or float(m.max()) <= 0.0:
            continue
        phi = getattr(c, phi_name, None)
        if phi is None:
            continue
        Fdict = compute_field_strength(np.asarray(phi, np.float32), np.asarray(G, np.float32))
        Fxy = np.asarray(Fdict.get("xy", 0.0), np.float32)  # (H,W,K,K)
        if Fxy.shape[:2] != (H, W):
            continue
        Fx = np.sqrt(np.clip(np.sum(Fxy * Fxy, axis=(-2, -1)), 0.0, np.finfo(np.float32).max)).astype(np.float32)
        num += (m * Fx)
        den += np.maximum(m, 0.0)

    F = np.zeros((H, W), np.float32)
    sel = den > 1e-6
    F[sel] = (num[sel] / np.maximum(den[sel], 1e-6)).astype(np.float32)
    return np.exp(-float(gamma) * np.clip(F, 0.0, np.finfo(np.float32).max)).astype(np.float32)






def directed_kl_weight_after_transport(
    Ai, Aj,
    mu_i, S_i, mu_j, S_j,
    *,
    fiber, H, W, K, ov, tau,
    assume_diag_is_stddev: bool = True,
    debug: bool = False,
    ctx: Optional[object] = None,   # CacheHub entry point
    G: Optional[np.ndarray] = None, # Optional on fallback; central cache path ignores it
):
    """
    w(x) = exp( - KL( N_i || T_{j->i}(N_j) ) / tau ) on overlap ov.
    Transport Ω_ij is taken from central cache (preferred) or computed locally.
    """
    import numpy as np
    import core.config as config

    if not np.any(ov):
        return np.zeros((H, W), np.float32)

    # --- eps, overlap indices
    eps_log = float(getattr(config, "log_eps", 1e-8))
    iy, ix = np.nonzero(ov)

    # ---- slice means/covs on overlap (accept diagonal Σ as stddev if requested)
    mui = np.asarray(mu_i, np.float32)[iy, ix]   # (M, K)
    muj = np.asarray(mu_j, np.float32)[iy, ix]   # (M, K)

    Si  = np.asarray(S_i,  np.float32)[iy, ix]   # (M, K,K) or (M, K)
    Sj  = np.asarray(S_j,  np.float32)[iy, ix]

    if Si.ndim == 2:  # diagonal
        diag = Si if not assume_diag_is_stddev else Si * Si
        Si = np.einsum("mk,ij->mij", diag, np.eye(K, dtype=np.float32), optimize=True)
    if Sj.ndim == 2:
        diag = Sj if not assume_diag_is_stddev else Sj * Sj
        Sj = np.einsum("mk,ij->mij", diag, np.eye(K, dtype=np.float32), optimize=True)

    # Sanitize (SPD) before transport
    Si = sanitize_sigma(Si, eps=eps_log)
    Sj = sanitize_sigma(Sj, eps=eps_log)

    # --- choose φ fields
    if fiber == "q":
        phi_i_full = np.asarray(Ai.phi,       np.float32)
        phi_j_full = np.asarray(Aj.phi,       np.float32)
        gen_attr   = "generators_q"
    elif fiber == "p":
        phi_i_full = np.asarray(Ai.phi_model, np.float32)
        phi_j_full = np.asarray(Aj.phi_model, np.float32)
        gen_attr   = "generators_p"
    else:
        raise ValueError(f"fiber must be 'q' or 'p', got {fiber!r}")

    # ---------- Ω from central cache (preferred), else legacy fallback ----------
    if ctx is not None:
        # Centralized: Ω_{ij} = E_i @ inv(E_j), cached inside ctx.cache["omega"].
        # tc_Omega reads φ and generators directly from Ai/Aj.
        Omega_full = Omega(ctx, Ai, Aj, which=fiber)           # (H, W, K, K)
        Omega_ij   = Omega_full[iy, ix]                           # (M, K, K)
    else:
        # Fallback (no ctx): compute on-the-fly using explicit generators.
        # If G not provided, try reading from the agent (keeps old call sites working).
        if G is None:
            G = getattr(Ai, gen_attr, None)
        if G is None:
            raise ValueError(
                f"directed_kl_weight_after_transport: missing generators for fiber='{fiber}' (K={K}). "
                f"Pass G=... or set Ai.{gen_attr}."
            )

        # Compute Ω_ij over the overlapped points
        phi_i = phi_i_full[iy, ix]                                # (M, 3)
        phi_j = phi_j_full[iy, ix]                                # (M, 3)
        O_i      = exp_lie_algebra_irrep(phi_i,  np.asarray(G, np.float32))
        O_j_inv  = exp_lie_algebra_irrep(-phi_j, np.asarray(G, np.float32))
        Omega_ij = np.matmul(O_i, O_j_inv)                        # (M, K, K)

    if debug:
        RtR = np.einsum("mab,mac->mbc", Omega_ij, Omega_ij, optimize=True)
        I   = np.broadcast_to(np.eye(K, dtype=np.float32), RtR.shape)
        dev = np.max(np.abs(RtR - I))
        if dev > 1e-4:
            print(f"[DKL] non-orthogonal Ω on fiber='{fiber}' (max dev {dev:.2e})")

    # --- transport j → i on the overlap
    muj_t = np.einsum("mab,mb->ma", Omega_ij, muj, optimize=True)
    OmT   = np.swapaxes(Omega_ij, -1, -2)
    Sj_t  = np.einsum("mab,mbc,mdc->mad", Omega_ij, Sj, OmT, optimize=True)
    Sj_t  = sanitize_sigma(Sj_t, eps=eps_log)

    # --- per-pixel KL and weights
    d = kl_gaussian(mui, Si, muj_t, Sj_t, eps=eps_log).astype(np.float32)  # (M,)
    d = np.clip(np.nan_to_num(d, nan=0.0, posinf=1e6, neginf=0.0), 0.0, 1e6)

    invT = 1.0 / max(1e-8, float(tau))
    wvec = np.exp(-d * invT).astype(np.float32)

    w = np.zeros((H, W), np.float32)
    w[iy, ix] = wvec
    return w



def blended_spawn_metrics_child_vs_parents(
    ctx, child, parents, *, Gq=None, Gp=None,
    tau_q=0.45, tau_p=0.45, alpha=0.5, sm_tau=0.40, mask_tau=1e-3
):
    """
    Compute spawn caches for a child against many parents using
    omega_child_to_parents_batched (overlap-cropped, batched Ω).
    Returns dict with HxW fields:
      nb_count, min_kl_q/p, softmin_kl_q/p, cover_weight, A_final, KL_blend
    """
    
    import core.config as _cfg

    

    def _sanitize_sigma(S):
        try:
            from core.numerical_utils import sanitize_sigma as _ss
            return _ss(S, eps=float(getattr(_cfg, "log_eps", 1e-8)))
        except Exception:
            try:
                from numerical_utils import project_spd as _proj
                return _proj(S, eps=float(getattr(_cfg, "log_eps", 1e-8)))
            except Exception:
                return S

    # --- shapes & init ------------------------------------------------------------
    H, W = np.asarray(child.mask, np.float32).shape[:2]
    present_i   = (np.asarray(child.mask, np.float32) > float(mask_tau))
    nb_count     = np.zeros((H, W), np.int32)
    min_kl_q     = np.full((H, W), np.inf, np.float32)
    min_kl_p     = np.full((H, W), np.inf, np.float32)
    sumexp_q     = np.zeros((H, W), np.float32)
    sumexp_p     = np.zeros((H, W), np.float32)
    cover_weight = np.zeros((H, W), np.float32)
    tiny = 1e-9

    # child fields / availability
    mu_q_i = getattr(child, "mu_q_field", None)
    sg_q_i = getattr(child, "sigma_q_field", None)
    mu_p_i = getattr(child, "mu_p_field", None)
    sg_p_i = getattr(child, "sigma_p_field", None)
    have_q = (mu_q_i is not None) and (sg_q_i is not None) and (Gq is not None)
    have_p = (mu_p_i is not None) and (sg_p_i is not None) and (Gp is not None)

    # quick parent id map
    id2p = {int(getattr(p, "id", -1)): p for p in (parents or []) if p is not None}

    # --- Q fiber (batched Ω with overlap bboxes) ----------------------------------
    if have_q:
        packs_q = omega_child_to_parents_batched(ctx, child, parents, Gq, invert_j=True, mask_tau=mask_tau)
        for pid, (y0, y1, x0, x1), Ω in packs_q:
            p = id2p.get(int(pid))
            if p is None:
                continue
            jmask = np.asarray(getattr(p, "mask", 0.0), np.float32)[y0:y1, x0:x1]
            valid = present_i[y0:y1, x0:x1] & (jmask > float(mask_tau))
            if not np.any(valid):
                continue
            vy, vx = np.nonzero(valid)

            mu_i_win = np.asarray(mu_q_i, np.float32)[y0:y1, x0:x1, :]          # (h,w,K)
            S_i_win  = np.asarray(sg_q_i, np.float32)[y0:y1, x0:x1, :, :]
            mu_j_win = np.asarray(getattr(p, "mu_q_field"), np.float32)[y0:y1, x0:x1, :]
            S_j_win  = np.asarray(getattr(p, "sigma_q_field"), np.float32)[y0:y1, x0:x1, :, :]

            muj_t = np.einsum("...ab,...b->...a", Ω, mu_j_win, optimize=True)
            ΩT    = np.swapaxes(Ω, -1, -2)
            Sj_t  = np.einsum("...ab,...bc,...dc->...ad", Ω, S_j_win, ΩT, optimize=True)
            Sj_t  = _sanitize_sigma(Sj_t)

            kq_win = kl_gaussian(mu_i_win, S_i_win, muj_t, Sj_t).astype(np.float32)  # (h,w)

            np.add.at(nb_count[y0:y1, x0:x1], (vy, vx), 1)
            np.minimum.at(min_kl_q[y0:y1, x0:x1], (vy, vx), kq_win[vy, vx])
            np.add.at(sumexp_q[y0:y1, x0:x1], (vy, vx),
                      np.exp(-kq_win[vy, vx] / max(sm_tau, tiny)).astype(np.float32))
            np.add.at(cover_weight[y0:y1, x0:x1], (vy, vx), jmask[vy, vx])

    # --- P fiber (batched Ω with overlap bboxes) ----------------------------------
    if have_p:
        packs_p = omega_child_to_parents_batched(ctx, child, parents, Gp, invert_j=True, mask_tau=mask_tau)
        for pid, (y0, y1, x0, x1), Ω in packs_p:
            p = id2p.get(int(pid))
            if p is None:
                continue
            jmask = np.asarray(getattr(p, "mask", 0.0), np.float32)[y0:y1, x0:x1]
            valid = present_i[y0:y1, x0:x1] & (jmask > float(mask_tau))
            if not np.any(valid):
                continue
            vy, vx = np.nonzero(valid)

            mu_i_win = np.asarray(mu_p_i, np.float32)[y0:y1, x0:x1, :]
            S_i_win  = np.asarray(sg_p_i, np.float32)[y0:y1, x0:x1, :, :]
            mu_j_win = np.asarray(getattr(p, "mu_p_field"), np.float32)[y0:y1, x0:x1, :]
            S_j_win  = np.asarray(getattr(p, "sigma_p_field"), np.float32)[y0:y1, x0:x1, :, :]

            muj_t = np.einsum("...ab,...b->...a", Ω, mu_j_win, optimize=True)
            ΩT    = np.swapaxes(Ω, -1, -2)
            Sj_t  = np.einsum("...ab,...bc,...dc->...ad", Ω, S_j_win, ΩT, optimize=True)
            Sj_t  = _sanitize_sigma(Sj_t)

            kp_win = kl_gaussian(mu_i_win, S_i_win, muj_t, Sj_t).astype(np.float32)

            np.add.at(nb_count[y0:y1, x0:x1], (vy, vx), 1)
            np.minimum.at(min_kl_p[y0:y1, x0:x1], (vy, vx), kp_win[vy, vx])
            np.add.at(sumexp_p[y0:y1, x0:x1], (vy, vx),
                      np.exp(-kp_win[vy, vx] / max(sm_tau, tiny)).astype(np.float32))
            np.add.at(cover_weight[y0:y1, x0:x1], (vy, vx), jmask[vy, vx])

    # --- finalize softmins --------------------------------------------------------
    valid_any = (nb_count > 0)
    softmin_q = np.zeros_like(sumexp_q, np.float32)
    softmin_p = np.zeros_like(sumexp_p, np.float32)
    if np.any(valid_any):
        if np.any(sumexp_q):
            softmin_q[valid_any] = (-sm_tau * np.log(np.maximum(sumexp_q[valid_any], tiny))).astype(np.float32)
        if np.any(sumexp_p):
            softmin_p[valid_any] = (-sm_tau * np.log(np.maximum(sumexp_p[valid_any], tiny))).astype(np.float32)

    # --- blend and agreement ------------------------------------------------------
    have_q_any = np.isfinite(min_kl_q).any()
    have_p_any = np.isfinite(min_kl_p).any()
    if have_q_any and have_p_any:
        tau_blend = float(alpha) * float(tau_q) + (1.0 - float(alpha)) * float(tau_p)
        KL_blend  = (float(alpha) * softmin_q + (1.0 - float(alpha)) * softmin_p).astype(np.float32)
    elif have_q_any:
        tau_blend = float(tau_q)
        KL_blend  = softmin_q.astype(np.float32)
    elif have_p_any:
        tau_blend = float(tau_p)
        KL_blend  = softmin_p.astype(np.float32)
    else:
        tau_blend = 1.0
        KL_blend  = np.zeros((H, W), np.float32)

    A_final = np.zeros_like(KL_blend, np.float32)
    if np.any(valid_any):
        A_final[valid_any] = np.exp(-KL_blend[valid_any] / max(tau_blend, tiny)).astype(np.float32)
    A_final = np.clip(A_final, 0.0, 1.0)

    # --- NaN guards + min-KL invalid regions to +inf ------------------------------
    KL_blend  = np.nan_to_num(KL_blend,  copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    A_final   = np.nan_to_num(A_final,   copy=False, nan=0.0, posinf=1.0, neginf=0.0)
    softmin_q = np.nan_to_num(softmin_q, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    softmin_p = np.nan_to_num(softmin_p, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    min_kl_q  = np.where(valid_any, min_kl_q, np.inf).astype(np.float32)
    min_kl_p  = np.where(valid_any, min_kl_p, np.inf).astype(np.float32)

    return dict(
        nb_count=nb_count,
        min_kl_q=min_kl_q, min_kl_p=min_kl_p,
        softmin_q=softmin_q, softmin_p=softmin_p,
        cover_weight=cover_weight,
        A_final=A_final, KL_blend=KL_blend,
    )


def apply_evidence_modulators(w, Mi, Mj, ov, *, A=None, agree_power=1.0, BF=None):
    """
    Combine per-pair alignment weight with spatial terms:
      wij = Mi * Mj * w * 1_ov * (BF if provided) * (A^agree_power if provided)
    """
    wij = (Mi * Mj) * w
    wij *= ov.astype(np.float32)
    if BF is not None:
        wij *= BF
    if (A is not None) and (float(agree_power) != 0.0):
        wij *= (np.asarray(A, np.float32) ** float(agree_power))
    return wij.astype(np.float32, copy=False)




