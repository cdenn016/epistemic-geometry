
from __future__ import annotations
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple
import numpy as np

# Your existing utilities
from core.omega import exp_lie_algebra_irrep, retract_phi_principal
from core.numerical_utils import safe_omega_inv
import math
import core.config as CFG


"""
Cache Policy — Rule of One
1) One cache hub: runtime_ctx.cache (cleared once per step).
2) One facade to read/write: this file (E_grid, Omega, Jinv_grid, Phi, Phi_tilde, Lambda_*).
3) One warm point per step: ensure_dirty(...) — nowhere else.
Consumers must never pre-warm; they just call the facade.
"""

def _ctx_cfg(ctx):
    # Prefer ctx.config if you have one; otherwise fall back to CFG module
    src = getattr(ctx, "config", None)
    def _get(k, default=None):
        if src is not None and k in src:
            return src[k]
        return getattr(CFG, k, default)
    return {
        "group_name": _get("group_name", "so3"),
        "phi_clip": _get("phi_clip", 2.8),
        "periodic_wrap": _get("periodic_wrap", True),
        "so3_irreps_are_orthogonal": _get("so3_irreps_are_orthogonal", True),
        "mask_edge_soften_sigma": _get("mask_edge_soften_sigma", 0.0),
    }



def _sanitize_axis_angle(phi, *, max_theta=None, margin=None):
    """
    Sanitize an SO(3) axis-angle field φ (…,3):
      - NaN/Inf ⇒ 0
      - Retract to principal ball via retract_so3_principal(…, margin)
      - Optional hard cap: ||φ|| ← min(||φ||, max_theta) after retract

    Returns float32 with same leading shape as φ.
    """
    import math
    import numpy as np
    try:
        import core.config as config
        if margin is None:
            # Prefer a single canonical key, fall back for compatibility
            margin = float(getattr(config, "phi_principal_margin",
                         getattr(config, "so3_principal_margin", 1e-4)))
    except Exception:
        margin = 1e-4

    phi = np.asarray(phi, np.float32)
    phi = np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)

    # Non SO(3) case: pass through (still float32 + sanitized NaNs)
    if phi.shape[-1] != 3:
        return phi

    # 1) retract to principal ball
    out = retract_phi_principal(phi, margin=margin).astype(np.float32, copy=False)

    # 2) optional cap after retract
    if max_theta is not None:
        th = np.linalg.norm(out, axis=-1, keepdims=True)
        cap = float(min(float(max_theta), math.pi - margin))
        scale = np.minimum(1.0, cap / np.maximum(th, 1e-12)).astype(np.float32)
        out = (out * scale).astype(np.float32, copy=False)

    return out



# ---------------------------------------------------------------------------
# Internal helpers (keys, accessors)
# ---------------------------------------------------------------------------

def _aid(a: Any) -> int:
    """Robust agent id (falls back to object's id if missing)."""
    return int(getattr(a, "id", id(a)))

def _bases_key(agent) -> tuple[int, int, int]:
    aid = int(getattr(agent, "id", id(agent)))
    Kq = int(np.asarray(agent.mu_q_field).shape[-1])
    Kp = int(np.asarray(agent.mu_p_field).shape[-1])
    return (aid, Kq, Kp)

# --- drop-in replacement ---
def _morphism_key(agent, kind: str, cfg: dict, step_tag: int,
                  Kq: int, Kp: int, Phi0: Optional[np.ndarray]) -> tuple:
    import hashlib, json, numpy as np

    def _cfg_md5(c: dict) -> str:
        keys = ["group_name", "so3_irreps_are_orthogonal", "periodic_wrap", "phi_clip"]
        s = json.dumps({k: c.get(k, None) for k in keys},
                       sort_keys=True, separators=(",", ":"))
        return hashlib.md5(s.encode()).hexdigest()

    def _arr_digest(a: Optional[np.ndarray]) -> Optional[str]:
        if a is None:
            return None
        a = np.asarray(a, np.float32, order="C")
        h = hashlib.sha256()
        h.update(str(a.shape).encode())       # shape in the hash, just in case
        h.update(a.tobytes(order="C"))
        return h.hexdigest()

    # Pull current algebra fields
    phi_q = getattr(agent, "phi", None)          # belief field φ
    phi_p = getattr(agent, "phi_model", None)    # model  field φ̃

    # Which sides are used for this morphism?
    # Φ (q_to_p):   left=φ̃, right=φ
    # Φ̃ (p_to_q):  left=φ,  right=φ̃
    if kind == "q_to_p":
        left_phi, right_phi = phi_p, phi_q
    elif kind == "p_to_q":
        left_phi, right_phi = phi_q, phi_p
    else:
        raise ValueError(f"unknown morphism kind: {kind!r}")

    return ("morphism",
            int(getattr(agent, "id", id(agent))),
            str(kind),
            int(step_tag),
            (int(Kq), int(Kp)),
            _cfg_md5(cfg),
            _arr_digest(Phi0),
            _arr_digest(left_phi),
            _arr_digest(right_phi))



def _get_generators(agent: Any, which: str) -> np.ndarray:
    if which == "q":
        G = getattr(agent, "generators_q", None)
    else:
        G = getattr(agent, "generators_p", None)
    if G is None:
        raise ValueError(f"[transport_cache] generators ({which}) missing on agent id={_aid(agent)}")
    return G



def _get_phi(agent: Any, which: str) -> np.ndarray:
    if which == "q":
        phi = getattr(agent, "phi", None)
    else:
        phi = getattr(agent, "phi_model", None)
    if phi is None:
        raise ValueError(f"[transport_cache] phi field ({which}) missing on agent id={_aid(agent)}")
    return phi




# ---------------------------------------------------------------------------
#                              Caches
# ---------------------------------------------------------------------------

def E_grid(ctx: Any, agent: Any, which: str = "q", *, tol=None) -> np.ndarray:
    C = ctx.cache
    cfg = _ctx_cfg(ctx)
    step_tag = int(getattr(ctx, "global_step", -1))
    wrap_flag = bool(cfg.get("periodic_wrap", True))

    phi = _get_phi(agent, which)        # (...,3)
    G   = _get_generators(agent, which) # (3,K,K)

    key = C.key_E(agent_id=_aid(agent), step=step_tag, phi=phi,
                  cfg=cfg, wrap_flag=wrap_flag, tol=tol)
    E = C.get("exp", key)
    if E is not None:
        return E

    # (retain your existing sanitization & small-angle fast path)
    phi = _sanitize_axis_angle(phi)
    small_thr = float(getattr(CFG, "phi_small_threshold", 1e-6))
    if phi.shape[-1] == 3:
        th = np.linalg.norm(phi, axis=-1)
        if np.all(th < small_thr):
            K = int(G.shape[-2])
            I = np.eye(K, dtype=np.float32)
            E = np.broadcast_to(I, phi.shape[:-1] + (K, K)).copy()
            C.put("exp", key, E)
            return E

    max_norm = float(getattr(CFG, "exp_clip_max_norm", 10.0))
    E = exp_lie_algebra_irrep(phi, G, max_norm=max_norm)  # (...,K,K)
    C.put("exp", key, E)
    return E




def Omega(ctx: Any, ai: Any, aj: Any, which: str = "q", *, tol=None) -> np.ndarray:
    C = ctx.cache
    cfg = _ctx_cfg(ctx)
    step_tag = int(getattr(ctx, "global_step", -1))
    wrap_flag = bool(cfg.get("periodic_wrap", True))

    phi_i = _get_phi(ai, which)
    phi_j = _get_phi(aj, which)

    key = C.key_Omega(agent_i=_aid(ai), agent_j=_aid(aj), step=step_tag,
                      phi_i=phi_i, phi_j=phi_j, cfg=cfg, wrap_flag=wrap_flag, tol=tol)
    Om = C.get("omega", key)
    if Om is not None:
        return Om

    Ei = E_grid(ctx, ai, which=which, tol=tol)
    Ej = E_grid(ctx, aj, which=which, tol=tol)
    Ej_inv = safe_omega_inv(Ej)
    Om = np.matmul(Ei, Ej_inv)
    C.put("omega", key, Om)
    return Om




def Jinv_grid(ctx, agent, which="q", bbox=None, *, tol=None):
    C = ctx.cache
    cfg = _ctx_cfg(ctx)
    step_tag = int(getattr(ctx, "global_step", -1))
    wrap_flag = bool(cfg.get("periodic_wrap", True))

    phi = _get_phi(agent, which)
    key = C.key_Jinv(agent_id=_aid(agent), step=step_tag, phi=phi,
                     cfg=cfg, wrap_flag=wrap_flag, tol=tol)
    J = C.get("jinv", key)  # or use a dedicated 'jinv' namespace if you prefer
    if J is None:
        phi = _sanitize_axis_angle(phi)
        J = build_dexpinv_matrix(phi)  # (...,3,3)
        C.put("jinv", key, J)

    if bbox is None:
        return J
    y0,y1,x0,x1 = bbox
    return J[y0:y1, x0:x1, :, :]




# --- change Phi(...) to use the upgraded key and CacheHub get/put ---
def Phi(ctx, agent, kind: str = "q_to_p"):
    assert ctx is not None and hasattr(ctx, "cache"), "Phi: ctx with cache required"
    C   = ctx.cache
    cfg = _ctx_cfg(ctx)
    step_tag = int(getattr(ctx, "global_step", -1))

    Kq = int(np.asarray(agent.mu_q_field).shape[-1])
    Kp = int(np.asarray(agent.mu_p_field).shape[-1])

    Phi_0_in, Phi_tilde_0_in = get_morphism_bases(ctx, agent)
 
  

    # Build an appropriate key for the requested direction
    base0 = Phi_0_in if kind == "q_to_p" else Phi_tilde_0_in
    key   = _morphism_key(agent, kind, cfg, step_tag, Kq, Kp, base0)

    M = C.get("morphism", key)
    if M is not None:
        return M

    # Build both once (re-uses E cache)
    Phi_full, Phit_full = _build_morphisms_with_cache(
        ctx, agent, Phi_0_in, Phi_tilde_0_in, group_name=_infer_group_name()
    )

    # Store both with their own keys
    key_qp = _morphism_key(agent, "q_to_p", cfg, step_tag, Kq, Kp, Phi_0_in)
    key_pq = _morphism_key(agent, "p_to_q", cfg, step_tag, Kq, Kp, Phi_tilde_0_in)
    C.put("morphism", key_qp,  Phi_full)
    C.put("morphism", key_pq,  Phit_full)

    # transport_cache.py  (Phi)
    out = Phi_full if kind == "q_to_p" else Phit_full
    #print(f"[DEBUG:Phi:out] aid={getattr(agent,'id','?')} kind={kind} M.shape={out.shape}")
    return out







def Phi_tilde(ctx, agent) -> np.ndarray:
    return Phi(ctx, agent, kind="p_to_q")



def Lambda_dn(ctx, child, parent, which="q", *, tol=None):
    C, cfg = ctx.cache, _ctx_cfg(ctx)
    step = int(getattr(ctx, "global_step", -1))
    key = C.key_Lambda(child_id=_aid(child), parent_id=_aid(parent), step=step,
                       phi_child=_get_phi(child, which), phi_parent=_get_phi(parent, which),
                       cfg=cfg, which=which, up=False, tol=tol)
    Lam = C.get("lambda", key)
    if Lam is None:
        E_c = E_grid(ctx, child, which=which, tol=tol)
        E_p = E_grid(ctx, parent, which=which, tol=tol)
        Lam = np.einsum("...ik,...kj->...ij", E_c, safe_omega_inv(E_p), optimize=True)
        C.put("lambda", key, Lam)
    return Lam

def Lambda_up(ctx, child, parent, which="q", *, tol=None):
    C, cfg = ctx.cache, _ctx_cfg(ctx)
    step = int(getattr(ctx, "global_step", -1))
    key = C.key_Lambda(child_id=_aid(child), parent_id=_aid(parent), step=step,
                       phi_child=_get_phi(child, which), phi_parent=_get_phi(parent, which),
                       cfg=cfg, which=which, up=True, tol=tol)
    Lam = C.get("lambda", key)
    if Lam is None:
        E_c = E_grid(ctx, child, which=which, tol=tol)
        E_p = E_grid(ctx, parent, which=which, tol=tol)
        Lam = np.einsum("...ik,...kj->...ij", E_p, safe_omega_inv(E_c), optimize=True)
        C.put("lambda", key, Lam)
    return Lam

def Theta_dn(ctx, child, parent, *, tol=None):
    C, cfg = ctx.cache, _ctx_cfg(ctx)
    step = int(getattr(ctx, "global_step", -1))
    key = C.key_Theta(child_id=_aid(child), parent_id=_aid(parent), step=step,
                      phi_child_q=_get_phi(child, "q"), phi_parent_q=_get_phi(parent, "q"),
                      phi_child_p=_get_phi(child, "p"), phi_parent_p=_get_phi(parent, "p"),
                      cfg=cfg, tol=tol)
    out = C.get("theta", key)
    if out is None:
        Lq = Lambda_dn(ctx, child, parent, which="q", tol=tol)
        Lp = Lambda_dn(ctx, child, parent, which="p", tol=tol)
        Phi_c  = Phi(ctx, child,  kind="q_to_p")
        Phit_c = Phi(ctx, child,  kind="p_to_q")
        Theta_q = np.einsum("...ik,...kj->...ij", Phit_c, Lp, optimize=True)
        Theta_p = np.einsum("...ik,...kj->...ij", Phi_c,  Lq, optimize=True)
        out = {"q": Theta_q, "p": Theta_p}
        C.put("theta", key, out)
    return out


def Theta_up(ctx, child, parent, *, tol=None):
    """
    Cross-scale morphisms Θ for the 'up' direction (child -> parent):
      Θ_q = Φ_p^T  Λ_p^up        : (..., Kq_parent, Kp_child)
      Θ_p = Φ_p    Λ_q^up        : (..., Kp_parent, Kq_child)
    Versioned by (child/parent φ on both fibers, step, cfg).
    """
    C, cfg = ctx.cache, _ctx_cfg(ctx)
    step = int(getattr(ctx, "global_step", -1))

    # Versioned key (depends on both fibers for child and parent)
    key = C.key_Theta(
        child_id=_aid(child), parent_id=_aid(parent), step=step,
        phi_child_q=_get_phi(child, "q"),  phi_parent_q=_get_phi(parent, "q"),
        phi_child_p=_get_phi(child, "p"),  phi_parent_p=_get_phi(parent, "p"),
        cfg=cfg, tol=tol,
    )

    out = C.get("theta", key)
    if out is not None:
        return out

    # Build constituents (all cached themselves)
    Lq_up = Lambda_up(ctx, child, parent, which="q")  # (..., Kq_parent, Kq_child)
    Lp_up = Lambda_up(ctx, child, parent, which="p")  # (..., Kp_parent, Kp_child)
    Phi_p  = Phi(ctx, parent, kind="q_to_p")          # (..., Kp_parent, Kq_parent)
    Phit_p = Phi(ctx, parent, kind="p_to_q")          # (..., Kq_parent, Kp_parent)

    # Shape sanity (comment out if noisy)
    _assert_chain("Theta_up.q",  Phit_p, Lp_up)       # (Kq_p×Kp_p) @ (Kp_p×Kp_c) -> (Kq_p×Kp_c)
    _assert_chain("Theta_up.p",  Phi_p,  Lq_up)       # (Kp_p×Kq_p) @ (Kq_p×Kq_c) -> (Kp_p×Kq_c)

    # Compose
    Theta_q = np.einsum("...ik,...kj->...ij", Phit_p, Lp_up, optimize=True)  # (..., Kq_parent, Kp_child)
    Theta_p = np.einsum("...ik,...kj->...ij", Phi_p,  Lq_up, optimize=True)  # (..., Kp_parent, Kq_child)

    out = {"q": Theta_q, "p": Theta_p}
    C.put("theta", key, out)
    return out




#==============================================================================
#
#
#
#==============================================================================




def get_morphism_bases(ctx, agent):
    """Return (Phi_0, Phi_tilde_0) from persistent cache; seed once if absent."""
    ns = ctx.cache.nsp("morph_base")
    key = _bases_key(agent)
    pair = ns.get(key)
    if pair is None:
        # Back-compat: slurp from agent attributes once; else identity
        Kq = key[1]; Kp = key[2]
        Phi0  = getattr(agent, "Phi_0",       None)
        Phit0 = getattr(agent, "Phi_tilde_0", None)
        if Phi0 is None:
            Phi0 = _eye_like(Kp, Kq)
        if Phit0 is None:
            Phit0 = _eye_like(Kq, Kp)
        pair = (np.asarray(Phi0,  np.float32), np.asarray(Phit0, np.float32))
        ns[key] = pair
    return pair

def set_morphism_bases(ctx, agent, Phi0, Phit0):
    """Write (Phi_0, Φ̃_0) into persistent cache."""
    ctx.cache.nsp("morph_base")[_bases_key(agent)] = (
        np.asarray(Phi0,  np.float32),
        np.asarray(Phit0, np.float32),
    )


def E_grid_field(ctx, phi_field, generators, *, sign=+1):
    """
    Exponential from *field* (not from agent): E = exp(sign * φ) under the given generators.
    Uses the same sanitization as E_grid.
    """
    phi = _sanitize_axis_angle(np.asarray(phi_field, np.float32))
    phi = (float(sign) * phi).astype(np.float32, copy=False)
    G   = np.asarray(generators, np.float32)
    # clip via the same max-norm logic if you want parity with E_grid:
    # max_norm = float(getattr(CFG, "exp_clip_max_norm", 10.0))
    # return exp_lie_algebra_irrep(phi, G, max_norm=max_norm)
    return exp_lie_algebra_irrep(phi, G)




def omega_child_to_parents_batched(ctx, child, parents, G, *, invert_j=True, mask_tau=1e-3):
    """
    Compute Ω_{i->j} on overlap windows for one child vs many parents, in batch.

    Returns:
      list[(parent_id: int, bbox: (y0,y1,x0,x1), Omega_win: (h,w,K,K) float32)]

    Notes:
      - Infers fiber ('q' or 'p') from the shape of G vs child's generators.
      - Crops to the tight overlap bbox per parent to avoid full-frame work.
      - Uses a batched linear solve for Ej^{-1} (stable & faster than forming inv).
    """
    

    # -------- infer fiber from G's shape (must match child's generators) ---------
    which = None
    G = np.asarray(G, np.float32)
    if getattr(child, "generators_q", None) is not None and \
       tuple(np.shape(child.generators_q)) == tuple(G.shape):
        which = "q"
    elif getattr(child, "generators_p", None) is not None and \
         tuple(np.shape(child.generators_p)) == tuple(G.shape):
        which = "p"
    else:
        raise ValueError(
            f"[omega_child_to_parents_batched] cannot infer fiber from G.shape={G.shape}; "
            f"child.gen_q={getattr(child,'generators_q',None) and np.shape(child.generators_q)}, "
            f"child.gen_p={getattr(child,'generators_p',None) and np.shape(child.generators_p)}"
        )

    # -------- build Ei once; share memory when cropping --------------------------
    Ei_full = E_grid(ctx, child, which=which)  # (H,W,K,K)
    H, W, K, _ = Ei_full.shape

    out = []
    cmask = np.asarray(getattr(child, "mask", 0.0), np.float32)

    for p in (parents or []):
        if p is None:
            continue

        pmask = np.asarray(getattr(p, "mask", 0.0), np.float32)
        ov, bbox = _overlap_and_bbox(cmask, pmask, mask_tau)  # tight bbox
        if bbox is None:
            continue

        y0, y1, x0, x1 = bbox
        h, w = (y1 - y0), (x1 - x0)
        if h <= 0 or w <= 0:
            continue

        # crop Ei/Ej once per parent
        Ei_win = Ei_full[y0:y1, x0:x1, :, :]                          # (h,w,K,K)
        Ej_win = E_grid(ctx, p, which=which)[y0:y1, x0:x1, :, :]       # (h,w,K,K)

        # build Ω_{ij} window
        if invert_j:
            # Solve Ej^T * X^T = Ei^T  ⇒ X = (solve(Ej^T, Ei^T))^T
            N  = h * w
            A  = Ej_win.reshape(N, K, K).transpose(0, 2, 1)  # A = Ej^T
            B  = Ei_win.reshape(N, K, K).transpose(0, 2, 1)  # B = Ei^T
            Xt = np.linalg.solve(A, B)                       # (N,K,K)
            X  = Xt.transpose(0, 2, 1).reshape(h, w, K, K)   # (h,w,K,K)
        else:
            # Convention w/o inversion if you ever need it: Ω = Ei @ Ej^T
            X = np.einsum("...ab,...cb->...ac", Ei_win, Ej_win, optimize=True)

        out.append((int(getattr(p, "id", id(p))), bbox, X.astype(np.float32, copy=False)))

    return out


def Fisher_blocks(ctx, agent, which: str = "q", *, tol=None):
    """
    Cache per-pixel Fisher blocks for a Gaussian: precision Σ^{-1} (and log|Σ|).
    Returns dict with entries you can reuse across updates.
    """
    C   = ctx.cache
    cfg = _ctx_cfg(ctx)
    step_tag = int(getattr(ctx, "global_step", -1))

    # pull fields
    mu  = getattr(agent, "mu_q_field" if which=="q" else "mu_p_field")
    Sig = getattr(agent, "sigma_q_field" if which=="q" else "sigma_p_field")

    key = C.key_Fisher(agent_id=_aid(agent), step=step_tag,
                       mu=mu, Sigma=Sig, cfg=cfg, which=which, tol=tol)
    F = C.get("fisher", key)
    if F is not None:
        return F

    # compute once (vectorized over pixels)
    Sig = np.asarray(Sig, np.float32)                     # (...,K,K)
    # stable precision via cholesky if you have it; fallback to inv
    try:
        L = np.linalg.cholesky(Sig)                       # (...,K,K)
        Linv = np.linalg.inv(L)
        Prec = Linv.swapaxes(-1,-2) @ Linv               # Σ^{-1}
        logdet = 2.0 * np.log(np.clip(np.diagonal(L, axis1=-2, axis2=-1), 1e-30, None)).sum(axis=-1)
    except Exception:
        Prec   = np.linalg.inv(Sig)
        sign, logabs = np.linalg.slogdet(Sig)
        logdet = logabs

    # for natural gradient on μ, Fisher(μ)=Σ^{-1}; keep that directly
    F = {
        "precision": Prec.astype(np.float32, copy=False),   # (...,K,K)
        "logdet":    logdet.astype(np.float32, copy=False)  # (...,)
        # If/when you implement Fisher wrt Σ params, add blocks/operators here.
    }
    C.put("fisher", key, F)
    return F



#==============================================================================
#
#                  Gauge Curvature
#
#==============================================================================





def plaquette_product(ctx, phi_field, generators, *, boundary="periodic", split_edge=False, sign=+1):
    """
    Compute 1×1 plaquette P(i,j) = Ux(i,j) · Uy(i+1,j) · Ux(i,j+1)^T · Uy(i,j)^T
    using central E cache.

    Inputs
    ------
    phi_field: (..., H, W, d) algebra field at sites (SO(3) in R^3)
    generators: (K, K, 3) real irrep generators used by exp_lie_algebra_irrep
    boundary: 'periodic' | 'clamp' | 'zero'
    split_edge: if True, build edge links as exp(½ φ_here) then compose with neighbor exp(½ φ_there)
                (slightly better symmetry when φ varies a lot)

    Returns
    -------
    P: (..., H, W, K, K) plaquette matrices (orthogonal if inputs are)
    """
    # 1) site exponentials from cache: E(i,j) = exp(sign * φ(i,j))
    E = E_grid_field(ctx, phi_field, generators, sign=sign)           # (..., H, W, K, K)
    Et = np.swapaxes(E, -1, -2)                                 # (..., H, W, K, K) transpose

    if not split_edge:
        # simplest link choice: use site exponentials as links
        Ux_ij   = E
        Uy_ij   = E
        Ux_i_j1 = _roll2(E, (0, +1), boundary=boundary)
        Uy_i1_j = _roll2(E, (+1, 0), boundary=boundary)
    else:
        # symmetric "half-edge" links: Ux(i,j) ≈ exp(½φ(i,j)) · exp(½φ(i,j+1))
        # reuse cache by calling E_grid on scaled fields
        half = 0.5
        E_half_here  = E_grid_field(ctx, half * phi_field, generators, sign=sign)
        E_half_right = _roll2(E_grid_field(ctx, half * phi_field, generators, sign=sign), (0, +1), boundary=boundary)
        E_half_down  = _roll2(E_grid_field(ctx, half * phi_field, generators, sign=sign), (+1, 0), boundary=boundary)
        E_half_t_here = np.swapaxes(E_half_here, -1, -2)

       
        Ux_ij   = E_half_here @ E_half_right
        Uy_ij   = E_half_here @ E_half_down
        # neighbors for the other two edges:
        Ux_i_j1 = _roll2(Ux_ij, (0, +1), boundary=boundary)
        Uy_i1_j = _roll2(Uy_ij, (+1, 0), boundary=boundary)

    # 2) plaquette P = Ux(i,j) Uy(i+1,j) Ux(i,j+1)^T Uy(i,j)^T
    P = Ux_ij @ Uy_i1_j @ np.swapaxes(Ux_i_j1, -1, -2) @ np.swapaxes(Uy_ij, -1, -2)
    return P


def curvature_plaquette(ctx, phi_field, generators, *,
                        boundary="periodic",
                        split_edge=False,
                        metric="fro",
                        return_P=False,
                        eps=1e-7):
    """
    Gauge-invariant lattice curvature density from 1×1 plaquettes.

    metric:
      - 'fro'  : 0.5 * ||I - P||_F^2  (cheap, stable; ~ ||log P||^2 near identity)
      - 'angle': use principal rotation angle θ from trace(P) when K==3 and P∈SO(3),
                 returns θ^2 (NaN-safe; clipped)

    Returns
    -------
    curv: (..., H, W) float32
    (optional) P: (..., H, W, K, K)
    """
    P = plaquette_product(ctx, phi_field, generators, boundary=boundary, split_edge=split_edge, sign=+1)
    K  = P.shape[-1]
    IK = np.eye(K, dtype=P.dtype)

    if metric == "fro":
        D = P - IK
        # frob norm squared per site
        curv = 0.5 * np.sum(D * D, axis=(-1, -2))
    elif metric == "angle":
        # only meaningful for faithful SO(3) K=3; fall back if not
        if K != 3:
            D = P - IK
            curv = 0.5 * np.sum(D * D, axis=(-1, -2))
        else:
            # θ = arccos((trace(P)-1)/2), clamp arg for numeric safety
            tr = np.trace(P, axis1=-2, axis2=-1)
            x  = np.clip((tr - 1.0) * 0.5, -1.0 + 1e-6, 1.0 - 1e-6)
            theta = np.arccos(x)
            curv = theta * theta
    else:
        raise ValueError(f"curvature_plaquette: unknown metric '{metric}'")

    curv = curv.astype(np.float32, copy=False)
    return (curv, P) if return_P else curv





# ---------------------------------------------------------------------------
# Optional Λ hooks (centralize later; stubs return cached value if present)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Cached dexpinv matrices Jinv(phi)  (SO(3) axis-angle principal ball)
# ---------------------------------------------------------------------------

def _overlap_and_bbox(mask_i, mask_j, thr):
    """
    Return (ov, (y0,y1,x0,x1)) where ov is boolean overlap on full grid,
    and bbox tightly crops to the minimal rectangle that contains ov==True.
    If no overlap, returns (ov, None).
    """
    
    mi = np.asarray(mask_i, np.float32) > float(thr)
    mj = np.asarray(mask_j, np.float32) > float(thr)
    ov = (mi & mj)
    if not np.any(ov):
        return ov, None
    ys, xs = np.nonzero(ov)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return ov, (y0, y1, x0, x1)




def _alpha_theta(theta: np.ndarray, theta2: np.ndarray) -> np.ndarray:
    """
    Your existing LUT/Taylor alpha(θ) helper. If you already have one
    (e.g., in natural_gradient.py), import and call that instead.
    """
    # Minimal stable version (feel free to swap for your LUT):
    eps = 1e-12
    out = np.empty_like(theta, dtype=np.float32)
    small = theta < 1e-3
    # Taylor: alpha ≈ 1/12 + θ²/720
    out[small] = (1.0/12.0 + theta2[small] * (1.0/720.0)).astype(np.float32)
    # General: alpha = (1/θ²)(1 - (θ/2)cot(θ/2))
    t = theta[~small]
    t2 = theta2[~small]
    half = 0.5 * t
    out[~small] = ((1.0 / np.maximum(t2, eps)) * (1.0 - half * (np.cos(half) / np.maximum(np.sin(half), eps)))).astype(np.float32)
    return out



def build_dexpinv_matrix(phi: np.ndarray, eps: float = 1e-12, phi_clip: float = 2.8) -> np.ndarray:
    """
    Stable SO(3) dexp^{-1}(phi) in closed form, with safe series near 0 and cap near pi.
    Returns Jinv with shape (..., 3, 3).
    """
    # axis-angle
    phi = np.asarray(phi, dtype=np.float32)
    th   = np.linalg.norm(phi, axis=-1)                      # (...,)
    th_c = np.clip(th, 0.0, phi_clip)                        # cap away from pi
    u    = np.divide(phi, th[..., None] + eps, dtype=np.float32)

    # ad(u)
    ux, uy, uz = u[..., 0], u[..., 1], u[..., 2]
    zero = np.zeros_like(ux, dtype=np.float32)
    adu = np.stack([
        np.stack([ zero, -uz,   uy], axis=-1),
        np.stack([  uz,  zero, -ux], axis=-1),
        np.stack([ -uy,   ux,  zero], axis=-1),
    ], axis=-2).astype(np.float32)  # (..., 3, 3)

    # h(θ) = (θ/2) cot(θ/2) with stable small-θ series
    half  = 0.5 * th_c
    small = th_c < 1e-3
    h = np.empty_like(th_c, dtype=np.float32)

    # series near 0: h = 1 - θ^2/12 - θ^4/720 + O(θ^6)
    t2 = th_c**2
    h[small] = 1.0 - (t2[small] / 12.0) - (t2[small] * t2[small] / 720.0)

    # stable far from 0
    not_small = ~small
    half_ns = np.clip(half[not_small], 1e-6, np.pi/2 - 1e-6)
    h[not_small] = half_ns / np.tan(half_ns)

    # J^{-1}(φ) = I - (θ/2) ad(u) + (1 - h) ad(u)^2
    I    = np.eye(3, dtype=np.float32)
    adu2 = adu @ adu
    c1   = (-half)[..., None, None]                # coefficient for ad(u)
    c2   = (1.0 - h)[..., None, None]             # coefficient for ad(u)^2  <-- FIXED

    Jinv = I + c1 * adu + c2 * adu2
    return Jinv.astype(np.float32)





# ---------------------------------------------------------------------------
# Bundle morphisms (same-level) and cross-scale morphisms (Θ)
# Cached per-step in "morphism" and "theta" namespaces
# ---------------------------------------------------------------------------





def _infer_group_name():
    return str(getattr(CFG, "group_name", "so3")).lower()

def _eye_like(Kd: int, Ks: int, dtype=np.float32):
    M = np.zeros((Kd, Ks), dtype=dtype)
    m = min(Kd, Ks);  M[np.arange(m), np.arange(m)] = 1.0
    return M


def _broadcast_morphism_template(M, target_tail, lead_shape):
    M = np.asarray(M, np.float32)
    if M.shape[-2:] != tuple(target_tail):
        raise ValueError(f"[transport_cache] Bad morphism base shape {M.shape}, expected {tuple(target_tail)}")
    if M.ndim == 2:
        M = np.broadcast_to(M, tuple(lead_shape) + tuple(target_tail))
    return M



def _build_morphisms_with_cache(ctx, agent, Phi_0, Phi_tilde_0, *, group_name=None):
    """
    Build Φ and Φ̃ using the central E_grid cache. Assumes ctx/cache is present.
    Φ       = E_p · Φ₀ · E_q^{-1}
    Φ̃      = E_q · Φ̃₀ · E_p^{-1}
    """
    group = (group_name or _infer_group_name())
    so3_orth = bool(getattr(CFG, "so3_irreps_are_orthogonal", True)) and group == "so3"

    # transport_cache.py  (_build_morphisms_with_cache)
    E_q = E_grid(ctx, agent, which="q").astype(np.float32, copy=False)  # (. Kq, Kq)
    E_p = E_grid(ctx, agent, which="p").astype(np.float32, copy=False)  # (. Kp, Kp)
    
    Kq = int(E_q.shape[-1]); Kp = int(E_p.shape[-1])
    lead = np.broadcast_shapes(E_q.shape[:-2], E_p.shape[:-2])
    
    # --- NEW DEBUG & GUARDS ---
    Kq_mu = int(np.asarray(agent.mu_q_field).shape[-1])
    Kp_mu = int(np.asarray(agent.mu_p_field).shape[-1])

    
    if Kq != Kq_mu or Kp != Kp_mu:
        raise ValueError(f"[transport_cache] Fiber-size mismatch: "
                         f"E_grid says (Kq,Kp)=({Kq},{Kp}) but mu-fields say ({Kq_mu},{Kp_mu}). "
                         f"Check generators_q/p shapes.")




    Kq = int(E_q.shape[-1]); Kp = int(E_p.shape[-1])
    lead = np.broadcast_shapes(E_q.shape[:-2], E_p.shape[:-2])

    # templates: ensure shape & broadcast
    Phi_0       = _broadcast_morphism_template(Phi_0,       (Kp, Kq), lead)
    Phi_tilde_0 = _broadcast_morphism_template(Phi_tilde_0, (Kq, Kp), lead)

    # inverses (transpose for orthogonal)
    E_q_inv = np.swapaxes(E_q, -1, -2) if so3_orth else safe_omega_inv(E_q).astype(np.float32, copy=False)
    E_p_inv = np.swapaxes(E_p, -1, -2) if so3_orth else safe_omega_inv(E_p).astype(np.float32, copy=False)

    # einsums
    Phi       = np.einsum("...ik,...kj,...jl->...il", E_p, Phi_0,       E_q_inv, optimize=True).astype(np.float32, copy=False)
    Phi_tilde = np.einsum("...ik,...kj,...jl->...il", E_q, Phi_tilde_0, E_p_inv, optimize=True).astype(np.float32, copy=False)
    return Phi, Phi_tilde




# ---------------------------------------------------------------------------
# Warming & invalidation
# ---------------------------------------------------------------------------

def warm_E(ctx: Any, agents: Sequence[Any], whiches: Iterable[str] = ("q", "p")) -> None:
    """Precompute exp(φ) for many agents (idempotent)."""
    for a in agents:
        for w in whiches:
            try:
                _ = E_grid(ctx, a, which=w)
            except Exception:
                # Keep warm-up best-effort; avoid failing the whole pass.
                print("warm-e fail ")

def warm_Jinv(ctx: Any, agents: Sequence[Any], whiches: Iterable[str] = ("q", "p")) -> None:
    """Precompute Jinv for many agents (idempotent)."""
    for a in agents:
        for w in whiches:
            try:
                _ = Jinv_grid(ctx, a, which=w)
            except Exception:
                print("warm-jinv fail ")



# ---------------------------------------------------------------------------
# Transitional compatibility helpers (optional)
# ---------------------------------------------------------------------------


def _matmul_shapes_ok(A, B):
    # A(..., i, k) @ B(..., k, j) -> (..., i, j)
    if A is None or B is None:
        return False
    if A.ndim < 2 or B.ndim < 2:
        return False
    return A.shape[-1] == B.shape[-2]

def _assert_chain(name, *mats):
    # Ensure M1@M2@... is dimensionally valid
    for idx in range(len(mats) - 1):
        if not _matmul_shapes_ok(mats[idx], mats[idx+1]):
            a = mats[idx].shape if mats[idx] is not None else None
            b = mats[idx+1].shape if mats[idx+1] is not None else None
            raise ValueError(f"[{name}] shape mismatch at link {idx}: {a} @ {b}")

def _roll2(a, shift_hw, boundary="periodic", fill=None):
    """
    2D roll with simple boundary handling.
    - periodic: np.roll
    - clamp: out-of-range pulls from edge
    - zero: pad with `fill` (or identity if used with matrices)
    """
    sh, sw = shift_hw
    if boundary == "periodic":
        return np.roll(np.roll(a, sh, axis=-3), sw, axis=-2)

    H, W = a.shape[-3], a.shape[-2]
    if boundary == "clamp":
        # roll, then clamp edges by copying nearest valid row/col
        out = np.roll(np.roll(a, sh, axis=-3), sw, axis=-2)
        if sh > 0:  out[..., :sh, :, :] = out[..., sh:sh+1, :, :]
        if sh < 0:  out[..., H+sh:, :, :] = out[..., H+sh-1:H+sh, :, :]
        if sw > 0:  out[..., :, :sw, :] = out[..., :, sw:sw+1, :]
        if sw < 0:  out[..., :, W+sw:, :] = out[..., :, W+sw-1:W+sw, :]
        return out

    # boundary == "zero" (or anything else): pad with fill (expects matrices)
    out = np.roll(np.roll(a, sh, axis=-3), sw, axis=-2)
    if fill is None:
        # try to infer identity of the right size
        K = a.shape[-1]
        fill = np.eye(K, dtype=a.dtype)
    if sh > 0:  out[..., :sh, :, :] = fill
    if sh < 0:  out[..., H+sh:, :, :] = fill
    if sw > 0:  out[..., :, :sw, :] = fill
    if sw < 0:  out[..., :, W+sw:, :] = fill
    return out
