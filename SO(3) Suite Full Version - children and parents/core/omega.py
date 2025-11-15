# -*- coding: utf-8 -*-
"""
omega.py — Lie-group exponentials, transports, BCH helpers, and curvature ops.
CacheHub-ready: uses ctx.cache namespaces 'exp' and 'omega' when provided.

Author: c&c
"""
from __future__ import annotations

from typing import Tuple, Dict

from typing import Optional, Iterable
import numpy as np
from scipy.linalg import expm

from core.numerical_utils import safe_omega_inv

    


# Public API
__all__ = [
    "exp_lie_algebra_irrep", "retract_phi_principal",  
   
    "d_exp_phi_exact",
    "d_exp_phi_tilde_exact",
    "compute_gauge_potential",
    "compute_field_strength",
    "compute_field_strength_general",
    "compute_curvature_gradient_analytical",
    
]


# =============================================================================
#                               Cache helpers
# =============================================================================

def _arr_key(a):
    a = np.asarray(a)
    # (data pointer, shape, dtype) — stable and cheap
    return (int(a.__array_interface__['data'][0]), a.shape, a.dtype.str)

def axis_angle_norm(phi: np.ndarray) -> np.ndarray:
    """‖φ‖ with keepdims over last axis."""
    phi = np.asarray(phi)
    return np.linalg.norm(phi, axis=-1, keepdims=True)


# -----------------------------------------------------------------------------
# Principal-ball retraction (axis–angle)
# -----------------------------------------------------------------------------

def retract_phi_principal(phi, *, margin=None, max_theta=None):
    """
    Project axis–angle φ (…,3) onto the SO(3) principal ball (‖φ‖ ≤ π − margin),
    with stable wrap/reflect/clamp and optional post-retract hard cap.

    - margin: if None, read core.config.phi_principal_margin (fallback 1e-4)
    - max_theta: optional hard cap applied AFTER retract (≤ π − margin)

    Returns float32, batched, idempotent.
    """
    import math
    import numpy as np

    # margin policy
    if margin is None:
        try:
            import core.config as config
            margin = float(getattr(config, "phi_principal_margin", 1e-4))
        except Exception:
            margin = 1e-4

    v = np.asarray(phi, np.float32)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    if v.shape[-1] != 3:
        return v

    eps   = 1e-12
    twopi = np.float32(2.0 * np.pi)
    # norms + unit directions
    t = axis_angle_norm(v)
    t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    u = v / np.maximum(t, eps)

    # wrap to [0, 2π), reflect (>π) back, then clamp to π−margin
    t_mod     = np.mod(t, twopi)
    over      = (t_mod > np.pi)
    t_ref     = np.where(over, (twopi - t_mod), t_mod)
    u_ref     = np.where(over, -u, u)
    t_clamped = np.minimum(t_ref, (np.pi - float(margin)))

    out = (u_ref * t_clamped).astype(np.float32, copy=False)

    # optional hard cap after retract
    if max_theta is not None:
        cap   = float(min(float(max_theta), math.pi - float(margin)))
        th    = np.linalg.norm(out, axis=-1, keepdims=True)
        scale = np.minimum(1.0, cap / np.maximum(th, 1e-12)).astype(np.float32)
        out   = (out * scale).astype(np.float32, copy=False)

    return out




def pre_retract_fields(objs):
    """Project φ and φ̃ into the SO(3) principal ball before building caches."""
    for a in (objs or []):
        if a is None:
            continue
        if hasattr(a, "phi_field") and a.phi_field is not None:
            a.phi_field = retract_phi_principal(a.phi_field)
        if hasattr(a, "phi_model_field") and a.phi_model_field is not None:
            a.phi_model_field = retract_phi_principal(a.phi_model_field)





# =============================================================================
#                          Exponentials / Transports
# =============================================================================



def exp_lie_algebra_irrep(
    v_batch: np.ndarray,
    generators: np.ndarray,
    *,
    threshold: float = 1e-3,
    max_norm: Optional[float] = None,
    parallelize_expm: bool = False,
    n_jobs: int = -1,
    **_legacy,  # accepts group_name=..., group=..., etc. (ignored)
) -> np.ndarray:
    """
    Compute exp(Σ v^a G_a) in an arbitrary K×K irrep for a batch of v.

    v_batch:    (..., 3) axis–angle coefficients in the chosen basis
    generators: (3, K, K) irrep generators (typically skew for SO(3))
    Returns:    (..., K, K) float32

    Notes:
      - Accepts legacy kwargs (e.g., group_name=..., group=...) and ignores them.
      - 'threshold' controls the small-angle Taylor cutoff.
      - 'max_norm' clips ||v|| to stabilize far from 0.
    """
   
    v = np.asarray(v_batch, np.float64)
    G = np.asarray(generators, np.float64)

    if v.ndim == 0 or v.shape[-1] != 3:
        raise ValueError(f"exp_lie_algebra_irrep: expected (...,3) v_batch, got {v.shape}")
    if G.ndim != 3 or G.shape[0] != 3 or G.shape[-1] != G.shape[-2]:
        raise ValueError(f"exp_lie_algebra_irrep: generators must be (3,K,K), got {G.shape}")

    batch_shape = v.shape[:-1]
    d = 3
    K = int(G.shape[-1])

    # Optional global norm clip (stable even when ||v||≈0)
    if max_norm is not None:
        vv = v.reshape(-1, d)
        n = np.linalg.norm(vv, axis=1)
        scale = np.minimum(1.0, max_norm / (n + 1e-16)).astype(v.dtype)
        v = (vv * scale[:, None]).reshape(batch_shape + (d,))

    # Algebra element: Xi = v·G
    Xi = np.einsum("...a,aij->...ij", v, G, optimize=True)

    # Enforce skew numerically (harmless for general reps; stabilizes SO(3))
    X = 0.5 * (Xi - np.swapaxes(Xi, -1, -2))

    flat = X.reshape(-1, K, K)
    out = np.empty_like(flat)

    norms = np.linalg.norm(v.reshape(-1, d), axis=1)
    mask_small = norms < float(threshold)

    # 4th-order Taylor near identity
    if np.any(mask_small):
        Xs = flat[mask_small]
        I = np.eye(K, dtype=flat.dtype)[None]
        X2 = Xs @ Xs
        X3 = X2 @ Xs
        X4 = X2 @ X2
        out[mask_small] = I + Xs + 0.5*X2 + (1.0/6.0)*X3 + (1.0/24.0)*X4

    # Exact expm elsewhere (optionally parallelized)
    if np.any(~mask_small):
        Xh = flat[~mask_small]
        if parallelize_expm and Xh.shape[0] >= 64:
            try:
                from joblib import Parallel, delayed, parallel_backend
                with parallel_backend("threading"):
                    res = Parallel(n_jobs=n_jobs, batch_size=8)(
                        delayed(expm)(X_i) for X_i in Xh
                    )
            except Exception:
                res = [expm(X_i) for X_i in Xh]
        else:
            res = [expm(X_i) for X_i in Xh]
        out[~mask_small] = np.stack(res, axis=0)

    out = np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
    return out.reshape(batch_shape + (K, K)).astype(np.float32)





# =============================================================================
#                       Exact dexp derivatives for SO(3)
# =============================================================================



def d_exp_phi_exact(
    phi_vec: np.ndarray,
    generators: np.ndarray,
    *,
    exp_phi_all: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> Iterable[np.ndarray]:
    """
    ∂/∂φ^a e^{Φ} with Φ = Σ_a φ^a G_a  (SO(3) left-trivialized).
    Returns [dE/dφ^0, dE/dφ^1, dE/dφ^2], each (...,K,K).
    """
    phi_vec = np.asarray(phi_vec, dtype=np.float32)
    G = np.asarray(generators, dtype=np.float32)

    *shape, d = phi_vec.shape
    assert d == 3, "d_exp_phi_exact assumes SO(3) (d=3)."
    K = int(G.shape[-1])
    N = int(np.prod(shape)) if shape else 1

    φ = phi_vec.reshape(N, 3)                                 # (N,3)
    Φ = np.einsum("na,aij->nij", φ, G, optimize=True)         # (N,K,K)

    θ = np.linalg.norm(φ, axis=1)                             # (N,)
    c1 = np.empty_like(θ, dtype=np.float32)
    c2 = np.empty_like(θ, dtype=np.float32)
    small = θ < 1e-4
    if np.any(small):
        t = θ[small]; t2 = t*t; t4 = t2*t2
        c1[small] = 0.5 - t2/24.0 + t4/720.0
        c2[small] = 1.0/6.0 - t2/120.0 + t4/5040.0
    big = ~small
    if np.any(big):
        tb = θ[big]
        c1[big] = (1.0 - np.cos(tb)) / np.maximum(tb*tb, eps)
        c2[big] = (tb - np.sin(tb)) / np.maximum(tb*tb*tb, eps)

    if exp_phi_all is None:
       
        E = exp_lie_algebra_irrep(φ, G, parallelize_expm=False)  # (N,K,K)
    else:
        E = np.asarray(exp_phi_all, dtype=np.float32).reshape(N, K, K)

    # Commutators: ad_Φ(G_a) = [Φ, G_a], ad_Φ^2(G_a) = [Φ, [Φ, G_a]]
    ΦN = Φ[None, ...]                        # (1,N,K,K)
    G_A = G[:, None, :, :]                   # (3,1,K,K)

    # [Φ, G_a] = Φ G_a − G_a Φ  -> (3,N,K,K)
    comm1 = (ΦN @ G_A) - (G_A @ ΦN)

    # [Φ, [Φ, G_a]] -> (3,N,K,K)
    comm2 = (ΦN @ comm1) - (comm1 @ ΦN)

    # Q_a = G_a − c1 ad_Φ(G_a) + c2 ad_Φ^2(G_a)
    Q = G_A - c1[None, :, None, None] * comm1 + c2[None, :, None, None] * comm2  # (3,N,K,K)

    # d e^{Φ} / dφ^a = E · Q_a
    dE = np.einsum("nij,anjk->anik", E, Q, optimize=True)     # (3,N,K,K)
    dE = dE.reshape(3, *shape, K, K).astype(np.float32, copy=False)
    return [dE[i] for i in range(3)]


def d_exp_phi_tilde_exact(
    phi_tilde_vec: np.ndarray,
    generators: np.ndarray,
    *,
    exp_phi_tilde_all: Optional[np.ndarray] = None,
    eps: float = 1e-12,
):
    """
    Companion for φ̃; same construction and signs.
    """
    φ̃ = np.asarray(phi_tilde_vec, dtype=np.float32)
    G = np.asarray(generators, dtype=np.float32)

    *shape, d = φ̃.shape
    assert d == 3, "d_exp_phi_tilde_exact assumes SO(3) (d=3)."
    K = int(G.shape[-1])
    N = int(np.prod(shape)) if shape else 1

    φ = φ̃.reshape(N, 3)
    Φ = np.einsum("na,aij->nij", φ, G, optimize=True)         # (N,K,K)

    θ = np.linalg.norm(φ, axis=1)
    c1 = np.empty_like(θ, dtype=np.float32)
    c2 = np.empty_like(θ, dtype=np.float32)
    small = θ < 1e-4
    if np.any(small):
        t = θ[small]; t2 = t*t; t4 = t2*t2
        c1[small] = 0.5 - t2/24.0 + t4/720.0
        c2[small] = 1.0/6.0 - t2/120.0 + t4/5040.0
    big = ~small
    if np.any(big):
        tb = θ[big]
        c1[big] = (1.0 - np.cos(tb)) / np.maximum(tb*tb, eps)
        c2[big] = (tb - np.sin(tb)) / np.maximum(tb*tb*tb, eps)

    if exp_phi_tilde_all is None:
       
        E = exp_lie_algebra_irrep(φ, G, parallelize_expm=False)
    else:
        E = np.asarray(exp_phi_tilde_all, dtype=np.float32).reshape(N, K, K)

    ΦN = Φ[None, ...]
    G_A = G[:, None, :, :]
    comm1 = (ΦN @ G_A) - (G_A @ ΦN)
    comm2 = (ΦN @ comm1) - (comm1 @ ΦN)
    Q = G_A - c1[None, :, None, None] * comm1 + c2[None, :, None, None] * comm2

    dE = np.einsum("nij,anjk->anik", E, Q, optimize=True)
    dE = dE.reshape(3, *shape, K, K).astype(np.float32, copy=False)
    return [dE[i] for i in range(3)]





# =============================================================================
#                             Curvature operators
# =============================================================================

def compute_gauge_potential(phi, dx=1.0):
    """
    A_μ^a = ∂_μ φ^a using central differences (periodic BC). Returns a tuple over μ.
    """
    if phi.ndim < 2:
      raise ValueError("phi must have at least one spatial and one Lie algebra dimension.")
    D = phi.ndim - 1
    return tuple((np.roll(phi, -1, axis=μ) - np.roll(phi, 1, axis=μ)) / (2.0 * dx) for μ in range(D))


def compute_field_strength(phi: np.ndarray, generators: np.ndarray, dx: float = 1.0) -> Dict[str, np.ndarray]:
    """
    2D nonabelian field strength F_{xy} from φ^a and generators.
    Returns {"xy": F_xy} with shape (H,W,K,K).
    """
    H, W, d = phi.shape
    phi = np.asarray(phi, np.float32)
    G   = np.asarray(generators, np.float32)

    Phi = np.einsum("hwd,dkm->hwkm", phi, G)  # (H,W,K,K)

    A_x = (np.roll(Phi, -1, 0) - np.roll(Phi, 1, 0)) / (2 * dx)
    A_y = (np.roll(Phi, -1, 1) - np.roll(Phi, 1, 1)) / (2 * dx)

    dAy_dx = (np.roll(A_y, -1, 0) - np.roll(A_y, 1, 0)) / (2 * dx)
    dAx_dy = (np.roll(A_x, -1, 1) - np.roll(A_x, 1, 1)) / (2 * dx)

    comm = np.einsum("...ij,...jk->...ik", A_x, A_y) - np.einsum("...ij,...jk->...ik", A_y, A_x)
    F_xy = dAy_dx - dAx_dy + comm
    return {"xy": F_xy.astype(np.float32, copy=False)}


def compute_field_strength_general(phi: np.ndarray, generators: np.ndarray, dx: float = 1.0) -> Dict[Tuple[int,int], np.ndarray]:
    """
    All nonabelian curvature components F_{μν} for φ on a D-dim grid.
    """
    phi = np.asarray(phi, np.float32)
    G   = np.asarray(generators, np.float32)
    Phi = np.einsum("...d,dkm->...km", phi, G)
    D = phi.ndim - 1
    F: Dict[Tuple[int,int], np.ndarray] = {}

    A = [(np.roll(Phi, -1, mu) - np.roll(Phi, 1, mu)) / (2 * dx) for mu in range(D)]

    for mu in range(D):
        for nu in range(mu + 1, D):
            dA_nu_mu = (np.roll(A[nu], -1, mu) - np.roll(A[nu], 1, mu)) / (2 * dx)
            dA_mu_nu = (np.roll(A[mu], -1, nu) - np.roll(A[mu], 1, nu)) / (2 * dx)
            comm = np.einsum("...ij,...jk->...ik", A[mu], A[nu]) - np.einsum("...ij,...jk->...ik", A[nu], A[mu])
            F[(mu, nu)] = (dA_nu_mu - dA_mu_nu + comm).astype(np.float32, copy=False)
    return F


def compute_curvature_gradient_analytical(phi, generators, dx: float = 1.0, metric_ab: np.ndarray | None = None):
    """
    ∇_φ Tr(F^2). If generators aren't trace-orthonormal, pass metric_ab = [Tr(G^a G^b)].
    """
    H, W, d = phi.shape
    phi = np.asarray(phi, np.float32)
    G   = np.asarray(generators, np.float32)

    A_x = (np.roll(phi, -1, 0) - np.roll(phi, 1, 0)) / (2 * dx)
    A_y = (np.roll(phi, -1, 1) - np.roll(phi, 1, 1)) / (2 * dx)

    A_x_mat = np.einsum("hwd,dkm->hwkm", A_x, G)
    A_y_mat = np.einsum("hwd,dkm->hwkm", A_y, G)

    dAy_dx = (np.roll(A_y_mat, -1, 0) - np.roll(A_y_mat, 1, 0)) / (2 * dx)
    dAx_dy = (np.roll(A_x_mat, -1, 1) - np.roll(A_x_mat, 1, 1)) / (2 * dx)
    comm_xy = A_x_mat @ A_y_mat - A_y_mat @ A_x_mat
    F_xy = dAy_dx - dAx_dy + comm_xy

    dF_dx = (np.roll(F_xy, -1, 0) - np.roll(F_xy, 1, 0)) / (2 * dx)
    dF_dy = (np.roll(F_xy, -1, 1) - np.roll(F_xy, 1, 1)) / (2 * dx)
    comm1 = A_x_mat @ F_xy - F_xy @ A_x_mat
    comm2 = A_y_mat @ F_xy - F_xy @ A_y_mat
    divF = dF_dx + dF_dy + comm1 + comm2

    if metric_ab is None:
        grad = 2.0 * np.stack([np.einsum("...ij,ij->...", divF, G[a]) for a in range(d)], axis=-1)
    else:
        Ginvs = safe_omega_inv(metric_ab.astype(np.float64, copy=False))
        comps = np.array([np.einsum("...ij,ij->...", divF, G[a]) for a in range(d)])  # (d,H,W)
        grad = 2.0 * np.einsum("ab,b...->a...", Ginvs, comps).transpose(1, 2, 0)
    return grad.astype(np.float32, copy=False)


# =============================================================================
#                       Holonomy / transport helpers
# =============================================================================





