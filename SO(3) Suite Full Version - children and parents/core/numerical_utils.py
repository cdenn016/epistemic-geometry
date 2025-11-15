# -*- coding: utf-8 -*-
"""
Numerical utilities (strict-SPD, cache-hub friendly, no legacy prints)

Author: c&c
"""
from __future__ import annotations

import sys
import io
import contextlib
import warnings
from collections import defaultdict
from typing import Optional, Tuple

import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import logm  # for safe_logm

import core.config as config

# -----------------------------------------------------------------------------
# Globals / counters
# -----------------------------------------------------------------------------

fallback_counter = defaultdict(int)
EPS = float(getattr(config, "eps", 1e-8))


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def isfinite_scalar(x) -> bool:
    """
    True iff x can be coerced to a scalar float and is finite.
    Accepts numpy arrays (size=1), numpy scalars, Python floats/ints.
    """
    import math
    try:
        a = np.asarray(x)
        if a.size == 1:
            return math.isfinite(float(a))
        return np.all(np.isfinite(a.astype(np.float64, copy=False)))
    except Exception:
        try:
            return math.isfinite(float(x))
        except Exception:
            return False


def resize_nn(a: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor resize for arrays with leading (H,W, ...)."""
    H, W = shape
    h, w = a.shape[:2]
    if (h, w) == (H, W):
        return a
    yi = np.clip(np.rint(np.linspace(0, h - 1, H)).astype(int), 0, h - 1)
    xi = np.clip(np.rint(np.linspace(0, w - 1, W)).astype(int), 0, w - 1)
    return a[yi][:, xi]


def log_fallback(tag: str, msg: str, count_key: Optional[str] = None, error: bool = False):
    """
    Standardized fallback logger.
    - Increments a counter when count_key is provided.
    - Warns by default; raises if config.fail_on_fallback or error=True.
    """
    full = f"[{tag}] {msg}"
    if count_key:
        fallback_counter[count_key] += 1
        full = f"[{tag}] Fallback #{fallback_counter[count_key]}: {msg}"
    if getattr(config, "fail_on_fallback", False) or error:
        raise RuntimeError(full)
    warnings.warn(full, RuntimeWarning)
    sys.stdout.flush()


# -----------------------------------------------------------------------------
# SPD projections / inverses
# -----------------------------------------------------------------------------

def project_spd(
    S: np.ndarray,
    eig_floor: Optional[float] = None,
    eig_cap: Optional[float] = None,
    cond_cap: Optional[float] = None,
    trace_target: Optional[float] = None,
) -> np.ndarray:
    """
    Project symmetric matrices S[...,K,K] onto SPD by eigen clipping (no inversion).
    """
    if eig_floor is None:
        eig_floor = float(getattr(config, "sigma_eig_floor", 1e-6))
    S = 0.5 * (S + np.swapaxes(S, -1, -2))
    w, V = np.linalg.eigh(S)
    w = np.maximum(w, eig_floor)
    if cond_cap is not None:
        w_min = w.min(axis=-1, keepdims=True)
        w = np.minimum(w, w_min * float(cond_cap))
    if eig_cap is not None:
        w = np.minimum(w, float(eig_cap))
    S_proj = (V * w[..., None, :]) @ np.swapaxes(V, -1, -2)
    if trace_target is not None:
        tr = np.trace(S_proj, axis1=-2, axis2=-1)[..., None, None]
        S_proj = S_proj * (float(trace_target) / np.clip(tr, 1e-12, None))
    return 0.5 * (S_proj + np.swapaxes(S_proj, -1, -2))


def safe_inv(Sigma: np.ndarray, eps: float = 1e-10, max_eig: Optional[float] = None, debug: bool = False) -> np.ndarray:
    """
    Batch-safe inverse of SPD matrices with eigenvalue clipping.
    Supports input shape (..., K, K).
    """
    shape = Sigma.shape
    K = shape[-1]
    flat = 0.5 * (Sigma + np.swapaxes(Sigma, -1, -2)).reshape(-1, K, K)
    eigvals, eigvecs = np.linalg.eigh(flat)
    if debug:
        try:
            print(f"[safe_inv] eig min={eigvals.min(axis=1).min():.2e}  max={eigvals.max(axis=1).max():.2e}")
        except Exception:
            pass
    eigvals = np.maximum(eigvals, eps)
    if max_eig is not None:
        eigvals = np.minimum(eigvals, float(max_eig))
    inv_eigs = 1.0 / eigvals
    inv = (eigvecs * inv_eigs[..., None, :]) @ np.swapaxes(eigvecs, -1, -2)
    inv = 0.5 * (inv + np.swapaxes(inv, -1, -2))
    inv = np.nan_to_num(inv, nan=eps, posinf=eps, neginf=eps)
    return inv.reshape(shape)


def safe_batch_inverse(S: np.ndarray, name: str = "Σ", mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Vectorized matrix inverse with optional mask and robust per-slice fallback.
    Inputs:
        S   : (..., K, K)
        mask: optional (...,) boolean; False → returns identity for that slice
    """
    S = np.asarray(S)
    *_, K, _ = S.shape
    eye = np.eye(K, dtype=S.dtype)
    X = 0.5 * (S + np.swapaxes(S, -1, -2))
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        X = X.copy()
        X[~mask] = eye
    try:
        inv = np.linalg.inv(X)
        inv = 0.5 * (inv + np.swapaxes(inv, -1, -2))
    except LinAlgError:
        inv = np.empty_like(X)
        Xf = X.reshape(-1, K, K)
        out = inv.reshape(-1, K, K)
        for i, A in enumerate(Xf):
            try:
                Ai = np.linalg.inv(A)
            except LinAlgError:
                log_fallback("safe_batch_inverse", f"{name}: singular slice {i}", count_key="safe_batch_inverse")
                Ai = eye
            out[i] = 0.5 * (Ai + Ai.T)
    return inv


def _invert_matrix_field(Omega_full: np.ndarray) -> Optional[np.ndarray]:
    """Invert a matrix field (H,W,K,K) in one shot; None on failure."""
    try:
        H, W, K, _ = Omega_full.shape
        M = Omega_full.reshape(-1, K, K)
        Minv = np.linalg.inv(M)
        return Minv.reshape(H, W, K, K).astype(np.float32, copy=False)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Lie algebra helpers
# -----------------------------------------------------------------------------

def soft_clip_phi_norm(phi: np.ndarray, max_norm: float = np.pi, eps: float = 1e-8) -> np.ndarray:
    """
    Smoothly compress φ ∈ ℝᵈ to have ‖φ‖ ≤ max_norm.
    """
    phi = np.asarray(phi, np.float32)
    n = np.linalg.norm(phi, axis=-1, keepdims=True)
    scale = np.where(n > max_norm, max_norm / (n + eps), 1.0)
    return phi * scale


def safe_omega_inv(M: np.ndarray, eps: float = 1e-10, debug: bool = False) -> np.ndarray:
    """
    Safely invert an (approximately) orthogonal matrix field.
    - If ||M^T M - I||_F <= tol, return M^T (fast path).
    - Else, re-orthogonalize via SVD and return Q^T with Q = argmin_Q ||M-Q||_F, Q orthogonal.
    Tolerance is adapted to dtype (float32 vs float64).
    """
    M = np.asarray(M)
    assert M.shape[-2] == M.shape[-1], "safe_omega_inv: matrix must be square"
    K = M.shape[-1]

    # Use float64 internally for stability
    M64 = M.astype(np.float64, copy=False)
    MT = np.swapaxes(M64, -1, -2)
    I = np.eye(K, dtype=np.float64)

    # Deviation from orthogonality, per batch
    dev = MT @ M64 - I
    dev_fro = np.linalg.norm(dev.reshape(-1, K * K), axis=1)

    # Adaptive tolerance: at least eps, but also ~50 * machine epsilon of input dtype
    dtype_eps = np.finfo(M.dtype).eps if np.issubdtype(M.dtype, np.floating) else 0.0
    tol = max(float(eps), 50.0 * float(dtype_eps))  # ≈5e-6 for float32, 1e-14 for float64

    if np.all(dev_fro <= tol):
        return np.swapaxes(M, -1, -2)  # fast path, keep original dtype

    # Slow but robust path: SVD re-orthogonalization
    flat = M64.reshape(-1, K, K)
    out = np.empty_like(flat)
    for i, A in enumerate(flat):
        try:
            U, S, Vt = np.linalg.svd(A, full_matrices=False)
            Q = U @ Vt
            out[i] = Q.T
        except Exception:
            # last resort: numeric inverse transpose
            out[i] = np.linalg.inv(A).T

    max_dev = float(dev_fro.max())
    if debug:
        print(f"[safe_omega_inv] max dev {max_dev:.3e} > tol {tol:.3e} — used SVD re-orth")
    log_fallback("safe_omega_inv", f"orthogonality deviation {max_dev:.3e} > tol {tol:.3e}", count_key="omega_inv")
    return out.reshape(M.shape).astype(M.dtype, copy=False)



# -----------------------------------------------------------------------------
# Sigma sanitation (vectorized)
# -----------------------------------------------------------------------------

def _sanitize_one_sigma(S: np.ndarray, eps: float = 1e-4, debug: bool = False) -> Tuple[np.ndarray, bool]:
    """
    Ensure a single Σ is symmetric positive-definite (SPD).
    Returns: (Σ_sanitized, clipped_flag)
    """
    S = 0.5 * (S + S.T)
    if not np.all(np.isfinite(S)):
        if debug:
            print("[sanitize_sigma] NaNs/Infs → eps*I")
        return eps * np.eye(S.shape[0], dtype=S.dtype), True
    try:
        w, V = np.linalg.eigh(S)
    except LinAlgError:
        if debug:
            print("[sanitize_sigma] eigh failed → eps*I")
        return eps * np.eye(S.shape[0], dtype=S.dtype), True
    clipped = np.any(w < eps)
    if clipped:
        w = np.clip(w, eps, None)
        S = (V * w[None, :]) @ V.T
        S = 0.5 * (S + S.T)
    return S, clipped


def sanitize_sigma(
    Sigma: np.ndarray,
    debug: bool = True,
    strict: bool = True,
    eig_floor: Optional[float] = None,
    cond_cap: Optional[float] = None,
    eig_cap: Optional[float] = None,
    trace_target: Optional[float] = None,
    eps: float = None,
) -> np.ndarray:
    """
    Ensure Σ is SPD. Vectorized:
      - strict=True  : project all (most robust).
      - strict=False : project only those that violate simple checks.
    """
    eig_floor = float(eig_floor if eig_floor is not None else getattr(config, "sigma_eig_floor", 1e-6))
    cond_cap = None if cond_cap is None else float(cond_cap)
    eig_cap = None if eig_cap is None else float(eig_cap)
    trace_target = None if trace_target is None else float(trace_target)
    if eps is None:
        eps = EPS

    S = 0.5 * (Sigma + np.swapaxes(Sigma, -1, -2))
    # sanitize NaN/Inf
    bad = ~np.isfinite(S).all(axis=(-2, -1))
    if np.any(bad):
        if debug:
            print(f"[sanitize_sigma] {int(bad.sum())} matrices had NaN/Inf; replaced with floor*I")
        K = S.shape[-1]
        eye = np.eye(K, dtype=S.dtype)
        S = S.copy()
        S[bad] = eig_floor * eye

    if strict:
        return project_spd(S, eig_floor=eig_floor, eig_cap=eig_cap, cond_cap=cond_cap, trace_target=trace_target)

    # light path
    w = np.linalg.eigvalsh(S)
    too_small = (w <= eig_floor).any(axis=-1)
    too_cond = (w.max(axis=-1) > np.maximum(w.min(axis=-1), eig_floor) * (cond_cap if cond_cap else np.inf))
    bad2 = bad | too_small | too_cond
    if not np.any(bad2):
        return S
    Sout = S.copy()
    Sout[bad2] = project_spd(Sout[bad2], eig_floor=eig_floor, eig_cap=eig_cap, cond_cap=cond_cap, trace_target=trace_target)
    if debug:
        total = int(np.prod(S.shape[:-2]))
        print(f"[sanitize_sigma] projected {int(bad2.sum())}/{total} matrices")
    return Sout


def _spd_project_batch(
    A: np.ndarray,
    eig_floor: float = 1e-6,
    eig_cap: Optional[float] = None,
    cond_cap: Optional[float] = None,
    trace_target: Optional[float] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """Project a batch A[...,K,K] onto SPD via eigen clipping."""
    A = 0.5 * (A + np.swapaxes(A, -1, -2))
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, eig_floor)
    if cond_cap is not None:
        w_min = w.min(axis=-1, keepdims=True)
        w = np.minimum(w, w_min * cond_cap)
    if eig_cap is not None:
        w = np.minimum(w, eig_cap)
    A_proj = (V * w[..., None, :]) @ np.swapaxes(V, -1, -2)
    if trace_target is not None:
        tr = np.trace(A_proj, axis1=-2, axis2=-1)[..., None, None]
        A_proj = A_proj * (trace_target / np.clip(tr, eps, None))
    return 0.5 * (A_proj + np.swapaxes(A_proj, -1, -2))


# -----------------------------------------------------------------------------
# Matrix log / logdet
# -----------------------------------------------------------------------------

def safe_logm(M: np.ndarray, fallback_value: Optional[np.ndarray] = None, imag_tol: float = 1e-10, context: str = "Ω") -> np.ndarray:
    """
    Compute logm(M) with stderr suppressed; warn on non-negligible imaginary part.
    """
    try:
        with io.StringIO() as buf, contextlib.redirect_stderr(buf):
            L = logm(M)
        imag_norm = np.linalg.norm(np.imag(L), ord="fro")
        if imag_norm > imag_tol:
            log_fallback("safe_logm", f"{context}: ‖Im(logm)‖={imag_norm:.2e} > tol={imag_tol}")
        return np.real_if_close(L, tol=imag_tol)
    except Exception as e:
        log_fallback("safe_logm", f"{context}: logm failed ({e})", count_key="safe_logm")
        K = M.shape[-1]
        return fallback_value if fallback_value is not None else np.zeros((K, K), dtype=M.dtype)

def safe_logdet(Sigma, eps=1e-8, alpha=1e-2, clip_range=(-1e3, 1e3)):
    """
    Safe logdet for (stacked) SPD-ish matrices.
    Strategy:
      1) Symmetrize and clean NaNs/±inf
      2) Scalar scaling by mean diagonal: A = s * B  ⇒  logdet(A) = K*log(s) + logdet(B)
      3) Try Cholesky on B + jitter*I with adaptive jitter
      4) Fallback to eigenvalues with floor
    Parameters (kept for compatibility):
      eps   : base jitter and numeric floor (default 1e-8)
      alpha : eigenvalue floor *in scaled space* (default 1e-2)
      clip_range : final clipping of logdet (lo, hi) to keep energies sane
    Returns:
      ndarray of log-determinants with batch shape Sigma.shape[:-2]
    """
    

    A = np.asarray(Sigma, np.float64)
    K = A.shape[-1]
    assert A.ndim >= 2 and A.shape[-2] == K, "safe_logdet: last two dims must be (K,K)"

    # 1) Symmetrize + clean
    A = 0.5 * (A + np.swapaxes(A, -1, -2))
    A = np.nan_to_num(A, posinf=np.inf, neginf=0.0)

    # 2) Scalar scaling by mean diagonal to tame condition numbers
    d = np.trace(A, axis1=-2, axis2=-1) / max(K, 1)         # (...,)
    s = np.clip(d, eps, np.inf)                              # avoid 0/neg
    B = A / s[..., None, None]
    base = K * np.log(s)                                     # add back later

    # 3) Cholesky with adaptive jitter
    I = np.eye(K, dtype=np.float64)
    Bf = B.reshape(-1, K, K)
    out = np.empty(Bf.shape[0], dtype=np.float64)
    fb_mask = np.zeros(Bf.shape[0], dtype=bool)

    for i in range(Bf.shape[0]):
        Bij = Bf[i]
        jit = float(max(eps, 1e-15))
        ok = False
        for _ in range(8):                                   # up to ~1e-8 → 1e-7 … 1e0
            try:
                L = np.linalg.cholesky(Bij + jit * I)
                out[i] = 2.0 * np.sum(np.log(np.diag(L)))
                ok = True
                break
            except np.linalg.LinAlgError:
                jit *= 10.0
        if not ok:
            # 4) Eigen fallback with floor in the *scaled* space
            fb_mask[i] = True
            w = np.linalg.eigvalsh(Bij)
            w = np.clip(w, max(alpha, eps), None)
            out[i] = np.sum(np.log(w))

    logdet = out.reshape(base.shape) + base

    # Final clip for numerical sanity (match previous signature)
    if clip_range is not None:
        lo, hi = clip_range
        logdet = np.clip(logdet, lo, hi)

    # Single warning + optional external counter hook (if defined)
    n_fb = int(fb_mask.sum())
    if n_fb:
        msg = f"[safe_logdet] fallback triggered for {n_fb}/{fb_mask.size} matrices"
        warnings.warn(msg, RuntimeWarning)
        lf = globals().get("log_fallback", None)
        if callable(lf):
            try:
                lf("safe_logdet", msg, count_key="safe_logdet")
            except Exception:
                pass

    return logdet


# -----------------------------------------------------------------------------
# Masked stats
# -----------------------------------------------------------------------------

def masked_cond_proxy(S: np.ndarray, mask: np.ndarray, eps: float = 1e-12) -> float:
    """
    Proxy for condition number on support using tr/exp(logdet).
    """
    H, W, K = S.shape[:3]
    S2 = S.reshape(-1, K, K)
    m = mask.reshape(-1).astype(bool)
    if not np.any(m):
        return 1.0
    tr = np.trace(S2[m], axis1=-2, axis2=-1)
    sign, logdet = np.linalg.slogdet(S2[m])
    logdet = np.where(sign > 0, logdet, np.log(eps))
    cond_proxy = tr * np.exp(-logdet / K)
    return float(np.median(cond_proxy)) if cond_proxy.size else 1.0


def masked_stat(x: Optional[np.ndarray], mask: np.ndarray, default: float = 0.0) -> float:
    """Median over finite values on support; default if empty."""
    if x is None:
        return default
    xx = np.asarray(x)
    m = mask > float(getattr(config, "support_cutoff_eps", 0.0))
    vals = xx[m]
    if vals.size == 0:
        return default
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if vals.size else default
