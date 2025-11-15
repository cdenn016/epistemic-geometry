# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 18:56:28 2025

@author: chris and christine
"""

from __future__ import annotations
import numpy as np
import core.config as config
from transport.preprocess_utils import preprocess_all_agents

from scipy.linalg import eigh, expm

from core.omega import exp_lie_algebra_irrep
from transport.bundle_morphism_utils import have_parent_morphisms, init_parent_morphisms

from updates.update_refresh_utils import list_parents


_EPS = 1e-6


# --- add below your existing helpers inside bundle_coarsegrain_init.py ---
def _soft_support(mask_2d: np.ndarray) -> np.ndarray:
    
    tau = float(getattr(config, "support_cutoff_eps", 1e-3))
    return (np.asarray(mask_2d, np.float32) > tau)

# --- put this in parent_spawn.py (or your CG module) ---

def _periodic_parent_coarsegrain(ctx, children, Gq, Gp, params, *, level: int = 1):
    """
    Run canonical coarse-graining for every existing parent, on a fixed period.

    Runtime-cache design:
      - Preprocess ONLY children (ensures φ/φ̃, Φ/Φ̃, μ/Σ are sane).
      - Do NOT prebuild Λ caches; Lambda_up(...) pulls from the central cache.
      - Each parent is coarse-grained independently; CG warms exp grids as needed.
    """

    # level-aware parent fetch
    parents = list_parents(ctx, level=level)
    if not parents:
        return {"cg_ok": 0, "cg_fail": 0}

    # schedule
    step_now = int(getattr(ctx, "global_step", 0))
    period = int(getattr(config, "parent_cg_period", params.get("parent_cg_period", 50)))
    if period <= 0 or (step_now % period) != 0:
        return {"cg_ok": 0, "cg_fail": 0}

    cg_eps = float(getattr(config, "cg_eps", 1e-6))
    # CG should gate at soft support level, not a high display τ
    cg_tau = float(getattr(config, "support_tau",
                           getattr(config, "support_cutoff_eps", 1e-4)))

    # 1) preprocess only children (keeps parents lazy/fixed-up inside CG)
    child_only = [c for c in (children or []) if c is not None]
    if child_only:
        try:
            preprocess_all_agents(child_only, params, G_q=Gq, G_p=Gp, ctx=ctx)
        except Exception as e:
            print(f"[CG] preprocess(children) failed: {e}")

    # 2) coarse-grain per-parent
    cg_ok, cg_fail = 0, 0
    # Skip truly empty masks; use support τ here as well
    write_tau = float(getattr(config, "support_tau",
                              getattr(config, "support_cutoff_eps", 1e-4)))

    for P in parents:
        try:
            m = np.asarray(getattr(P, "mask", None), np.float32)
            if m is None or m.size == 0 or not np.any(m > write_tau):
                continue

            coarsegrain_parent_from_children(
                P, children,
                eps=cg_eps,
                mask_tau=cg_tau,
                G_q=Gq,
                G_p=Gp,
                ctx=ctx,
            )
            cg_ok += 1
        except Exception as e:
            pid = getattr(P, "id", "?")
            print(f"[CG] periodic coarsegrain failed for parent {pid}: {e}")
            cg_fail += 1

    return {"cg_ok": cg_ok, "cg_fail": cg_fail}


# =========================
# main coarsegrainer
# =========================

        



def coarsegrain_parent_from_children(
    P, children, *, eps=1e-6, mask_tau=None, G_q=None, G_p=None, ctx=None
):
    
    _EPS = 1e-6

    # --- local helpers -----------------------------------------------------------
    def ensure_hw(m, H, W):
        m = np.asarray(m)
        if m.ndim == 2 and m.shape == (H, W):
            return m
        if m.size == H * W:
            return m.reshape(H, W)
        raise ValueError(f"mask shape {m.shape} incompatible with {(H,W)}")

    def apply_sigma_background(
        Sigma, support, *, s_bg=1e5, feather_px=2, jitter=1e-6
    ):
        """
        Log–Euclidean blend between Sigma (inside) and c·I (outside), periodic feathering.
        Keeps matrices well-behaved across the boundary.
        """
        
        
    
        Sigma = np.asarray(Sigma, np.float64)
        H, W, K, _ = Sigma.shape
        sup = np.asarray(support, bool).reshape(H, W)
    
        # periodic feather alpha \in [0,1]
        alpha = sup.astype(np.float64)
        for _ in range(int(max(0, feather_px))):
            alpha = (alpha
                     + np.roll(alpha, 1, 0) + np.roll(alpha, -1, 0)
                     + np.roll(alpha, 1, 1) + np.roll(alpha, -1, 1)) / 5.0
        alpha = np.clip(alpha, 0.0, 1.0)
        a = alpha[..., None, None]  # (H,W,1,1)
    
        # target: c·I ; log(c·I) = (log c)·I
        c = float(s_bg)
        c = 1e-30 if not np.isfinite(c) or c <= 0 else min(c, 1e30)
        log_cI = np.log(c) * np.eye(K, dtype=np.float64)[None, None, ...]
    
        # logm(Sigma) via eigen-decomp (SPD → real eigs)
        Sf = Sigma.reshape(-1, K, K)
        logS = np.empty_like(Sf)
        for i in range(Sf.shape[0]):
            # symmetrize + jitter
            S = 0.5 * (Sf[i] + Sf[i].T) + jitter * np.eye(K)
            w, Q = eigh(S, turbo=True)
            w = np.clip(w, 1e-12, None)
            logS[i] = (Q * np.log(w)) @ Q.T
        logS = logS.reshape(H, W, K, K)
    
        # blend in log-space, then exp
        L = a * logS + (1.0 - a) * log_cI
        # expm per pixel (K is usually small; loop is OK)
        Lf = L.reshape(-1, K, K)
        out = np.empty_like(Lf)
        for i in range(Lf.shape[0]):
            out[i] = expm(Lf[i])
        out = out.reshape(H, W, K, K)
        # symmetrize + tiny jitter
        out = 0.5 * (out + np.swapaxes(out, -1, -2)) + jitter * np.eye(K)[None, None, ...]
        return out.astype(np.float32)

    # --- optional: import your retraction; inline fallback if missing ------------
    def _fallback_retract_so3_principal(phi, margin=1e-3):
        """
        Minimal axis-angle principal-ball projector: clamp ||phi|| to (pi - margin).
        phi: (..., 3)
        """
        phi = np.asarray(phi, np.float32)
        theta = np.linalg.norm(phi, axis=-1, keepdims=True)  # (...,1)
        pi = np.pi
        max_th = (pi - float(margin))
        # avoid divide-by-zero; safe scale
        scale = np.where(theta > max_th, max_th / np.maximum(theta, 1e-12), 1.0).astype(np.float32)
        return phi * scale

    _retract_so3_principal_fn = _fallback_retract_so3_principal

    # --- gather & guards ---------------------------------------------------------
    H, W = np.asarray(P.mask).shape[:2]
    mu_q = getattr(P, "mu_q_field", None)
    mu_p = getattr(P, "mu_p_field", None)
    sg_q = getattr(P, "sigma_q_field", None)
    sg_p = getattr(P, "sigma_p_field", None)
    phi  = getattr(P, "phi", None)
    phi_m = getattr(P, "phi_model", getattr(P, "phi_tilde", None))

    if (mu_q is None) or (mu_p is None) or (sg_q is None) or (sg_p is None) or (phi is None):
        return {"ok": False, "why": "parent_fields_missing"}

    Kq = int(mu_q.shape[-1]); Kp = int(mu_p.shape[-1])

    stau = float(getattr(config, "support_tau", 0.01))
    supp = _effective_cg_support(P, children, H, W, tau=(mask_tau if mask_tau is not None else stau))
    if not np.any(supp):
        return {"ok": False, "why": "empty_support"}
    supp = ensure_hw(supp, H, W)

    assigned = getattr(P, "assigned_child_ids", set()) or set()
    use_assigned_only = (len(assigned) > 0)

    w_list, mq_list, mp_list, Sq_list, Sp_list, ph_list, phm_list = [], [], [], [], [], [], []
    for C in children:
        cid = int(getattr(C, "id", -1))
        if use_assigned_only and (cid not in assigned):
            continue
        cm = np.asarray(getattr(C, "mask", 0.0), np.float32)
        w = np.clip(cm, 0.0, 1.0) * supp.astype(np.float32)
        if not np.any(w):
            continue

        w_list.append(w[..., None])  # (H,W,1)
        mq_list.append(np.asarray(getattr(C, "mu_q_field"),    np.float32))
        mp_list.append(np.asarray(getattr(C, "mu_p_field"),    np.float32))
        Sq_list.append(np.asarray(getattr(C, "sigma_q_field"), np.float32))
        Sp_list.append(np.asarray(getattr(C, "sigma_p_field"), np.float32))
        ph_list.append(np.asarray(getattr(C, "phi"),           np.float32))
        child_phi_m = getattr(C, "phi_model", getattr(C, "phi_tilde", None))
        phm_list.append(np.zeros((H, W, 3), np.float32) if child_phi_m is None
                        else np.asarray(child_phi_m, np.float32))

    if len(w_list) == 0:
        return {"ok": False, "why": "no_child_contrib"}

    # weights: (H,W,Nc) normalized on covered pixels
    Wstack = np.concatenate(w_list, axis=-1)                 # (H,W,Nc)
    sumW   = np.sum(Wstack, axis=-1, keepdims=True)          # (H,W,1)
    Wnorm  = Wstack / np.maximum(sumW, _EPS)                 # (H,W,Nc)

    def _wmean(stack_fields):
        X = np.stack(stack_fields, axis=-1)                  # (..., D, Nc) or (H,W,K,K,Nc)
        return np.sum(X * Wnorm[..., None, :], axis=-1)

    def _wmean_spd(stack_fields):
        X = np.stack(stack_fields, axis=-1)                  # (H,W,K,K,Nc)
        Y = np.sum(X * Wnorm[..., None, None, :], axis=-1)   # (H,W,K,K)
        Y = 0.5 * (Y + np.swapaxes(Y, -1, -2))
        I = np.eye(Y.shape[-1], dtype=np.float32)[None, None, ...]
        return (Y + 1e-5 * I).astype(np.float32)

    # --- means on support --------------------------------------------------------
    mu_q_est = _wmean(mq_list)                               # (H,W,Kq)
    mu_p_est = _wmean(mp_list)                               # (H,W,Kp)
    sg_q_est = _wmean_spd(Sq_list)                           # (H,W,Kq,Kq)
    sg_p_est = _wmean_spd(Sp_list)                           # (H,W,Kp,Kp)

    # --- φ / φ̃ aggregation (fast small-angle → exact Karcher) -------------------
    w_maps = [Wstack[..., i] for i in range(Wstack.shape[-1])]   # list of (H,W)
    phi_est = aggregate_phi_field(P=P, which="phi",
                                  phi_list=ph_list, w_list=w_maps, scalar_wts=None,
                                  G=G_q, H=H, W=W)
    phi_m_est = aggregate_phi_field(P=P, which="phi_model",
                                    phi_list=phm_list, w_list=w_maps, scalar_wts=None,
                                    G=G_p, H=H, W=W)

    # --- periodic harmonic inpaint on support -----------------------------------
    cover = (sumW[..., 0] > 0.0)
    mu_q_est = _harmonic_inpaint_vec(mu_q_est, cover, supp, iters=6)
    mu_p_est = _harmonic_inpaint_vec(mu_p_est, cover, supp, iters=6)
    sg_q_est = _harmonic_inpaint_spd(sg_q_est, cover, supp, iters=6, jitter=1e-5)
    sg_p_est = _harmonic_inpaint_spd(sg_p_est, cover, supp, iters=6, jitter=1e-5)
    if phi_est is not None:
        phi_est = _harmonic_inpaint_vec(phi_est, cover, supp, iters=4)
    if phi_m_est is not None:
        phi_m_est = _harmonic_inpaint_vec(phi_m_est, cover, supp, iters=4)

    # --- NEW: post-CG SO(3) principal-ball retraction for φ and φ̃ --------------
    try:
        margin = float(getattr(config, "so3_principal_margin", 1e-3))
    except Exception:
        margin = 1e-3

    if phi_est is not None:
        # sanitize NaNs/Infs before retract
        phi_est = np.where(np.isfinite(phi_est), phi_est, 0.0).astype(np.float32)
        phi_est = _retract_so3_principal_fn(phi_est, margin=margin)

    if phi_m_est is not None:
        phi_m_est = np.where(np.isfinite(phi_m_est), phi_m_est, 0.0).astype(np.float32)
        phi_m_est = _retract_so3_principal_fn(phi_m_est, margin=margin)

    # --- NEW: set smooth large-variance background for Σ outside support ---------
    s_bg = float(getattr(config, "sigma_background_var", 1e5))
    feather = int(getattr(config, "sigma_background_feather_px", 2))
    sg_q_full = apply_sigma_background(sg_q_est, supp, s_bg=s_bg, feather_px=feather, jitter=1e-6)
    sg_p_full = apply_sigma_background(sg_p_est, supp, s_bg=s_bg, feather_px=feather, jitter=1e-6)

    # --- commit (Σ: replace full field; μ/φ: write on support only) -------------
    _commit_on_support(P, "mu_q_field",    mu_q_est, supp)
    _commit_on_support(P, "mu_p_field",    mu_p_est, supp)

    # Replace entire covariance fields with blended background (no gaps).
    _commit_on_support(P, "sigma_q_field", sg_q_full, np.ones((H, W), bool))
    _commit_on_support(P, "sigma_p_field", sg_p_full, np.ones((H, W), bool))

    if phi_est is not None:
        _commit_on_support(P, "phi", phi_est, supp)
    if phi_m_est is not None:
        if hasattr(P, "phi_model"):
            _commit_on_support(P, "phi_model", phi_m_est, supp)
        if hasattr(P, "phi_tilde"):
            _commit_on_support(P, "phi_tilde", phi_m_est, supp)

    return {"ok": True, "why": "", "support_px": int(supp.sum())}



def _phi_agg_mode(which: str):
    """
    Decide φ aggregation mode from config.
    Modes:
      - "exact" → always Karcher (group) mean
      - "fast"  → always small-angle algebra mean
      - "auto"  → small-angle if safe, else Karcher
    Keys (checked in order of precedence):
      cg_phi_force_fast[_model], cg_phi_force_exact[_model], cg_phi_mode[_model]
    """
    
    key = "model" if which in ("phi_model", "phi_tilde") else "phi"
    # hard overrides
    if bool(getattr(config, f"cg_{key}_force_fast",  False)):
        return "fast"
    if bool(getattr(config, f"cg_{key}_force_exact", False)):
        return "exact"
    # soft setting
    mode = str(getattr(config, f"cg_{key}_mode", "auto")).lower()
    return "auto" if mode not in ("auto","exact","fast") else mode

# ---- phi aggregation helpers -----------------------------------------------


def aggregate_phi_field(*, P, which, phi_list, w_list, scalar_wts, G, H, W):
  
    if not phi_list:
        return None

    mode = _phi_agg_mode(which)  # <— NEW

    Phi = np.stack([np.asarray(ph, np.float64) for ph in phi_list], axis=0)  # (n,H,W,3)
    n, _, _, d = Phi.shape

    # weights
    if w_list and isinstance(w_list[0], np.ndarray):
        Wts = np.stack([np.asarray(wm, np.float64) for wm in w_list], axis=0)     # (n,H,W)
    else:
        Wts = np.asarray(scalar_wts if scalar_wts else [1.0]*n, np.float64)[:,None,None]
    Wts = np.clip(Wts, 0.0, np.inf)

    # small-angle gate (only when mode=="auto")
    thr_abs    = float(getattr(config, "cg_phi_small_angle_thr",    0.20))
    thr_spread = float(getattr(config, "cg_phi_small_angle_spread", 0.10))
    use_fast = False
    if mode == "fast":
        use_fast = True
    elif mode == "auto":
        norms = np.linalg.norm(Phi, axis=-1)                 # (n,H,W)
        use_fast = (np.nanmax(norms * (Wts > 0)) <= thr_abs) and \
                   (np.nanmax(np.nanstd(norms, axis=0)) <= thr_spread)

    if use_fast or (G is None and mode != "exact"):
        # algebra mean
        # AFTER (pure 2-D; no leading 1 so boolean matches (H,W)):
        Wsum2D  = Wts.sum(axis=0) + 1e-12                   # (H,W)
        Phi_avg = (Wts[..., None] * Phi).sum(axis=0) / Wsum2D[..., None]  # (H,W,3)
        out     = np.zeros((H, W, 3), np.float32)
        nz      = (Wsum2D > 0)                              # (H,W)
        out[nz, ...] = Phi_avg[nz, ...].astype(np.float32)
        return out

    # else: exact Karcher mean in irrep
    return CGH_karcher_mean_phi_irrep(phi_list, w_list, H, W, G, scalar_wts=scalar_wts)


def CGH_karcher_mean_phi_irrep(phi_list, w_list, H, W, G, *,
                               scalar_wts=None, max_iters=10, tol=1e-5, eta=1.0):
    """
    Weighted Karcher/Fréchet mean of φ fields in an SO(3) irrep.
    Inputs:
      - phi_list: list of (H,W,3) float arrays (Lie algebra coords)
      - w_list:   list of (H,W) weights OR None (use scalar_wts)
      - G:        (3,K,K) real generators for this irrep
    Returns:
      - (H,W,3) float32
    """
    
    try:
        from scipy.linalg import logm
    except Exception:
        logm = None

    # ---- validate shapes ---------------------------------------------------------
    n = len(phi_list)
    assert n >= 1, "phi_list empty"
    G = np.asarray(G, np.float64)
    assert G.shape[0] == 3 and G.ndim == 3, "G must be (3,K,K)"
    K = G.shape[-1]

    Phi = np.stack([np.asarray(ph, np.float64) for ph in phi_list], axis=0)  # (n,H,W,3)
    assert Phi.shape[1] == H and Phi.shape[2] == W and Phi.shape[3] == 3, "phi_list bad shapes"

    # ---- weights: normalize to sum=1 per-pixel or globally -----------------------
    if w_list and isinstance(w_list[0], np.ndarray):
        Wts = np.stack([np.asarray(wm, np.float64) for wm in w_list], axis=0)  # (n,H,W)
    else:
        sw = np.asarray(scalar_wts if scalar_wts is not None else [1.0]*n, np.float64)
        Wts = np.broadcast_to(sw[:, None, None], (n, H, W))

    Wts = np.clip(Wts, 0.0, np.inf)
    Wsum = Wts.sum(axis=0) + 1e-12                    # (H,W)
    Wts  = Wts / Wsum                                  # normalize (n,H,W)

    # ---- init at (weighted) algebraic mean --------------------------------------
    Phi_mean = (Wts[..., None] * Phi).sum(axis=0)     # (H,W,3)

    if logm is None:
        return Phi_mean.astype(np.float32)

    # ---- precompute Gram matrix for projection in this irrep ---------------------
    # inner product: <A,B>_F := 0.5 * tr(A^T B)
    G_flat = G.reshape(3, K*K)
    M = 0.5 * (G_flat @ G_flat.T)                     # (3,3)
    # Regularize in case of near-singularity (shouldn’t happen, but be safe)
    M += 1e-12 * np.eye(3)
    M_inv = np.linalg.inv(M)

    # helper: project a KxK matrix L onto algebra basis -> 3-vector c
    # b_a = 0.5 * tr(G_a^T L) = 0.5 * <G_a, L>_F
    def proj_to_algebra(L):
        # L: (...,K,K)
        b = 0.5 * np.einsum("aij,...ij->...a", G, L)      # (...,3)
        # c = M^{-1} b
        return np.tensordot(b, M_inv, axes=([-1], [0]))   # (...,3)

    # ---- iterate geodesic averaging ---------------------------------------------
    # we’ll operate on a flat view for the logm loop
    for _ in range(int(max_iters)):
        R_mean = exp_lie_algebra_irrep(Phi_mean, G).astype(np.float64)  # (H,W,K,K)
        R_invT = np.swapaxes(R_mean, -1, -2)                            # orthogonal inverse

        # accumulate tangent updates (H,W,3)
        updates = np.zeros_like(Phi_mean, np.float64)

        for k in range(n):
            Rk = exp_lie_algebra_irrep(Phi[k], G).astype(np.float64)    # (H,W,K,K)
            Delta = np.einsum("...ik,...kj->...ij", R_invT, Rk)         # (H,W,K,K)

            # matrix log per pixel (complex -> real skew)
            # work on a flat view to avoid repeated reshape()
            Delta_f = Delta.reshape(H*W, K, K)
            L_f = np.empty_like(Delta_f, dtype=np.complex128)
            for i in range(Delta_f.shape[0]):
                L_f[i] = logm(Delta_f[i])
            L = L_f.reshape(H, W, K, K)
            L = np.real_if_close(L, tol=1e-9).astype(np.float64)

            # force skew-symmetry in numerical sense (so(3) image)
            L = 0.5 * (L - np.swapaxes(L, -1, -2))

            # project to algebra coefficients with proper Gram correction
            proj = proj_to_algebra(L)                                  # (H,W,3)
            updates += (Wts[k, ..., None] * proj)

        step = eta * updates                                          # (H,W,3)
        Phi_next = Phi_mean + step

        # stopping criterion (handle NaNs gracefully)
        diff = np.linalg.norm(Phi_next - Phi_mean, axis=-1)           # (H,W)
        diff[np.isnan(diff)] = np.inf
        if np.nanmax(diff) < tol:
            Phi_mean = Phi_next
            break
        Phi_mean = Phi_next

    return Phi_mean.astype(np.float32)

def fast_mean_phi_small_angle(Pmat, *, weights=None, tau_abs: float, tau_spread: float):
    """
    Small-angle mean of algebra 3-vectors φ with accept/reject gate.
    Pmat: (n,H,W,3) or (n,3)
    weights: (n,) or None
    Returns:
      (ok, phi_bar) where:
        - if (n,3): ok is bool; phi_bar is (3,) or None
        - if (n,H,W,3): ok is (H,W) bool mask; phi_bar is (H,W,3) with zeros where not ok
    """
    
    Pmat = np.asarray(Pmat, np.float64)

    if Pmat.ndim == 2:  # (n,3)
        norms = np.linalg.norm(Pmat, axis=-1)
        ok = (norms.max() <= tau_abs) and (norms.std() <= tau_spread)
        if not ok:
            return False, None
        if weights is None:
            phi_bar = Pmat.mean(axis=0)
        else:
            w = np.asarray(weights, np.float64)
            w = np.clip(w, 0, np.inf)
            w = w / (w.sum() + 1e-12)
            phi_bar = (Pmat * w[:, None]).sum(axis=0)
        return True, phi_bar.astype(np.float32)

    # spatial: (n,H,W,3)
    assert Pmat.ndim == 4 and Pmat.shape[-1] == 3
    n, H, W, _ = Pmat.shape

    norms = np.linalg.norm(Pmat, axis=-1)                 # (n,H,W)
    ok = (norms.max(axis=0) <= tau_abs) & (norms.std(axis=0) <= tau_spread)  # (H,W)

    if weights is None:
        phi_bar = Pmat.mean(axis=0)                       # (H,W,3)
    else:
        w = np.asarray(weights, np.float64)
        w = np.clip(w, 0, np.inf)
        w = w / (w.sum() + 1e-12)
        phi_bar = (Pmat * w.reshape(n, 1, 1, 1)).sum(axis=0)

    # zero-out pixels that fail the gate
    phi_bar = np.where(ok[..., None], phi_bar, 0.0)
    return ok, phi_bar.astype(np.float32)





def _try_fast_small_angle(P, which, phi_list, w_list, scalar_wts):
    """
    Small-angle fast mean using per-pixel weights ONLY.
    If unavailable/invalid, bail to Karcher.
    """
    
    if not phi_list or getattr(config, "force_exact_phi_mean", False):
        return False, None

    tau_abs = float(getattr(config, "phi_small_abs", 5e-2))
    tau_sp  = float(getattr(config, "phi_small_spread", 5e-2))

    Pmat = np.stack(phi_list, axis=0).astype(np.float32)   # (n,H,W,3)
    norms = np.linalg.norm(Pmat, axis=-1)
    if (np.nanmax(norms) > tau_abs) or (np.nanstd(norms) > tau_sp):
        return False, None

    # require per-pixel weights aligned with phi_list entries
    if not w_list or len(w_list) != Pmat.shape[0]:
        return False, None
    try:
        Wts = np.stack(w_list, axis=0).astype(np.float32)  # (n,H,W)
    except Exception:
        return False, None
    if Wts.ndim != 3 or Wts.shape[1:] != Pmat.shape[1:3]:
        return False, None

    Wts = np.clip(Wts, 0.0, np.inf)
    Wsum = Wts.sum(axis=0)                                 # (H,W)
    has_cov = (Wsum > 0)

    phi_bar = (Wts[..., None] * Pmat).sum(axis=0)
    phi_bar = phi_bar / np.maximum(Wsum, 1e-12)[..., None]
    phi_bar[~has_cov, :] = 0.0
    return True, phi_bar.astype(np.float32)




#==============================================================================
#
#          # ========================== bundle_coarsegrain_utils.py ==========================
#                    All helpers used by the orchestrator. Pure functions where possible.
#
#===============================================================================






def _boolean_halo(seed_bool: np.ndarray, iters: int = 2) -> np.ndarray:
    halo = seed_bool.copy()
    for _ in range(int(iters)):
        nbr = ( np.roll(halo,1,0) | np.roll(halo,-1,0)
              | np.roll(halo,1,1) | np.roll(halo,-1,1)
              | np.roll(np.roll(halo,1,0),1,1) | np.roll(np.roll(halo,1,0),-1,1)
              | np.roll(np.roll(halo,-1,0),1,1) | np.roll(np.roll(halo,-1,0),-1,1) )
        halo |= nbr
    return halo

def _effective_cg_support(P, children, H, W, *, tau=None) -> np.ndarray:
    """Return boolean support for CG. Pure boolean ops (no bitwise on floats)."""
    tau = float(1e-6 if tau is None else tau)
    # 1) prefer reconciler-provided support
    if hasattr(P, "cg_support") and isinstance(P.cg_support, np.ndarray):
        S = (np.asarray(P.cg_support, np.float32) > tau)
        if np.any(S):
            return S
    # 2) fallback: assigned union @ support_tau, else seed halo, always include own mask>tau
    stau = float(getattr(config, "support_tau", 0.01))
    S = np.zeros((H, W), dtype=bool)
    assigned = getattr(P, "assigned_child_ids", set()) or set()
    if assigned:
        for c in children:
            if int(getattr(c, "id", -1)) in assigned:
                S |= (np.asarray(c.mask, np.float32) > stau)
    else:
        seed = getattr(P, "seed_mask", None)
        if isinstance(seed, np.ndarray) and seed.size == H * W:
            S |= _boolean_halo((np.asarray(seed, np.float32) >= 0.5), iters=2)
    S |= (np.asarray(getattr(P, "mask", 0.0), np.float32) > stau)
    return S

# --- broadcast-safe helpers (drop-in) ---

def _harmonic_inpaint_vec(V, cover, support, iters=8):
    
    V = np.asarray(V, np.float32)
    H, W, K = V.shape
    m_fix = np.asarray(cover,   bool)
    m_sup = np.asarray(support, bool)

    out = np.zeros_like(V, np.float32)
    # use ellipsis to index all channels and avoid shape mismatches
    out[m_fix, ...] = V[m_fix, ...]
    out[~m_sup, ...] = 0.0

    for _ in range(int(iters)):
        nb  = (np.roll(out,1,0) + np.roll(out,-1,0) + np.roll(out,1,1) + np.roll(out,-1,1)) * 0.25
        upd = (m_sup & ~m_fix)[..., None]            # (H,W,1) → broadcasts to (H,W,K)
        out = np.where(upd, nb, out)
    return out

def _harmonic_inpaint_spd(SG, cover, support, iters=8, jitter=1e-5):
    
    SG = np.asarray(SG, np.float32)
    H, W, K, _ = SG.shape
    m_fix = np.asarray(cover,   bool)
    m_sup = np.asarray(support, bool)

    out = SG.copy()
    out[~m_sup, ...] = 0.0

    for _ in range(int(iters)):
        nb  = (np.roll(out,1,0) + np.roll(out,-1,0) + np.roll(out,1,1) + np.roll(out,-1,1)) * 0.25
        upd = (m_sup & ~m_fix)[..., None, None]      # (H,W,1,1)
        out = np.where(upd, nb, out)

    out = 0.5 * (out + np.swapaxes(out, -1, -2))
    I = np.eye(K, dtype=np.float32)[None, None, ...]
    return out + jitter * I

def _commit_on_support(P, field_name: str, new_val, support):
    """Write new_val where support is True; keep previous elsewhere. Safe for (H,W,K) and (H,W,K,K)."""
    
    nv = np.asarray(new_val, np.float32)
    H, W = nv.shape[:2]
    sup = np.asarray(support, bool).reshape(H, W)

    old = getattr(P, field_name, None)
    if isinstance(old, np.ndarray) and old.shape == nv.shape:
        out = old.copy()
        out[sup, ...] = nv[sup, ...]
    else:
        out = nv
    setattr(P, field_name, out.astype(np.float32))


        
        
        

# ============================ Local helper implementations ============================

def near_identity_mask(L: np.ndarray, tau_L: float) -> np.ndarray:
    """
    Return a boolean mask over the first dimension of L selecting rows where
    the linear map is 'near identity' under Frobenius norm.
    L: (T, K, K)
    tau_L: threshold on ||L - I||_F / ||I||_F (dimensionless)
    """
    T, K, _ = L.shape
    I = np.eye(K, dtype=L.dtype)
    diff = L - I  # (T, K, K)
    num = np.linalg.norm(diff.reshape(T, -1), axis=1)
    den = np.linalg.norm(I.reshape(-1))  # = sqrt(K)
    return (num / max(den, 1.0)) <= float(tau_L)




# === bundle_coarsegrain_init.py: guards for fast/ exact paths =================

def _small_angle_ok(phi_list, *, abs_tau: float, spread_tau: float) -> bool:
    """Cheap predicate: accept fast φ-mean iff magnitudes are small and concentrated."""
    
    if not phi_list:
        return False
    X = np.stack(phi_list, axis=0)   # (n,H,W,3) or (n,3)
    norms = np.linalg.norm(X, axis=-1)
    return (np.nanmax(norms) <= abs_tau) and (np.nanstd(norms) <= spread_tau)


def _near_identity_ok(Lf, tau_L: float) -> bool:
    """
    Accept first-order push iff Λ is near-identity on the rows we will actually use.
    Lf is flattened (T,K,K) rows for the effective pixels.
    """
    
    if Lf is None or Lf.size == 0:
        return False
    T, K, _ = Lf.shape
    I = np.eye(K, dtype=Lf.dtype)
    diff = Lf - I
    # dimensionless Frobenius ratio
    num = np.linalg.norm(diff.reshape(T, -1), axis=1)
    den = np.sqrt(float(K))
    return bool((num <= tau_L * den).all())


# -------- φ/φ̃ aggregation (no helpers dict) ---------------------------------
# in bundle_coarsegrain_init.py (or wherever _try_fast_small_angle lives)


# ========================= /Local helper implementations =========================



# -------- morphisms + generators -------------------------------------------------
def ensure_parent_morphisms_if_missing(P, G_q, G_p, ctx):
    """
    Initializes parent bundle morphisms for newborns if missing.
    """
   
    if not have_parent_morphisms(P):
        init_parent_morphisms(P, G_q, G_p, ctx=ctx)
       


# -------- cache + shapes ---------------------------------------------------------
def resolve_cache_ns(ctx):
    """
    Returns a mutable mapping for ns('misc').
    Falls back to a simple dict if no CacheHub exists.
    """
    if ctx is None:
        return {}
    hub = getattr(ctx, "cache", None)
    return hub.ns("misc") if hub is not None else {}


def compute_core_mask(P, ns_misc, *, core_tau):
    """
    Cached parent 'core' support: mask > max(core_tau, 0.5*peak).
    Keyed by (cg_core, pid, core_tau, peak, H, W).
    """
    H, W = P.mask.shape[:2]
    pid = int(getattr(P, "id", -1))
    peak = float(P.mask.max() if P.mask.size else 0.0)
    key = ("cg_core", pid, float(core_tau), peak, H, W)
    core = ns_misc.get(key)
    if core is None:
        core = (np.asarray(P.mask, np.float32) > max(float(core_tau), 0.5 * peak))
        ns_misc[key] = core
    return core


def collect_child_pairs(P, children):
    """
    Returns filtered list of (child, weight) with weight > 0.
    Uses P.child_weights to determine per-child weight.
    """
    wmap = getattr(P, "child_weights", {}) or {}
    pairs = []
    for c in (children or []):
        if c is None:
            continue
        w = float(wmap.get(int(getattr(c, "id", -1)), 0.0))
        if w > 0.0:
            pairs.append((c, w))
    return pairs


def alloc_fiber_accumulators(H, W, K):
    if K <= 0:
        # Use empty arrays to simplify call sites
        Z = np.zeros((H, W), np.float32)
        return Z, np.zeros((H, W, 0), np.float32), np.zeros((H, W, 0, 0), np.float32)
    Wsum = np.zeros((H, W), np.float32)
    mu_acc = np.zeros((H, W, K), np.float32)
    Sig_acc = np.zeros((H, W, K, K), np.float32)
    return Wsum, mu_acc, Sig_acc


# -------- sparse indexing / tiling ----------------------------------------------
def indices_of(mask_bool):
    return np.flatnonzero(np.asarray(mask_bool, dtype=bool).reshape(-1))


def tile_slices(idx, *, tile=65536):
    for s in range(0, idx.size, tile):
        yield idx[s : s + tile]


# -------- per-(parent,child) cached views ---------------------------------------
def get_flattened_views(ns_misc, P, child, Lq, Lp, Kq, Kp, H, W):
    """
    Returns flattened (view) arrays for μ, Σ, Λ in q/p fibers, cached by (P,c).
    """
    pid = int(getattr(P, "id", -1))
    cid = int(getattr(child, "id", -1))
    key = ("cg_flat", pid, cid, Kq, Kp, H, W)
    cached = ns_misc.get(key)
    if cached is None:
        mu_q_f = child.mu_q_field.reshape(-1, Kq) if Kq else None
        sg_q_f = child.sigma_q_field.reshape(-1, Kq, Kq) if Kq else None
        mu_p_f = child.mu_p_field.reshape(-1, Kp) if Kp else None
        sg_p_f = child.sigma_p_field.reshape(-1, Kp, Kp) if Kp else None
        Lq_f = Lq.reshape(-1, Kq, Kq) if (Kq and Lq is not None) else None
        Lp_f = Lp.reshape(-1, Kp, Kp) if (Kp and Lp is not None) else None
        cached = (mu_q_f, sg_q_f, mu_p_f, sg_p_f, Lq_f, Lp_f)
        ns_misc[key] = cached
    return cached


def get_eff_mask_cached(ns_misc, P, child, tau, H, W, core):
    """
    Effective per-(parent,child) mask: eff = core ∧ (child.mask > τ), cached by (P,c,τ).
    """
    pid = int(getattr(P, "id", -1))
    cid = int(getattr(child, "id", -1))
    key = ("cg_eff", pid, cid, float(tau), H, W)
    eff = ns_misc.get(key)
    if eff is None:
        c_mask = (np.asarray(child.mask, np.float32) > float(tau))
        eff = (core & c_mask)
        ns_misc[key] = eff
    return eff







# -------- finalize μ/Σ into parent fields --------------------------------------
def finalize_fiber_into_parent(*, Wsum, mu_acc, Sig_acc, P_mu, P_Sigma, eps, K):
    """
    Normalizes accumulators and writes μ/Σ into parent field arrays.
    Ensures Σ is SPD via symmetrization + eps·I.
    """
    if K <= 0:
        return
    valid = (Wsum > 0)
    if not np.any(valid):
        return

    den = np.maximum(Wsum[..., None, None], 1e-12).astype(np.float64)
    Sig = (Sig_acc.astype(np.float64) / den)
    Sig = 0.5 * (Sig + np.swapaxes(Sig, -1, -2)) + float(eps) * np.eye(K, dtype=np.float64)

    muP = np.zeros_like(P_mu, dtype=np.float64)
    muP[valid] = (mu_acc.astype(np.float64)[valid] / Wsum[valid, None])

    P_mu[valid]    = muP[valid].astype(np.float32)
    P_Sigma[valid] = Sig[valid].astype(np.float32)









def _log_spd(A, eps):
    """Matrix log for SPD A (KxK). Returns (K,K)."""
    
    A = np.asarray(A)
    A = (A + A.T) * 0.5
    w, U = np.linalg.eigh(A)                      # w: (K,), U: (K,K)
    w = np.clip(w, eps, np.inf)
    # U @ diag(log w) @ U^T  -> implement as (U * logw) @ U^T
    logw = np.log(w).astype(U.dtype)
    L = (U * logw) @ U.T
    return (L + L.T) * 0.5

def _exp_sym(L):
    """Matrix exp for symmetric L (KxK). Returns (K,K)."""
    
    L = np.asarray(L)
    L = (L + L.T) * 0.5
    w, U = np.linalg.eigh(L)                      # w: (K,), U: (K,K)
    expw = np.exp(w).astype(U.dtype)
    E = (U * expw) @ U.T                          # U @ diag(exp w) @ U^T
    return (E + E.T) * 0.5

