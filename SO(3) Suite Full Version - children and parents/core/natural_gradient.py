# -*- coding: utf-8 -*-
"""
Natural-gradient utilities (Gaussian fields) and Lie-geometry helpers.

- No per-agent caches. Central caches (CacheHub) are used only via omega helpers.
- Strict SPD everywhere (full Σ, no diagonal shortcuts).
- Irrep-agnostic: supply generators for the target fiber K.

Author: c&c
"""
from __future__ import annotations

import numpy as np
import core.config as config

from core.numerical_utils import sanitize_sigma
from core.get_generators import get_generators
from core.numerical_utils import safe_batch_inverse
# omega helpers (all cache-hub–compatible)
from core.omega import d_exp_phi_exact
from transport.transport_cache import E_grid as tc_E_grid
from core.numerical_utils import safe_omega_inv

from transport.transport_cache import Fisher_blocks


# -----------------------------------------------------------------------------
# Natural gradient in Gaussian family (μ, Σ)
# -----------------------------------------------------------------------------

def apply_natural_gradient_batch(
    mu_field,
    sigma_field,
    grad_mu_field,
    grad_sigma_field,
    mask=None,
):
    """
    Vectorized natural gradient for Gaussian fields:
        δμ = -Σ ∇_μ L
        δΣ = -2 Σ sym(∇_Σ L) Σ

    Args
    ----
    mu_field         : (H,W,K)
    sigma_field      : (H,W,K,K) SPD (full)
    grad_mu_field    : (H,W,K)
    grad_sigma_field : (H,W,K,K)
    mask             : (H,W) bool or float weights (optional)

    Returns
    -------
    delta_mu   : (H,W,K)
    delta_sigma: (H,W,K,K)
    """
    H, W, K = mu_field.shape
    N = H * W
    eps = float(getattr(config, "eps", 1e-8))
    clip_sigma = getattr(config, "gradient_clip_sigma", None)
    clip_mu    = getattr(config, "gradient_clip_mu",    None)

    mu   = np.asarray(mu_field,       np.float32).reshape(N, K)
    Sig  = np.asarray(sigma_field,    np.float32).reshape(N, K, K)
    gmu  = np.asarray(grad_mu_field,  np.float32).reshape(N, K)
    gSig = np.asarray(grad_sigma_field, np.float32).reshape(N, K, K)

    # Strict SPD sanitize once
    Sig = sanitize_sigma(Sig, strict=True)

    # Symmetrize ∇_Σ
    gSig = 0.5 * (gSig + np.swapaxes(gSig, -1, -2))

    # Natural steps
    dmu = -np.einsum("nik,nk->ni", Sig, gmu, optimize=True)
    tmp = np.einsum("nij,njk->nik", Sig, gSig, optimize=True)
    dS  = -2.0 * np.einsum("nij,njk->nik", tmp, Sig, optimize=True)
    dS  = 0.5 * (dS + np.swapaxes(dS, -1, -2))  # symmetry guard

    # Mask / weights
    if mask is None:
        active = np.ones(N, dtype=bool)
        wts = None
    else:
        m = np.asarray(mask).reshape(N)
        if m.dtype == bool:
            active = m
            wts = None
        else:
            active = m > float(getattr(config, "support_cutoff_eps", 0.0))
            wts = m.astype(np.float32, copy=False)

    # Global norm clipping (optional)
    if clip_sigma is not None:
        ns = np.linalg.norm(dS.reshape(N, -1), axis=1)
        cm = (ns > clip_sigma) & active
        scale = np.ones(N, dtype=np.float32)
        scale[cm] = clip_sigma / np.maximum(ns[cm], eps)
        dS *= scale[:, None, None]

    if clip_mu is not None:
        nm = np.linalg.norm(dmu, axis=1)
        cm = (nm > clip_mu) & active
        scale = np.ones(N, dtype=np.float32)
        scale[cm] = clip_mu / np.maximum(nm[cm], eps)
        dmu *= scale[:, None]

    if wts is not None:
        dmu *= wts[:, None]
        dS  *= wts[:, None, None]

    # Zero outside support
    if not active.all():
        ia = ~active
        dmu[ia] = 0.0
        dS[ia]  = 0.0

    dmu = np.nan_to_num(dmu, nan=0.0, posinf=0.0, neginf=0.0).reshape(H, W, K)
    dS  = np.nan_to_num(dS,  nan=0.0, posinf=0.0, neginf=0.0).reshape(H, W, K, K)
    return dmu, dS




# -----------------------------------------------------------------------------
# Fisher metric (per-point) for φ / φ̃ fibers; inverse on support
# -----------------------------------------------------------------------------

def QQcompute_inverse_fisher_field_natural(agent, generators, field="phi", eps=1e-8):
    """
    Compute the per-point inverse Fisher metric for a Lie-algebra field on an agent:

        G_ab(x) = Tr[ Σ(x)^{-1} G^a Σ(x)^{-1} G^b ]   (a,b = 1..d)

    Returns an array (H,W,d,d) with zeros outside support.

    Parameters
    ----------
    agent      : agent with fields and mask
    generators : (d,K,K) Lie algebra basis for the chosen fiber
    field      : "phi" (belief fiber) or "phi_model" (model fiber)
    eps        : small reg. for inverses
    """
    # pick Σ for the chosen fiber
    if field == "phi":
        Sigma = np.asarray(agent.sigma_q_field, np.float32)
    elif field == "phi_model":
        Sigma = np.asarray(agent.sigma_p_field, np.float32)
    else:
        raise ValueError("field must be 'phi' or 'phi_model'")

    mask = (np.asarray(agent.mask) > float(getattr(config, "support_cutoff_eps", 1e-4)))
    Sigma = sanitize_sigma(Sigma, strict=True)
    H, W, K, _ = Sigma.shape
    d = int(generators.shape[0])

    out = np.zeros((H, W, d, d), dtype=np.float32)
    if not np.any(mask):
        return out

    # inverse Σ per-point (vectorized)
    
    Sinv = safe_batch_inverse(Sigma)  # (H,W,K,K)

    # T_a = Σ^{-1} G^a Σ^{-1}
    T = np.einsum("...ij,ajk->...aik", Sinv, generators, optimize=True)
    T = np.einsum("...aik,...kl->...ail", T,    Sinv,       optimize=True)

    # G_ab = Tr(T_a G^b)
    G_ab = np.einsum("...aij,bji->...ab", T, generators, optimize=True)
    G_ab = 0.5 * (G_ab + np.swapaxes(G_ab, -1, -2))

    # Invert small d×d block per masked point
    eye_d = np.eye(d, dtype=np.float32)
    flat_G   = G_ab.reshape(-1, d, d)
    flat_out = out.reshape(-1, d, d)
    idx = np.flatnonzero(mask.reshape(-1))

    for m in idx:
        M = flat_G[m]
        try:
            flat_out[m] = np.linalg.inv(M + eps * eye_d)
        except np.linalg.LinAlgError:
            flat_out[m] = eye_d  # safe fallback
    return out

def inverse_fisher_metric_field(ctx, agent, generators, *, which="q", eps=1e-8, tol=None):
    """
    Return per-pixel inverse Fisher metric for the Lie-algebra field (φ or φ̃):
        G_ab = Tr[ Σ^{-1} G^a Σ^{-1} G^b ], then (G + Gᵀ)/2, and invert per pixel.
    Shapes: Sigma (...,K,K), generators (d,K,K) -> out (H,W,d,d).
    """
    import numpy as np

    # --- pull μ, Σ for chosen fiber
    Sig = np.asarray(agent.sigma_q_field if which=="q" else agent.sigma_p_field, np.float32)
    mask = (np.asarray(agent.mask) > float(getattr(config, "support_cutoff_eps", 1e-4)))
    H, W, K, _ = Sig.shape
    d = int(generators.shape[0])
    Gens = np.asarray(generators, np.float32)  # (d,K,K)

    # --- cache key (versioned by Σ and generators)
    C = getattr(ctx, "cache", None)
    cfg = getattr(ctx, "config", {}) if hasattr(ctx, "config") else {}
    step = int(getattr(ctx, "global_step", -1))

    # helpers (same logic as your CacheHub)
    def _shape_sig(a): return tuple(int(x) for x in np.asarray(a).shape)
    def _digest(a):
        a = np.asarray(a, np.float32)
        if tol and tol > 0:
            a = np.round(a / tol) * tol
        return np.sha256(a.tobytes()).hexdigest() if hasattr(np, "sha256") else __import__("hashlib").sha256(a.tobytes()).hexdigest()
    import hashlib, json
    def _cfg_md5(c):
        keys = ["group_name","phi_clip","periodic_wrap","so3_irreps_are_orthogonal","mask_edge_soften_sigma"]
        s = json.dumps({k: c.get(k, None) for k in keys}, sort_keys=True, separators=(",",":"))
        return hashlib.md5(s.encode()).hexdigest()

    key = ("FisherMetricInv", which, int(getattr(agent,"id",id(agent))), step,
           _shape_sig(Sig), _cfg_md5(cfg), _digest(Sig), _digest(Gens))

    if C is not None:
        out_cached = C.get("fisher", key)
        if out_cached is not None:
            return out_cached

    # --- get Σ^{-1} (precision) from Fisher cache if present
    Prec = None
    try:
        
        Fb = Fisher_blocks(ctx, agent, which=which, tol=tol)
        Prec = np.asarray(Fb["precision"], np.float32)  # (H,W,K,K)
    except Exception:
        # Fallback: stable inverse here
        try:
            # Cholesky-based precision
            L = np.linalg.cholesky(Sig)                       # (...,K,K)
            Linv = np.linalg.inv(L)
            Prec = Linv.swapaxes(-1,-2) @ Linv               # Σ^{-1}
        except Exception:
            Prec = np.linalg.inv(Sig)

    # --- form T_a = Σ^{-1} G^a Σ^{-1}  (vectorized)
    # First multiply on the right: Prec @ G^aᵀ, then transpose back to keep matmul-friendly shapes
    # But einsum is straightforward and fast enough:
    T = np.einsum("...ij,ajk->...aik", Prec, Gens, optimize=True)     # (...,d,K,K)
    T = np.einsum("...aik,...kl->...ail", T, Prec, optimize=True)     # (...,d,K,K)

    # --- G_ab = Tr(T_a G^b)
    Gab = np.einsum("...aij,bji->...ab", T, Gens, optimize=True)      # (...,d,d)
    # Symmetrize
    Gab = 0.5*(Gab + np.swapaxes(Gab, -1, -2))

    # --- invert d×d SPD blocks in batch (masked)
    # Use eigh (SPD) to avoid per-pixel Python loop
    flat = Gab.reshape(-1, d, d)
    mflat = mask.reshape(-1)
    out = np.zeros_like(flat)
    if np.any(mflat):
        M = flat[mflat]                                              # (M,d,d)
        # Regularize for safety
        # eigh: M = Q Λ Qᵀ, invert Λ + eps
        w, V = np.linalg.eigh(M)
        w = np.clip(w, eps, None)
        Minv = (V / w[...,None,:]) @ np.swapaxes(V, -1, -2)          # (M,d,d)
        out[mflat] = Minv
    # zeros outside mask by construction
    out = out.reshape(H, W, d, d).astype(np.float32, copy=False)

    if C is not None:
        C.put("fisher", key, out)
    return out




def compute_fisher_geometry_gradient(agent_i, agents, params, field="phi", *, step_tag=None, ctx=None):
    """
    ∇_{φ_i}  Σ_j  || G_i  − Ω_ij G_j Ω_ijᵀ ||_F^2,   with  Ω_ij = E_i · E_j^{-1},
    using exact ∂E_i/∂φ_i^a.  E_x are pulled from the central transport cache.

    Notes
    -----
    - G_x = Σ_x^{-1}, fiber-aware (q or p).
    - If `ctx` is provided, E-grids come from tc_E_grid(ctx, ·, which), and E_j^{-1}
      is obtained via transpose (SO(3) orthogonal irreps) or a safe inverse.
    - step_tag is no longer needed here; cache scoping is handled by CacheHub.
    """
    # --- choose fiber + weights ---
    if field == "phi":
        phi_i      = np.asarray(agent_i.phi,            np.float32)
        Sigma_i    = np.asarray(agent_i.sigma_q_field,  np.float32)
        gen        = getattr(agent_i, "generators_q", None)
        fisher_w   = float(params.get("fisher_geometry_weight",
                                      getattr(config, "fisher_geometry_weight", 0.0)))
        sigma_name = "sigma_q_field"
        phi_name   = "phi"
        which_fiber = "q"
    elif field == "phi_model":
        phi_i      = np.asarray(agent_i.phi_model,      np.float32)
        Sigma_i    = np.asarray(agent_i.sigma_p_field,  np.float32)
        gen        = getattr(agent_i, "generators_p", None)
        fisher_w   = float(params.get("fisher_model_geometry_weight",
                                      getattr(config, "fisher_model_geometry_weight", 0.0)))
        sigma_name = "sigma_p_field"
        phi_name   = "phi_model"
        which_fiber = "p"
    else:
        raise ValueError("field must be 'phi' or 'phi_model'")

    if fisher_w <= 0.0:
        return np.zeros_like(phi_i, dtype=np.float32)

    if gen is None:
        gen, _ = get_generators(getattr(config, "group_name", "so3"),
                                Sigma_i.shape[-1], return_meta=True)

    mask_i = (np.asarray(agent_i.mask) >
              float(getattr(config, "support_cutoff_eps", 1e-4)))
    if not np.any(mask_i):
        return np.zeros_like(phi_i, dtype=np.float32)

    H, W, d = phi_i.shape
    K = Sigma_i.shape[-1]
    group = str(getattr(config, "group_name", "so3")).lower()
    so3_orth = bool(getattr(config, "so3_irreps_are_orthogonal", True)) and (group == "so3")

    # --- E_i from central cache (or raise if ctx missing) ---
    if ctx is None or getattr(ctx, "cache", None) is None:
        raise RuntimeError("compute_fisher_geometry_gradient requires a runtime ctx with a CacheHub")
    E_i = tc_E_grid(ctx, agent_i, which=which_fiber).astype(np.float32, copy=False)   # (H,W,K,K)

    # exact derivative list [a] : (H,W,K,K), reusing E_i to avoid recompute
    dE_list = d_exp_phi_exact(phi_i, gen, exp_phi_all=E_i)

    # --- Σ_i^{-1} = G_i ---
    Sigma_i = sanitize_sigma(Sigma_i, strict=True)
    Sigma_i_inv = safe_batch_inverse(Sigma_i)        # (H,W,K,K)
    G_i = Sigma_i_inv

    grad = np.zeros_like(phi_i, dtype=np.float32)
    Ei_flat  = E_i.reshape(-1, K, K)
    G_i_flat = G_i.reshape(-1, K, K)

    # neighbors only (expect agent_i.neighbors supplied)
    nb_list = getattr(agent_i, "neighbors", []) or []
    for nb in nb_list:
        j = nb["id"]
        agent_j = agents[j]
        mask_j  = (np.asarray(agent_j.mask) >
                   float(getattr(config, "support_cutoff_eps", 1e-4)))
        overlap = (mask_i & mask_j)
        if not np.any(overlap):
            continue

        idx = np.flatnonzero(overlap.reshape(-1))
        if idx.size == 0:
            continue

        # --- E_j and its inverse from the cache ---
        E_j  = tc_E_grid(ctx, agent_j, which=which_fiber).astype(np.float32, copy=False)  # (H,W,K,K)
        if so3_orth:
            E_j_inv = np.swapaxes(E_j, -1, -2)
        else:
            E_j_inv = safe_omega_inv(E_j).astype(np.float32, copy=False)

        Ei_idx   = Ei_flat[idx]                                     # (M,K,K)
        Ej_inv_i = E_j_inv.reshape(-1, K, K)[idx]                   # (M,K,K)

        # Ω_ij = E_i · E_j^{-1}
        Omega   = np.einsum("mik,mkj->mij", Ei_idx, Ej_inv_i, optimize=True)  # (M,K,K)
        Omega_T = np.transpose(Omega, (0, 2, 1))

        # G_j = Σ_j^{-1}
        Sigma_j = sanitize_sigma(np.asarray(getattr(agent_j, sigma_name), np.float32), strict=True)
        G_j_all = safe_batch_inverse(Sigma_j.reshape(-1, K, K))
        G_j     = G_j_all[idx]                                      # (M,K,K)

        # Δ = G_i - Ω G_j Ωᵀ
        G_i_idx = G_i_flat[idx]                                     # (M,K,K)
        Gj_rot  = np.einsum("mik,mkl,mlj->mij", Omega, G_j, Omega_T, optimize=True)
        Delta   = G_i_idx - Gj_rot                                   # (M,K,K)

        # dΩ/dφ_i^a = (dE_i^a) · E_j^{-1}
        for a in range(d):
            dEi   = np.asarray(dE_list[a], np.float32).reshape(-1, K, K)[idx]  # (M,K,K)
            dOm   = np.einsum("mik,mkj->mij", dEi, Ej_inv_i, optimize=True)   # (M,K,K)
            dOm_T = np.transpose(dOm, (0, 2, 1))

            # d(G_rot) = (dΩ) G_j Ωᵀ + Ω G_j (dΩ)ᵀ
            dGj_rot = (np.einsum("mik,mkl,mlj->mij", dOm, G_j, Omega_T, optimize=True) +
                       np.einsum("mik,mkl,mlj->mij", Omega, G_j, dOm_T, optimize=True))

            # dL/dφ_i^a = -2 ⟨Δ, dG_rot⟩
            dL_a = -2.0 * np.einsum("mij,mij->m", Delta, dGj_rot, optimize=True).astype(np.float32, copy=False)
            grad.reshape(-1, d)[idx, a] += dL_a

    grad[~mask_i] = 0.0
    grad *= np.float32(fisher_w)
    return grad
