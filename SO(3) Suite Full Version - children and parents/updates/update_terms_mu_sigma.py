# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 12:27:32 2025

@author: chris and christine
"""

import numpy as np
import core.config as config
from core.numerical_utils import sanitize_sigma, safe_batch_inverse
from core.natural_gradient import apply_natural_gradient_batch

from transport.transport_cache import Phi
from transport.transport_cache import E_grid, Omega, warm_E

from core.get_generators import get_generators
from core.omega import exp_lie_algebra_irrep
from core.numerical_utils import safe_omega_inv
from core.gaussian_core import push_gaussian
from transport.bundle_morphism_utils import gauge_covariant_transform_phi as _gct
# -------------------------------------------------------------------------
#                          PUBLIC ENTRYPOINTS
# -------------------------------------------------------------------------
def accumulate_mu_sigma_gradient(agent_i, agents, params, mode):
    """
    Accumulate natural gradient ∇μ and ∇Σ for belief or model.
    Prints a compact sign/scale line if DEBUG_MU_SIG is True.
    """
    import numpy as np
    import core.config as config

    # ---- helpers we rely on (import locally to avoid module-order issues)
    


    PRINT_SIGN = bool(params.get("DEBUG_MU_SIG", False))

    # pull ctx (may be None — wrappers handle it)
    ctx = params.get("runtime_ctx", None)

    eps         = float(getattr(config, "eps", 1e-8))
    support_eps = float(getattr(config, "support_cutoff_eps", 1e-4))

   
    # ---------- choose fields/weights per mode ----------
    if mode == "belief":
        alpha  = float(params.get("alpha",           getattr(config, "alpha",           1.0)))
        beta   = float(params.get("beta",            getattr(config, "beta",            0.0)))
        lambd  = float(params.get("feedback_weight", getattr(config, "feedback_weight", 0.0)))

        mu_q,  sigma_q  = agent_i.mu_q_field, agent_i.sigma_q_field
        mu_p,  sigma_p  = agent_i.mu_p_field, agent_i.sigma_p_field
        # Pull Φ, Φ̃
        if ctx is not None:
            Phi_mat       = Phi(ctx, agent_i, kind="q_to_p")
            Phi_tilde_mat = Phi(ctx, agent_i, kind="p_to_q")
        else:
            # Local fallback (no ctx): build from bases + φ without caching.
            try:
                
                phi       = getattr(agent_i, "phi", None)
                phi_model = getattr(agent_i, "phi_model", None)
                Gq = getattr(agent_i, "generators_q", None)
                Gp = getattr(agent_i, "generators_p", None)
                Phi_mat       = _gct(phi_model, phi, getattr(agent_i, "Phi_0", None),       G_q=Gq, G_p=Gp)
                Phi_tilde_mat = _gct(phi,       phi_model, getattr(agent_i, "Phi_tilde_0", None), G_q=Gp, G_p=Gq)
            except Exception:
                print("\n belief Phi/Phi~ FAIL\n")
                Phi_mat = Phi_tilde_mat = None

        # push p→q and q→p via bundle morphisms
        mu_p_push, sigma_p_push, sigma_p_push_inv = push_gaussian(
            mu=mu_p, Sigma=sigma_p, M=Phi_tilde_mat, eps=eps,
            symmetrize=True, sanitize=True,
            approx="auto" if getattr(config, "use_first_order_sigma_push", True) else "exact",
            near_identity_tau=float(getattr(config, "lambda_near_identity_tau", 5e-2)),
            return_inv=True,
        )
        mu_q_push, sigma_q_push, sigma_q_push_inv = push_gaussian(
            mu=mu_q, Sigma=sigma_q, M=Phi_mat, eps=eps,
            symmetrize=True, sanitize=True,
            approx="auto" if getattr(config, "use_first_order_sigma_push", True) else "exact",
            near_identity_tau=float(getattr(config, "lambda_near_identity_tau", 5e-2)),
            return_inv=True,
        )

        grad_self_mu, grad_self_sigma = grad_self_energy_belief(
            mu_q, sigma_q, mu_p_push, sigma_p_push_inv, eps=eps
        )
        grad_fb_mu, grad_fb_sigma = grad_feedback_energy_belief(
            mu_q, sigma_q, mu_p, mu_q_push, sigma_q_push, sigma_q_push_inv, sigma_p, Phi_mat, eps=eps
        )

        fiber      = "q"
        mu_key     = "mu_q_field"
        sigma_key  = "sigma_q_field"
        generators = getattr(agent_i, "generators_q", None)

    elif mode == "model":
        alpha  = float(params.get("alpha",           getattr(config, "alpha",           1.0)))
        beta   = float(params.get("beta_model",      getattr(config, "beta_model",      0.0)))
        lambd  = float(params.get("feedback_weight", getattr(config, "feedback_weight", 0.0)))

        mu_p,  sigma_p  = agent_i.mu_p_field, agent_i.sigma_p_field
        mu_q,  sigma_q  = agent_i.mu_q_field, agent_i.sigma_q_field
        if ctx is not None:
            Phi_mat       = Phi(ctx, agent_i, kind="q_to_p")
            Phi_tilde_mat = Phi(ctx, agent_i, kind="p_to_q")
        else:
            try:
                
                phi       = getattr(agent_i, "phi", None)
                phi_model = getattr(agent_i, "phi_model", None)
                Gq = getattr(agent_i, "generators_q", None)
                Gp = getattr(agent_i, "generators_p", None)
                Phi_mat       = _gct(phi_model, phi, getattr(agent_i, "Phi_0", None),       G_q=Gq, G_p=Gp)
                Phi_tilde_mat = _gct(phi,       phi_model, getattr(agent_i, "Phi_tilde_0", None), G_q=Gp, G_p=Gq)
            except Exception:
                print("\n model Phi/Phi~ FAIL\n")
                Phi_mat = Phi_tilde_mat = None

        mu_p_push, sigma_p_push, sigma_p_push_inv = push_gaussian(
            mu=mu_p, Sigma=sigma_p, M=Phi_tilde_mat, eps=eps,
            symmetrize=True, sanitize=True,
            approx="auto" if getattr(config, "use_first_order_sigma_push", True) else "exact",
            near_identity_tau=float(getattr(config, "lambda_near_identity_tau", 5e-2)),
            return_inv=True,
        )
        mu_q_push, sigma_q_push, sigma_q_push_inv = push_gaussian(
            mu=mu_q, Sigma=sigma_q, M=Phi_mat, eps=eps,
            symmetrize=True, sanitize=True,
            approx="auto" if getattr(config, "use_first_order_sigma_push", True) else "exact",
            near_identity_tau=float(getattr(config, "lambda_near_identity_tau", 5e-2)),
            return_inv=True,
        )

        grad_self_mu, grad_self_sigma = grad_self_energy_model(
            mu_q, sigma_q, mu_p_push, sigma_p_push_inv, Phi_tilde_mat, eps=eps
        )
        grad_fb_mu, grad_fb_sigma = grad_feedback_energy_model(
            mu_p, sigma_p, mu_q_push, sigma_q_push, sigma_q_push_inv, Phi_mat, eps=eps
        )

        fiber      = "p"
        mu_key     = "mu_p_field"
        sigma_key  = "sigma_p_field"
        generators = getattr(agent_i, "generators_p", None)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # Generators fallback (only for local Ω fallback)
    if generators is None:
        K = getattr(agent_i, sigma_key).shape[-1]
        generators, _ = get_generators(getattr(config, "group_name", "so3"), K, return_meta=True)

    mask_i = (np.asarray(agent_i.mask) > support_eps)

    # -------------------- neighbor alignment --------------------
    dmu_align    = np.zeros_like(getattr(agent_i, mu_key),    dtype=np.float32)
    dsigma_align = np.zeros_like(getattr(agent_i, sigma_key), dtype=np.float32)

    nb_list = getattr(agent_i, "neighbors", []) or []
    if nb_list:
        
        for nb in nb_list:
            j = nb["id"]
            agent_j = agents[j]
            mask_j = (np.asarray(agent_j.mask) > support_eps)
            overlap = mask_i & mask_j
            if not np.any(overlap):
                continue

            # Ω_ij from cache or local fallback
            if (ctx is not None) and (Omega is not None):
                Omega_ij = Omega(ctx, agent_i, agent_j, which=fiber)  # (...,K,K)
            else:
                if exp_lie_algebra_irrep is None:
                    continue  # no local fallback available
                phi_i = agent_i.phi if fiber == "q" else agent_i.phi_model
                phi_j = agent_j.phi  if fiber == "q" else agent_j.phi_model
                E_i   = exp_lie_algebra_irrep(np.asarray(phi_i, np.float32), generators)
                E_j   = exp_lie_algebra_irrep(np.asarray(phi_j, np.float32), generators)
                Omega_ij = np.matmul(E_i, safe_omega_inv(E_j, eps=eps))

            # Transport neighbor (μ_j, Σ_j) → i
            mu_j    = getattr(agent_j, mu_key)
            sigma_j = getattr(agent_j, sigma_key)
            mu_j_t, sigma_j_t, sigma_j_t_inv = push_gaussian(
                mu=mu_j, Sigma=sigma_j, M=Omega_ij, eps=eps,
                symmetrize=True, sanitize=True,
                approx="auto" if getattr(config, "use_first_order_sigma_push", True) else "exact",
                near_identity_tau=float(getattr(config, "lambda_near_identity_tau", 5e-2)),
                return_inv=True,
            )

            mu_i    = getattr(agent_i, mu_key)
            sigma_i = getattr(agent_i, sigma_key)

            dmu_i, dsigma_i = grad_alignment_energy_source(
                mu_i, sigma_i, mu_j_t, sigma_j_t_inv, eps=eps
            )

            # accumulate on overlap only
            dmu_align[overlap]    += dmu_i[overlap]
            dsigma_align[overlap] += dsigma_i[overlap]

    # -------------------- total Euclidean grads --------------------
    dmu_total    = alpha * grad_self_mu    + lambd * grad_fb_mu    + beta * dmu_align
    dsigma_total = alpha * grad_self_sigma + lambd * grad_fb_sigma + beta * dsigma_align

    # (belief) entropy regularizer on Σ
    if mode == "belief":
        dsigma_total = accumulate_entropy_sigma_gradient(agent_i, dsigma_total, params)

    # symmetrize + sanitize + numeric guards
    dmu_total    = np.nan_to_num(dmu_total,    nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    dsigma_total = 0.5 * (dsigma_total + np.swapaxes(dsigma_total, -1, -2))
    dsigma_total = np.nan_to_num(dsigma_total, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    # -------------------- project to natural gradient --------------------
    delta_mu, delta_sigma = apply_natural_gradient_batch(
        getattr(agent_i, mu_key),
        getattr(agent_i, sigma_key),
        dmu_total,
        dsigma_total,
        mask=agent_i.mask,
    )

    # -------------------- accumulate --------------------
    if mode == "belief":
        grad_mu_attr, grad_sigma_attr = "grad_mu_q", "grad_sigma_q"
    else:
        grad_mu_attr, grad_sigma_attr = "grad_mu_p", "grad_sigma_p"

    if getattr(agent_i, grad_mu_attr, None) is None:
        setattr(agent_i, grad_mu_attr, np.zeros_like(getattr(agent_i, mu_key), dtype=np.float32))
    if getattr(agent_i, grad_sigma_attr, None) is None:
        setattr(agent_i, grad_sigma_attr, np.zeros_like(getattr(agent_i, sigma_key), dtype=np.float32))

    setattr(agent_i, grad_mu_attr,    getattr(agent_i, grad_mu_attr)    + delta_mu)
    setattr(agent_i, grad_sigma_attr, getattr(agent_i, grad_sigma_attr) + delta_sigma)

    # -------------------- minimal validation print --------------------
    if PRINT_SIGN:
        aid = getattr(agent_i, "id", "?")
        # mean over spatial dims
        ax = tuple(range(delta_mu.ndim - 1))
        mu_mean = np.mean(delta_mu, axis=ax)
        # for Σ, summarize with Frobenius mean and signed trace components
        S = delta_sigma
        S_tr = np.mean(np.trace(S, axis1=-2, axis2=-1), axis=ax)  # scalar mean trace
        def sgns(vec): return "".join("+" if v>0 else "-" if v<0 else "0" for v in vec.tolist())
        print(f"[MU/SIG] mode={mode} ai={aid} "
              f"μ_sign={sgns(mu_mean)} μ_mean={np.array2string(mu_mean, precision=3, suppress_small=True)} "
              f"|μ|_mean={float(np.mean(np.abs(delta_mu))):.3e} "
              f"|Σ|_F_mean={float(np.mean(np.linalg.norm(S.reshape(*S.shape[:-2], -1), axis=-1))):.3e} "
              f"tr(ΣΔ)_mean={float(S_tr):.3e}",
              flush=True)


def accumulate_entropy_sigma_gradient(agent_i, grad_sigma_q, params):
    """
    Add entropy regularization to the Σ-gradient.

    Convention (default):
        E_total = E_other - gamma * H[q]
      ⇒ ∇_Σ E_total = ∇_Σ E_other - gamma * ∇_Σ H
      ⇒ with ∇_Σ H = ½ Σ^{-1}

    You can override the sign via params["entropy_in_energy_sign"] ∈ {+1, -1}.
    Use +1 if energy includes +gamma * H (penalty),
    use -1 if it includes -gamma * H (reward; default).
    """
    

    gamma = float(params.get("entropy_weight", getattr(config, "entropy_weight", 0.0)))
    if gamma == 0.0:
        return grad_sigma_q

    s = int(params.get("entropy_in_energy_sign", -1))
    sigma_q = sanitize_sigma(np.asarray(agent_i.sigma_q_field, dtype=np.float32), strict=True)
    grad_H  = 0.5 * safe_batch_inverse(sigma_q)

    mask = (np.asarray(agent_i.mask) > float(getattr(config, "support_cutoff_eps", 0.0)))[..., None, None]
    grad_H = grad_H * mask

    out = grad_sigma_q + gamma * s * grad_H
    out = 0.5 * (out + np.swapaxes(out, -1, -2))
    return out.astype(np.float32, copy=False)




#==============================================================================
#
#                    FEEDBACK / SELF TERMS (unchanged math, safer numerics)
#==============================================================================

def grad_alignment_energy_source(mu_i, sigma_i, mu_j_t, sigma_j_t_inv, eps=1e-8):
    delta_mu = mu_i - mu_j_t
    grad_mu = np.einsum("...ij,...j->...i", sigma_j_t_inv, delta_mu, optimize=True)

    # sanitize once per call
    sigma_i = sanitize_sigma(np.asarray(sigma_i, dtype=np.float32), strict=True)
    sigma_i_inv = safe_batch_inverse(sigma_i)

    grad_sigma = 0.5 * (sigma_j_t_inv - sigma_i_inv)
    grad_sigma = 0.5 * (grad_sigma + np.swapaxes(grad_sigma, -1, -2))
    return grad_mu.astype(np.float32, copy=False), grad_sigma.astype(np.float32, copy=False)


def grad_feedback_energy_belief(mu_q, sigma_q, mu_p, mu_q_push,
                                sigma_q_push, sigma_q_push_inv, sigma_p, Phi, eps=1e-8):
    delta_mu = mu_p - mu_q_push
    tmp = np.einsum("...ij,...j->...i", sigma_q_push_inv, delta_mu, optimize=True)
    grad_mu = -np.einsum("...ji,...j->...i", Phi, tmp, optimize=True)

    A = np.einsum("...i,...j->...ij", delta_mu, delta_mu, optimize=True)

    term1 = sigma_q_push_inv
    term2 = np.einsum("...ik,...kl,...lj->...ij", sigma_q_push_inv, A,       sigma_q_push_inv, optimize=True)
    term3 = np.einsum("...ik,...kl,...lj->...ij", sigma_q_push_inv, sigma_p, sigma_q_push_inv, optimize=True)
    M = term1 - term2 - term3

    grad_sigma = 0.5 * np.einsum("...ki,...kl,...lj->...ij", Phi, M, Phi, optimize=True)
    grad_sigma = 0.5 * (grad_sigma + np.swapaxes(grad_sigma, -1, -2))
    return grad_mu.astype(np.float32, copy=False), grad_sigma.astype(np.float32, copy=False)


def grad_feedback_energy_model(mu_p, sigma_p, mu_q_push,
                               sigma_q_push, sigma_q_push_inv, Phi=None, eps=1e-8):
    sigma_q_push = sanitize_sigma(np.asarray(sigma_q_push, dtype=np.float32), strict=True)
    if sigma_q_push_inv is None:
        sigma_q_push_inv = safe_batch_inverse(sigma_q_push)

    delta_mu = mu_p - mu_q_push
    grad_mu = np.einsum("...ij,...j->...i", sigma_q_push_inv, delta_mu, optimize=True)

    sigma_p = sanitize_sigma(np.asarray(sigma_p, dtype=np.float32), strict=True)
    sigma_p_inv = safe_batch_inverse(sigma_p)

    grad_sigma = 0.5 * (sigma_q_push_inv - sigma_p_inv)
    grad_sigma = 0.5 * (grad_sigma + np.swapaxes(grad_sigma, -1, -2))
    return grad_mu.astype(np.float32, copy=False), grad_sigma.astype(np.float32, copy=False)


def grad_self_energy_belief(mu_q, sigma_q, mu_p_push, sigma_p_push_inv, eps=1e-8, mask=None):
    delta_mu = mu_q - mu_p_push
    grad_mu = np.einsum("...ij,...j->...i", sigma_p_push_inv, delta_mu, optimize=True)

    sigma_q = sanitize_sigma(np.asarray(sigma_q, dtype=np.float32), strict=True)
    sigma_q_inv = safe_batch_inverse(sigma_q)

    grad_sigma = 0.5 * (sigma_p_push_inv - sigma_q_inv)
    grad_sigma = 0.5 * (grad_sigma + np.swapaxes(grad_sigma, -1, -2))

    if mask is not None:
        m = (np.asarray(mask) > float(getattr(config, "support_cutoff_eps", 0.0)))
        grad_mu    = grad_mu * m[..., None]
        grad_sigma = grad_sigma * m[..., None, None]

    return grad_mu.astype(np.float32, copy=False), grad_sigma.astype(np.float32, copy=False)


def grad_self_energy_model(mu_q, sigma_q, mu_p_push, sigma_p_push_inv, Phi_tilde, eps=1e-8, mask=None):
    """
    ∂ wrt (μ_p, Σ_p) of KL(q || Φ̃·p), with A = Φ̃ Σ_p Φ̃ᵀ and Δ = μ_q − μ_p^t.
    """
    delta_mu = mu_q - mu_p_push
    tmp = np.einsum("...ij,...j->...i", sigma_p_push_inv, delta_mu, optimize=True)
    grad_mu = -np.einsum("...ji,...j->...i", Phi_tilde, tmp, optimize=True)

    delta_mu_outer = np.einsum("...i,...j->...ij", delta_mu, delta_mu, optimize=True)

    M = sigma_p_push_inv
    G = M - np.einsum("...ik,...kl,...lj->...ij", M, delta_mu_outer, M, optimize=True) \
          - np.einsum("...ik,...kl,...lj->...ij", M, sigma_q,        M, optimize=True)

    Phi_tilde_T = np.swapaxes(Phi_tilde, -1, -2)
    grad_sigma = 0.5 * np.einsum("...ki,...ij,...jl->...kl", Phi_tilde_T, G, Phi_tilde, optimize=True)
    grad_sigma = 0.5 * (grad_sigma + np.swapaxes(grad_sigma, -1, -2))  # symmetrize

    if mask is not None:
        m = (np.asarray(mask) > float(getattr(config, "support_cutoff_eps", 0.0)))
        grad_mu    = grad_mu * m[..., None]
        grad_sigma = grad_sigma * m[..., None, None]

    return grad_mu.astype(np.float32, copy=False), grad_sigma.astype(np.float32, copy=False)








