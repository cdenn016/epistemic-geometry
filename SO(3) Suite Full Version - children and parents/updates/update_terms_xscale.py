# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 15:57:22 2025

@author: chris and christine
"""
import numpy as np
import core.config as config
from core.numerical_utils import sanitize_sigma, safe_batch_inverse, safe_inv
from transport.transport_cache import Lambda_dn, Lambda_up, Theta_dn, Theta_up, E_grid

from updates.update_terms_mu_sigma import (
    grad_alignment_energy_source,
    grad_self_energy_belief,
    grad_feedback_energy_model,
)
from updates.update_terms_phi import grad_kl_wrt_phi_i  # OK: lvl-0 does not import this module
from core.omega import (compute_curvature_gradient_analytical)

from core.omega import d_exp_phi_exact, d_exp_phi_tilde_exact
# -----------------------------------------------------------------------------
# μ/Σ adapter: accumulate grads against explicit Gaussian targets
# -----------------------------------------------------------------------------
from core.gaussian_core import push_gaussian

def compute_crossscale_mu_sigma_env(agent_i, *, ctx, params, mode: str):
    """
    Cross-scale μ/Σ contributions for agent_i, using your validated gradient kernels.

    mode ∈ {"belief","model"} selects which fiber of agent_i we’re updating.
    We support two mapping choices per branch:
      - same-fiber via Λ (default): alignment energy wrt mapped same-fiber targets
      - cross-fiber via Θ (optional): q-branch uses grad_self_energy_belief; p-branch uses grad_feedback_energy_model

    Config/param knobs (bool):
      env_down_q_use_theta, env_down_p_use_theta, env_up_q_use_theta, env_up_p_use_theta
    Weights (floats):
      lambda_env_down_q/_p, lambda_env_up_q/_p
    """
    

    assert mode in ("belief","model")

    # --- weights: Λ (same-fiber) and Θ (cross-fiber) kept separate ----------
    lam_down_q       = float(params.get("lambda_env_down_q",        getattr(config, "lambda_env_down_q",        0.0)))
    lam_down_p       = float(params.get("lambda_env_down_p",        getattr(config, "lambda_env_down_p",        0.0)))
    lam_up_q         = float(params.get("lambda_env_up_q",          getattr(config, "lambda_env_up_q",          0.0)))
    lam_up_p         = float(params.get("lambda_env_up_p",          getattr(config, "lambda_env_up_p",          0.0)))

    lam_down_theta_q = float(params.get("lambda_env_down_theta_q",  getattr(config, "lambda_env_down_theta_q",  0.0)))
    lam_down_theta_p = float(params.get("lambda_env_down_theta_p",  getattr(config, "lambda_env_down_theta_p",  0.0)))
    lam_up_theta_q   = float(params.get("lambda_env_up_theta_q",    getattr(config, "lambda_env_up_theta_q",    0.0)))
    lam_up_theta_p   = float(params.get("lambda_env_up_theta_p",    getattr(config, "lambda_env_up_theta_p",    0.0)))


    support_eps = float(getattr(config, "support_cutoff_eps", getattr(config, "support_tau", 1e-3)))
    eps = float(params.get("eps", getattr(config, "eps", 1e-8)))

    # --- fields & mask -------------------------------------------------------
    mu_q_i = getattr(agent_i, "mu_q_field", None)
    Si_q_i = getattr(agent_i, "sigma_q_field", None)
    mu_p_i = getattr(agent_i, "mu_p_field", None)
    Si_p_i = getattr(agent_i, "sigma_p_field", None)
    assert mu_q_i is not None and Si_q_i is not None and mu_p_i is not None and Si_p_i is not None



    m = np.asarray(getattr(agent_i, "mask", 0.0), np.float32)
    m_bool = (m > support_eps)

    # local accumulators (same shapes as μ/Σ being updated by this call)
    dmu_env    = np.zeros_like(mu_q_i if mode == "belief" else mu_p_i, dtype=np.float32)
    dsigma_env = np.zeros_like(Si_q_i if mode == "belief" else Si_p_i, dtype=np.float32)

    # --- helpers to accumulate one pair --------------------------------------
    def _acc_q_alignment(mu_q_t, Si_q_t):
        # same-fiber alignment: ∂ wrt (μ_q_i, Σ_q_i) of KL(N_i || N_t)
        Si_q_t = sanitize_sigma(np.asarray(Si_q_t, np.float32), strict=True)
        Si_q_t_inv = safe_batch_inverse(Si_q_t)
        gmu, gSig = grad_alignment_energy_source(mu_q_i, Si_q_i, mu_q_t, Si_q_t_inv, eps=eps)
        return gmu, gSig

    def _acc_p_alignment(mu_p_t, Si_p_t):
        Si_p_t = sanitize_sigma(np.asarray(Si_p_t, np.float32), strict=True)
        Si_p_t_inv = safe_batch_inverse(Si_p_t)
        gmu, gSig = grad_alignment_energy_source(mu_p_i, Si_p_i, mu_p_t, Si_p_t_inv, eps=eps)
        return gmu, gSig

    def _acc_q_cross(mu_p_push, Si_p_push):
        # cross-fiber: ∂ wrt (μ_q_i, Σ_q_i) of KL(q_i || p_push) using your kernel
        Si_p_push = sanitize_sigma(np.asarray(Si_p_push, np.float32), strict=True)
        Si_p_push_inv = safe_batch_inverse(Si_p_push)
        gmu, gSig = grad_self_energy_belief(mu_q_i, Si_q_i, mu_p_push, Si_p_push_inv, eps=eps, mask=None)
        return gmu, gSig

    def _acc_p_cross(mu_q_push, Si_q_push):
        # cross-fiber: ∂ wrt (μ_p_i, Σ_p_i) of KL(p_i || q_push) using your kernel
        Si_q_push = sanitize_sigma(np.asarray(Si_q_push, np.float32), strict=True)
        Si_q_push_inv = safe_batch_inverse(Si_q_push)
        gmu, gSig = grad_feedback_energy_model(mu_p_i, Si_p_i, mu_q_push, Si_q_push, Si_q_push_inv, Phi=None, eps=eps)
        return gmu, gSig

    # --- who are my parents/children? ---------------------------------------
    parents = []
    children = []
    if ctx is not None:
        aid = int(getattr(agent_i, "id", id(agent_i)))
        parents  = list((getattr(ctx, "parents_for_agent", {}) or {}).get(aid, []))
        children = list((getattr(ctx, "children_for_agent", {}) or {}).get(aid, []))
    else:
        parents  = list(getattr(agent_i, "parents",  [])) if hasattr(agent_i, "parents")  else []
        children = list(getattr(agent_i, "children", [])) if hasattr(agent_i, "children") else []

    # ========================= DOWNWARD: parents → agent_i ====================
    for P in parents:
        # q-branch target (Λ or Θ)
        if mode == "belief":
            # Λ↓_q : parent q → child q  (alignment)
            if lam_down_q != 0.0:
                Lq = Lambda_dn(ctx, agent_i, P, which="q")
                mu_q_par = getattr(P, "mu_q_field"); Si_q_par = getattr(P, "sigma_q_field")
                mu_q_t, Si_q_t = push_gaussian(
                    mu=mu_q_par, Sigma=Si_q_par, M=Lq,
                    eps=float(getattr(config, "eps", 1e-8)),
                    symmetrize=True, sanitize=True, approx="exact"
                )
                gmu, gSig = _acc_q_alignment(mu_q_t, Si_q_t)
                dmu_env    += lam_down_q * gmu
                dsigma_env += lam_down_q * gSig
                
              #  print(f"[Λ↓_q] child {agent_i.id} <- parent {P.id} "
              #        f"λ={lam_down_q:.3f} "
              #        f"‖grad_mu‖={np.linalg.norm(gmu):.3e} "
              #        f"‖grad_sigma‖={np.linalg.norm(gSig):.3e}")

            # Θ↓.q : parent p → child q  (self-belief)
            if lam_down_theta_q != 0.0:
                Th_q = Theta_dn(ctx, agent_i, P)["q"]
                mu_p_par = getattr(P, "mu_p_field"); Si_p_par = getattr(P, "sigma_p_field")
                mu_q_push, Si_q_push = push_gaussian(
                    mu=mu_p_par, Sigma=Si_p_par, M=Th_q,
                    eps=float(getattr(config, "eps", 1e-8)),
                    symmetrize=True, sanitize=True, approx="exact"
                )
                gmu, gSig = _acc_q_cross(mu_q_push, Si_q_push)
                dmu_env    += lam_down_theta_q * gmu
                dsigma_env += lam_down_theta_q * gSig
                #print(f"[Θ↓_q] child {agent_i.id} <- parent {P.id} "
                #      f"λ={lam_down_theta_q:.3f} "
                #      f"‖grad_mu‖={np.linalg.norm(gmu):.3e} "
                #      f"‖grad_sigma‖={np.linalg.norm(gSig):.3e}")


        # p-branch target (Λ or Θ)
        else:  # mode == "model"
            # Λ↓_p : parent p → child p  (alignment)
            if lam_down_p != 0.0:
                Lp = Lambda_dn(ctx, agent_i, P, which="p")
                mu_p_par = getattr(P, "mu_p_field"); Si_p_par = getattr(P, "sigma_p_field")
                mu_p_t, Si_p_t = push_gaussian(
                    mu=mu_p_par, Sigma=Si_p_par, M=Lp,
                    eps=float(getattr(config, "eps", 1e-8)),
                    symmetrize=True, sanitize=True, approx="exact"
                )
                gmu, gSig = _acc_p_alignment(mu_p_t, Si_p_t)
                dmu_env    += lam_down_p * gmu
                dsigma_env += lam_down_p * gSig
            # Θ↓.p : parent q → child p  (feedback-model)
            if lam_down_theta_p != 0.0:
                Th_p = Theta_dn(ctx, agent_i, P)["p"]
                mu_q_par = getattr(P, "mu_q_field"); Si_q_par = getattr(P, "sigma_q_field")
                
                mu_p_push, Si_p_push = push_gaussian(
                    mu=mu_q_par, Sigma=Si_q_par, M=Th_p,
                    eps=float(getattr(config, "eps", 1e-8)),
                    symmetrize=True, sanitize=True, approx="exact"
                )
                gmu, gSig = _acc_p_cross(mu_p_push, Si_p_push)
                dmu_env    += lam_down_theta_p * gmu
                dsigma_env += lam_down_theta_p * gSig


    # ========================== UPWARD: children → agent_i ====================
    for C in children:
        if mode == "belief":
            # Λ↑_q : child q → parent q  (alignment)
            if lam_up_q != 0.0:
                Lq = Lambda_up(ctx, C, agent_i, which="q")
                mu_q_ch = getattr(C, "mu_q_field"); Si_q_ch = getattr(C, "sigma_q_field")
                mu_q_t, Si_q_t = push_gaussian(
                    mu=mu_q_ch, Sigma=Si_q_ch, M=Lq,
                    eps=float(getattr(config, "eps", 1e-8)),
                    symmetrize=True, sanitize=True, approx="exact"
                )
                gmu, gSig = _acc_q_alignment(mu_q_t, Si_q_t)
                dmu_env    += lam_up_q * gmu
                dsigma_env += lam_up_q * gSig
            # Θ↑.q : child p → parent q  (self-belief)
            if lam_up_theta_q != 0.0:
                Th_q = Theta_up(ctx, C, agent_i)["q"]
                mu_p_ch = getattr(C, "mu_p_field"); Si_p_ch = getattr(C, "sigma_p_field")
                mu_q_push, Si_q_push = push_gaussian(
                    mu=mu_p_ch, Sigma=Si_p_ch, M=Th_q,
                    eps=float(getattr(config, "eps", 1e-8)),
                    symmetrize=True, sanitize=True, approx="exact"
                )
                gmu, gSig = _acc_q_cross(mu_q_push, Si_q_push)
                dmu_env    += lam_up_theta_q * gmu
                dsigma_env += lam_up_theta_q * gSig

        else:
            # Λ↑_p : child p → parent p  (alignment)
            if lam_up_p != 0.0:
                Lp = Lambda_up(ctx, C, agent_i, which="p")
                mu_p_ch = getattr(C, "mu_p_field"); Si_p_ch = getattr(C, "sigma_p_field")
                mu_p_t, Si_p_t = push_gaussian(
                    mu=mu_p_ch, Sigma=Si_p_ch, M=Lp,
                    eps=float(getattr(config, "eps", 1e-8)),
                    symmetrize=True, sanitize=True, approx="exact"
                )
                gmu, gSig = _acc_p_alignment(mu_p_t, Si_p_t)
                dmu_env    += lam_up_p * gmu
                dsigma_env += lam_up_p * gSig
            # Θ↑.p : child q → parent p  (feedback-model)
            if lam_up_theta_p != 0.0:
                Th_p = Theta_up(ctx, C, agent_i)["p"]
                mu_q_ch = getattr(C, "mu_q_field"); Si_q_ch = getattr(C, "sigma_q_field")
                mu_p_push, Si_p_push = push_gaussian(
                    mu=mu_q_ch, Sigma=Si_q_ch, M=Th_p,
                    eps=float(getattr(config, "eps", 1e-8)),
                    symmetrize=True, sanitize=True, approx="exact"
                )
                gmu, gSig = _acc_p_cross(mu_p_push, Si_p_push)
                dmu_env    += lam_up_theta_p * gmu
                dsigma_env += lam_up_theta_p * gSig


    # --- mask to agent_i support (same rule as your other terms) -------------
    if mode == "belief":
        dmu_env    = np.where(m_bool[..., None],     dmu_env,    0.0)
        dsigma_env = np.where(m_bool[..., None, None], dsigma_env, 0.0)
    else:
        dmu_env    = np.where(m_bool[..., None],     dmu_env,    0.0)
        dsigma_env = np.where(m_bool[..., None, None], dsigma_env, 0.0)

    return dmu_env.astype(np.float32, copy=False), dsigma_env.astype(np.float32, copy=False)

# ---------------------------------------------------------------------------
# φ / φ̃ cross-scale "environment" terms (Λ and Θ, up & down)
# ---------------------------------------------------------------------------
def accumulate_crossscale_phi_env(
    agent_i,
    *,
    generators_q,
    generators_p,
    params,
):
    """
    Add φ/φ̃ env grads for cross-scale interactions:
      - belief (φ):    use Λ on q (same-fiber) and Θ.q (p→q) from parents/children
      - model  (φ̃):   use Λ on p (same-fiber) and Θ.p (q→p) from parents/children

    Each path has an independent weight; all default to 0.0 (no behavior change).
    """
    import core.config as config
    

    ctx = params.get("runtime_ctx", None)

    # --- weights (separate Λ vs Θ, down vs up) -------------------------------
    # φ (belief; q-fiber)
    lam_dn_q      = float(params.get("lambda_phi_env_down_q",       getattr(config, "lambda_phi_env_down_q",       0.0)))
    lam_up_q      = float(params.get("lambda_phi_env_up_q",         getattr(config, "lambda_phi_env_up_q",         0.0)))
    lam_dn_theta_q= float(params.get("lambda_phi_env_down_theta_q", getattr(config, "lambda_phi_env_down_theta_q", 0.0)))
    lam_up_theta_q= float(params.get("lambda_phi_env_up_theta_q",   getattr(config, "lambda_phi_env_up_theta_q",   0.0)))
    # φ̃ (model; p-fiber)
    lam_dn_p      = float(params.get("lambda_phi_env_down_p",       getattr(config, "lambda_phi_env_down_p",       0.0)))
    lam_up_p      = float(params.get("lambda_phi_env_up_p",         getattr(config, "lambda_phi_env_up_p",         0.0)))
    lam_dn_theta_p= float(params.get("lambda_phi_env_down_theta_p", getattr(config, "lambda_phi_env_down_theta_p", 0.0)))
    lam_up_theta_p= float(params.get("lambda_phi_env_up_theta_p",   getattr(config, "lambda_phi_env_up_theta_p",   0.0)))

    # quick exit if all zero
    if max(lam_dn_q, lam_up_q, lam_dn_theta_q, lam_up_theta_q,
           lam_dn_p, lam_up_p, lam_dn_theta_p, lam_up_theta_p) == 0.0:
        return

    # --- basic fields & masks ------------------------------------------------
    eps_tau = float(getattr(config, "support_cutoff_eps", 1e-6))
    mask_i  = (np.asarray(agent_i.mask) > eps_tau)[..., None]

    mu_q_i, Si_q_i = agent_i.mu_q_field, agent_i.sigma_q_field
    mu_p_i, Si_p_i = agent_i.mu_p_field, agent_i.sigma_p_field

    # exp grids (for Ω = E_i E_j^{-1}); for cross-scale we set E_j = I so Ω=E_i
    E_q_i = E_grid(ctx, agent_i, which="q") if ctx is not None else None
    E_p_i = E_grid(ctx, agent_i, which="p") if ctx is not None else None

    # Jacobians
    Q_all_q = d_exp_phi_exact(np.asarray(agent_i.phi,       np.float32), generators_q)
    Q_all_p = d_exp_phi_tilde_exact(np.asarray(agent_i.phi_model, np.float32), generators_p)

    def _sigma_inv(S):
        S = sanitize_sigma(np.asarray(S, np.float32), eps=float(getattr(config, "eps", 1e-6)))
        return safe_inv(S, eps=float(getattr(config, "eps", 1e-6)))

    # Ω, E_j, exp(-φ_j) choices for cross-scale:
    # we already feed μ_t and Σ_t^{-1}, so we can take:
    #   Ω_ij = E_i,  E_j = I,  exp(-φ_j) = I
    def _phi_env_grad_q(mu_t, Si_t_inv):
        Omega_ij   = E_q_i if E_q_i is not None else np.eye(Si_q_i.shape[-1], dtype=np.float32)
        exp_phi_i  = Omega_ij
        exp_phi_j  = np.eye(Si_q_i.shape[-1], dtype=np.float32)
        exp_neg_j  = exp_phi_j  # I
        return grad_kl_wrt_phi_i(
            mu_i=mu_q_i, sigma_i=Si_q_i,
            mu_j=None, sigma_j=None,
            phi_i=np.asarray(agent_i.phi, np.float32), phi_j=None,
            Omega_ij=Omega_ij,
            generators=generators_q,
            exp_phi_i=exp_phi_i, exp_phi_j=exp_phi_j,
            mu_j_t=mu_t, sigma_j_t_inv=Si_t_inv,
            eps=float(getattr(config, "eps", 1e-6)),
            exp_neg_phi_j=exp_neg_j,
            Q_all=Q_all_q,
        )

    def _phi_env_grad_p(mu_t, Si_t_inv):
        Omega_ij   = E_p_i if E_p_i is not None else np.eye(Si_p_i.shape[-1], dtype=np.float32)
        exp_phi_i  = Omega_ij
        exp_phi_j  = np.eye(Si_p_i.shape[-1], dtype=np.float32)
        exp_neg_j  = exp_phi_j
        return grad_kl_wrt_phi_i(
            mu_i=mu_p_i, sigma_i=Si_p_i,
            mu_j=None, sigma_j=None,
            phi_i=np.asarray(agent_i.phi_model, np.float32), phi_j=None,
            Omega_ij=Omega_ij,
            generators=generators_p,
            exp_phi_i=exp_phi_i, exp_phi_j=exp_phi_j,
            mu_j_t=mu_t, sigma_j_t_inv=Si_t_inv,
            eps=float(getattr(config, "eps", 1e-6)),
            exp_neg_phi_j=exp_neg_j,
            Q_all=Q_all_p,
        )

    # ----- collect hierarchy from ctx or agent (same approach as μ/Σ) --------
    parents  = []
    children = []
    if ctx is not None:
        aid = int(getattr(agent_i, "id", -1))
        parents  = list((getattr(ctx, "parents_for_agent", {}) or {}).get(aid, []))
        children = list((getattr(ctx, "children_for_agent", {}) or {}).get(aid, []))
    else:
        parents  = list(getattr(agent_i, "parents",  [])) if hasattr(agent_i, "parents")  else []
        children = list(getattr(agent_i, "children", [])) if hasattr(agent_i, "children") else []

    # ========================= DOWNWARD: parents → i ==========================
    for P in parents:
        # Λ↓ on q (parent q → child q) → φ env on q
        if lam_dn_q != 0.0:
            Lq = Lambda_dn(ctx, agent_i, P, which="q")
            mu_q_t, Si_q_t = push_gaussian(
                mu=getattr(P, "mu_q_field"),
                Sigma=getattr(P, "sigma_q_field"),
                M=Lq,
                eps=float(getattr(config, "eps", 1e-8)),
                symmetrize=True, sanitize=True, approx="exact"
            )
            g = _phi_env_grad_q(mu_q_t, _sigma_inv(Si_q_t))
            agent_i.grad_phi = (getattr(agent_i, "grad_phi", 0.0) + lam_dn_q * g * mask_i).astype(np.float32, copy=False)
            # --- debug print
            
         #   print(f"[Λ↓_q φ] child {getattr(agent_i,'id',-1)} <- parent {getattr(P,'id',-1)} "
          #        f"λ={lam_dn_q:.3f} ‖grad‖={np.linalg.norm(g):.3e}\n")

        # Θ↓.q (parent p → child q) → φ env on q
        if lam_dn_theta_q != 0.0:
            Th_q = Theta_dn(ctx, agent_i, P)["q"]
            mu_qp, Si_qp = push_gaussian(
                mu=getattr(P, "mu_p_field"),
                Sigma=getattr(P, "sigma_p_field"),
                M=Th_q,
                eps=float(getattr(config, "eps", 1e-8)),
                symmetrize=True, sanitize=True, approx="exact"
            )
            g = _phi_env_grad_q(mu_qp, _sigma_inv(Si_qp))
            agent_i.grad_phi = (getattr(agent_i, "grad_phi", 0.0) + lam_dn_theta_q * g * mask_i).astype(np.float32, copy=False)
            # --- debug print
            
           # print(f"[Θ↓_q φ] child {getattr(agent_i,'id',-1)} <- parent {getattr(P,'id',-1)} "
           #       f"λ={lam_dn_theta_q:.3f} ‖grad‖={np.linalg.norm(g):.3e}\n")

        # Λ↓ on p (parent p → child p) → φ̃ env
        if lam_dn_p != 0.0:
            Lp = Lambda_dn(ctx, agent_i, P, which="p")
            mu_p_t, Si_p_t = push_gaussian(
                mu=getattr(P, "mu_p_field"),
                Sigma=getattr(P, "sigma_p_field"),
                M=Lp,
                eps=float(getattr(config, "eps", 1e-8)),
                symmetrize=True, sanitize=True, approx="exact"
            )
            g = _phi_env_grad_p(mu_p_t, _sigma_inv(Si_p_t))
            agent_i.grad_phi_tilde = (getattr(agent_i, "grad_phi_tilde", 0.0) + lam_dn_p * g * mask_i).astype(np.float32, copy=False)
            # --- debug print
            
         #   print(f"[Λ↓_p φ̃] child {getattr(agent_i,'id',-1)} <- parent {getattr(P,'id',-1)} "
         #         f"λ={lam_dn_p:.3f} ‖grad‖={np.linalg.norm(g):.3e}\n")

        # Θ↓.p (parent q → child p) → φ̃ env
        if lam_dn_theta_p != 0.0:
            Th_p = Theta_dn(ctx, agent_i, P)["p"]
            mu_pp, Si_pp = push_gaussian(
                mu=getattr(P, "mu_q_field"),
                Sigma=getattr(P, "sigma_q_field"),
                M=Th_p,
                eps=float(getattr(config, "eps", 1e-8)),
                symmetrize=True, sanitize=True, approx="exact"
            )
            g = _phi_env_grad_p(mu_pp, _sigma_inv(Si_pp))
            agent_i.grad_phi_tilde = (getattr(agent_i, "grad_phi_tilde", 0.0) + lam_dn_theta_p * g * mask_i).astype(np.float32, copy=False)
            # --- debug print
           
         #   print(f"[Θ↓_p φ̃] child {getattr(agent_i,'id',-1)} <- parent {getattr(P,'id',-1)} "
         #         f"λ={lam_dn_theta_p:.3f} ‖grad‖={np.linalg.norm(g):.3e}\n\n\n")

    # ========================== UPWARD: children → i ==========================
    for C in children:
        # Λ↑ on q (child q → parent q) → φ env on q
        if lam_up_q != 0.0:
            Lq = Lambda_up(ctx, C, agent_i, which="q")
            mu_q_t, Si_q_t = push_gaussian(
                mu=getattr(C, "mu_q_field"),
                Sigma=getattr(C, "sigma_q_field"),
                M=Lq,
                eps=float(getattr(config, "eps", 1e-8)),
                symmetrize=True, sanitize=True, approx="exact"
            )
            g = _phi_env_grad_q(mu_q_t, _sigma_inv(Si_q_t))
            agent_i.grad_phi = (getattr(agent_i, "grad_phi", 0.0) + lam_up_q * g * mask_i).astype(np.float32, copy=False)

        # Θ↑.q (child p → parent q) → φ env on q
        if lam_up_theta_q != 0.0:
            Th_q = Theta_up(ctx, C, agent_i)["q"]
            mu_qp, Si_qp = push_gaussian(
                mu=getattr(C, "mu_p_field"),
                Sigma=getattr(C, "sigma_p_field"),
                M=Th_q,
                eps=float(getattr(config, "eps", 1e-8)),
                symmetrize=True, sanitize=True, approx="exact"
            )
            g = _phi_env_grad_q(mu_qp, _sigma_inv(Si_qp))
            agent_i.grad_phi = (getattr(agent_i, "grad_phi", 0.0) + lam_up_theta_q * g * mask_i).astype(np.float32, copy=False)

        # Λ↑ on p (child p → parent p) → φ̃ env
        if lam_up_p != 0.0:
            Lp = Lambda_up(ctx, C, agent_i, which="p")
            mu_p_t, Si_p_t = push_gaussian(
                mu=getattr(C, "mu_p_field"),
                Sigma=getattr(C, "sigma_p_field"),
                M=Lp,
                eps=float(getattr(config, "eps", 1e-8)),
                symmetrize=True, sanitize=True, approx="exact"
            )
            g = _phi_env_grad_p(mu_p_t, _sigma_inv(Si_p_t))
            agent_i.grad_phi_tilde = (getattr(agent_i, "grad_phi_tilde", 0.0) + lam_up_p * g * mask_i).astype(np.float32, copy=False)

        # Θ↑.p (child q → parent p) → φ̃ env
        if lam_up_theta_p != 0.0:
            Th_p = Theta_up(ctx, C, agent_i)["p"]
            mu_pp, Si_pp = push_gaussian(
                mu=getattr(C, "mu_q_field"),
                Sigma=getattr(C, "sigma_q_field"),
                M=Th_p,
                eps=float(getattr(config, "eps", 1e-8)),
                symmetrize=True, sanitize=True, approx="exact"
            )
            g = _phi_env_grad_p(mu_pp, _sigma_inv(Si_pp))
            agent_i.grad_phi_tilde = (getattr(agent_i, "grad_phi_tilde", 0.0) + lam_up_theta_p * g * mask_i).astype(np.float32, copy=False)


