# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 16:05:57 2025

@author: chris and christine
"""

import numpy as np

from core.natural_gradient import inverse_fisher_metric_field as _invF_cached
from core.get_generators import get_generators            
from core.numerical_utils import sanitize_sigma, safe_inv, masked_cond_proxy    
from typing import Optional
from core.omega import retract_phi_principal


from transport.bundle_morphism_utils import initialize_intertwiners_for_agent, build_intertwiner_bases



def _rect_eye(Kdest: int, Ksrc: int, dtype=np.float32):
    M = np.zeros((Kdest, Ksrc), dtype=dtype)
    m = min(Kdest, Ksrc)
    M[np.arange(m), np.arange(m)] = 1.0
    return M

def _as_hw_block(M, H, W):
    """Broadcast (Kd,Ks) or (H,W,Kd,Ks) to (H,W,Kd,Ks)."""
    M = np.asarray(M)
    if M.ndim == 4 and M.shape[0] == H and M.shape[1] == W:
        return M
    if M.ndim == 2:
        return np.broadcast_to(M, (H, W, M.shape[0], M.shape[1])).astype(np.float32, copy=False)
    raise ValueError(f"Expected (Kd,Ks) or (H,W,Kd,Ks); got {M.shape}")

from transport.bundle_morphism_utils import (
    build_intertwiner_bases,
    _rescale_pair_fro,
    _choose_tilde_from_base_safe,  # if you need it elsewhere
)

def ensure_transforms(
    targets,
    generators_q=None,
    generators_p=None,
    *,
    ctx=None,
    intertwiner_method="casimir",    # "casimir" | "auto" | "nullspace"
    data_for_alignment=None,         # optional {'mu_q':..., 'mu_p':...}
    fro_target: float | None = None, # NEW: if set, rescale Φ0 and counter-scale Φ̃0
):
    import numpy as np

    if not targets:
        return

    # cache API
    set_bases = None
    try:
        from transport_cache import set_morphism_bases as _set_bases
        set_bases = _set_bases
    except Exception:
        pass

    for a in targets:
        if a is None:
            continue

        Kq = int(np.asarray(a.mu_q_field).shape[-1])
        Kp = int(np.asarray(a.mu_p_field).shape[-1])

        # --- build geometric bases ---
        Phi0, Phit0, meta = build_intertwiner_bases(
            np.asarray(generators_q, np.float32),
            np.asarray(generators_p, np.float32),
            method=str(intertwiner_method),
            data=data_for_alignment,
            return_meta=True,
        )

        # --- optional coherent Frobenius rescale (b) ---
        Phi0, Phit0, c = _rescale_pair_fro(Phi0, Phit0, fro_target)

        # --- persist on agent & cache (frozen) ---
        a.Phi_0 = np.asarray(Phi0, np.float32)
        a.Phi_tilde_0 = np.asarray(Phit0, np.float32)
        if set_bases is not None and ctx is not None:
            try:
                set_bases(ctx, a, a.Phi_0, a.Phi_tilde_0, allow_overwrite=False)
            except TypeError:
                set_bases(ctx, a, a.Phi_0, a.Phi_tilde_0)

        # --- improved logging (c): provenance + invariants ---
        # infer path if builder didn't set it explicitly
        path = (meta or {}).get("path")
        if path is None:
            if (meta or {}).get("reason") == "no_l_overlap":
                path = "zero"
            elif "nullspace_dim" in (meta or {}):
                path = "nullspace"
            else:
                path = intertwiner_method

        # invariants
        fro0 = float(np.linalg.norm(a.Phi_0, ord="fro"))
        frot = float(np.linalg.norm(a.Phi_tilde_0, ord="fro"))
        rank = int(np.linalg.matrix_rank(a.Phi_0))
        try:
            s0 = np.linalg.svd(a.Phi_0, compute_uv=False)
            smin = float(s0.min()) if s0.size else 0.0
            smax = float(s0.max()) if s0.size else 0.0
        except Exception:
            smin = smax = float("nan")

        # commutator residual (optional, cheap & very useful)
        def _comm_resid_norm(Gq, Gp, Phi):
            r2 = 0.0
            for aa in range(3):
                R = Gp[aa] @ Phi - Phi @ Gq[aa]
                r2 += float(np.linalg.norm(R, ord="fro")**2)
            return r2 ** 0.5

        resid = _comm_resid_norm(
            np.asarray(generators_q, np.float32),
            np.asarray(generators_p, np.float32),
            a.Phi_0.astype(np.float64)
        )

     #   print(
      #      "[bases]"
      #      f" aid={getattr(a,'id','?')}"
      #      f" path={path}"
      #      f" shape={a.Phi_0.shape}/{a.Phi_tilde_0.shape}"
      #      f" rank={rank}"
      #      f" ||Phi0||_F={fro0:.6f}"
      #      f" ||Phit0||_F={frot:.6f}"
      #      f" fro_prod={fro0*frot:.6f}"
      #      f" smin={smin:.6e} smax={smax:.6e}"
      #      f" resid={resid:.3e}"
       #     f" c={c:.6f}"
       #     f" meta_keys={sorted((meta or {}).keys())}"
       # )









# ---------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------

def preprocess_all_agents(agents, params=None, G_q=None, G_p=None, *, ctx: Optional[object] = None):
    """
    Build/refresh: generators, Σ sanitize + inverses, condition proxies,
    (optionally) Fisher. Also pre-warms exp grids via transport cache.
    Safe to call repeatedly; idempotent per step.
    """
    params = params or {}
    if ctx is not None:
        params = dict(params)
        params["runtime_ctx"] = ctx

    agents = [a for a in (agents or []) if a is not None]
    if not agents:
        return

    # preprocess structural fields per agent
    for a in agents:
        preprocess_agent_fields(a, params, G_q=G_q, G_p=G_p, ctx=ctx)





# ---------------------------------------------------------------------
# Per-agent preprocessing
# ---------------------------------------------------------------------

def preprocess_agent_fields(agent, params=None, G_q=None, G_p=None, *, ctx: Optional[object] = None):
    """
    Prepare per-agent fields (generators, Σ sanitize/inv, cond proxies, Fisher).
    NOTE: No cache pre-warming here (E/Jinv/Ω). That is handled centrally by ensure_dirty(...).
    """

    import numpy as np
    import core.config as config
    

    params     = params or {}
    eps        = float(getattr(config, "eps", 1e-8))
    group_name = str(getattr(config, "group_name", "so3"))
    strict     = bool(params.get("sanitize_strict", True))
    step_tag   = int(params.get("step", getattr(ctx, "global_step", -1) or -1))

    # idempotency per step
    if getattr(agent, "_preprocessed_step", None) == step_tag and step_tag >= 0:
        return

    # infer sizes & ensure mask
    try:
        Kq = int(agent.mu_q_field.shape[-1])
        Kp = int(agent.mu_p_field.shape[-1])
    except Exception as e:
        raise ValueError("preprocess_agent_fields: agent.mu_q_field / mu_p_field missing or malformed") from e

    if not hasattr(agent, "mask") or agent.mask is None:
        H, W = agent.mu_q_field.shape[:2]
        agent.mask = np.ones((H, W), np.float32)

    # generators: prefer provided, else build
    if G_q is not None: agent.generators_q = G_q
    if G_p is not None: agent.generators_p = G_p

    G_q_eff = getattr(agent, "generators_q", None)
    if G_q_eff is None or (isinstance(G_q_eff, np.ndarray) and G_q_eff.size == 0):
        G_q_eff, _ = get_generators(group_name, Kq, return_meta=True)
        agent.generators_q = G_q_eff

    G_p_eff = getattr(agent, "generators_p", None)
    if G_p_eff is None or (isinstance(G_p_eff, np.ndarray) and G_p_eff.size == 0):
        G_p_eff, _ = get_generators(group_name, Kp, return_meta=True)
        agent.generators_p = G_p_eff

    # SPD sanitize & explicit inverses (kept for compatibility; downstream can use Fisher cache)
    agent.sigma_q_field = sanitize_sigma(agent.sigma_q_field, strict=strict, eps=eps)
    agent.sigma_p_field = sanitize_sigma(agent.sigma_p_field, strict=strict, eps=eps)
    agent.sigma_q_inv   = safe_inv(agent.sigma_q_field, eps=eps)
    agent.sigma_p_inv   = safe_inv(agent.sigma_p_field, eps=eps)

    # condition proxies
    agent._cond_proxy_q = masked_cond_proxy(agent.sigma_q_field, agent.mask)
    agent._cond_proxy_p = masked_cond_proxy(agent.sigma_p_field, agent.mask)

    # Fisher caches (optional precompute).
    # This *computes Fisher explicitly if enabled*, which may populate cache as a side effect.
    # It is NOT a transport pre-warm and is OK to keep here for faster downstream steps.
    enable_fisher = bool(params.get("enable_fisher_precompute",
                          getattr(config, "enable_fisher_precompute", True)))
    if ctx is not None and enable_fisher:
        invF_q = invF_p = None
        try:
            # Prefer fully cached inverse-Fisher metric if available
            from transport.transport_cache import Fisher_blocks
            invF_q = _invF_cached(ctx, agent, agent.generators_q, which="q")
            invF_p = _invF_cached(ctx, agent, agent.generators_p, which="p")
            if invF_q is None or invF_p is None:
                Fq = Fisher_blocks(ctx, agent, which="q")
                Fp = Fisher_blocks(ctx, agent, which="p")
                agent._precision_q = Fq.get("precision", None)
                agent._precision_p = Fp.get("precision", None)
        except Exception:
            # Fall back to no Fisher precompute; downstream will compute lazily as needed.
            agent._precision_q = None
            agent._precision_p = None
            invF_q = None
            invF_p = None
        agent.inverse_fisher_phi = invF_q
        agent.inverse_fisher_phi_model = invF_p
    else:
        # Explicitly clear to avoid stale tensors lingering
        agent.inverse_fisher_phi = None
        agent.inverse_fisher_phi_model = None
        agent._precision_q = None
        agent._precision_p = None

    # optional SPD assert
    if bool(getattr(config, "assert_spd_after_sanitize", False)):
        wq = np.linalg.eigvalsh(0.5 * (agent.sigma_q_field + np.swapaxes(agent.sigma_q_field, -1, -2)))
        wp = np.linalg.eigvalsh(0.5 * (agent.sigma_p_field + np.swapaxes(agent.sigma_p_field, -1, -2)))
        floor = float(getattr(config, "sigma_eig_floor", 1e-6))
        if (wq <= floor).any() or (wp <= floor).any():
            raise RuntimeError("Non-SPD Σ after sanitize.")

    agent._preprocessed_step = step_tag





def zero_all_gradients(agent):
    """
    Zero (and if missing, create) all gradient buffers,
    keeping legacy aliases in sync and ensuring arrays are writeable.
    """
    def ensure_buf(name, like):
        arr = getattr(agent, name, None)
        if arr is None and like is not None:
            arr = np.zeros_like(like)
        elif arr is not None and not arr.flags.writeable:
            arr = np.array(arr, copy=True)
        setattr(agent, name, arr)
        return arr

    # Canonical grads (create from fields if needed)
    g_mu_q   = ensure_buf("grad_mu_q",    getattr(agent, "mu_q_field", None))
    g_sg_q   = ensure_buf("grad_sigma_q", getattr(agent, "sigma_q_field", None))
    g_mu_p   = ensure_buf("grad_mu_p",    getattr(agent, "mu_p_field", None))
    g_sg_p   = ensure_buf("grad_sigma_p", getattr(agent, "sigma_p_field", None))
    g_phi    = ensure_buf("grad_phi",       getattr(agent, "phi", None))
    g_phit   = ensure_buf("grad_phi_tilde", getattr(agent, "phi_model", None))

    # DEPRECATED: morphism gradient buffers — only zero if they already exist.
    g_Phi  = getattr(agent, "grad_Phi", None)
    if isinstance(g_Phi, np.ndarray) and not g_Phi.flags.writeable:
        g_Phi = np.array(g_Phi, copy=True); setattr(agent, "grad_Phi", g_Phi)
    g_Phit = getattr(agent, "grad_Phi_tilde", None)
    if isinstance(g_Phit, np.ndarray) and not g_Phit.flags.writeable:
        g_Phit = np.array(g_Phit, copy=True); setattr(agent, "grad_Phi_tilde", g_Phit)
 
    # Keep legacy aliases in sync (as references, not copies)
    if g_mu_q is not None:
        agent.grad_mu_q_field = g_mu_q
    if g_sg_q is not None:
        agent.grad_sigma_q_field = g_sg_q
    if g_mu_p is not None:
        agent.grad_mu_p_field = g_mu_p
    if g_sg_p is not None:
        agent.grad_sigma_p_field = g_sg_p

    # Zero safely
    for arr in (g_mu_q, g_sg_q, g_mu_p, g_sg_p, g_phi, g_phit, g_Phi, g_Phit):
        if isinstance(arr, np.ndarray):
            arr.fill(0)



