
import numpy as np
from core.omega import (compute_curvature_gradient_analytical)
from core.natural_gradient import compute_fisher_geometry_gradient
from core.numerical_utils import safe_inv, sanitize_sigma, safe_omega_inv
 
from transport.transport_cache import (E_grid, Jinv_grid, 
                                       Omega, build_dexpinv_matrix )
import inspect
from core.omega import d_exp_phi_exact, d_exp_phi_tilde_exact
from core.omega import exp_lie_algebra_irrep
#==============================================================================
#
#                    GATHER PHI GAUGE FIELD GRADIENT TERMS
#                          Beliefs and Models
#==============================================================================

def accumulate_phi_alignment_gradient( 
    agent_i, agent_j, generators, params,
    field="phi", beta_key=None, omega_cache_key=None,
    mu_key=None, sigma_key=None,
    fisher_inv_key=None, grad_key=None,
    Q_all_i=None,           # list/array (d, ..., K, K)
    exp_neg_phi_j=None,     # (..., K, K)
    agents=None,            # for Fisher regularizer
):
    """
    Same-level φ/φ̃ alignment gradient with minimal logging:
      - computes ∂ KL_i←j / ∂φ_i in parameter coords
      - pre/post maps via Jinv as covector/vector respectively
      - adds Lie-natural regularizers (already in body frame)
    Prints a single summary line with the sign (+/-/0) of the spatially
    averaged accumulated gradient (per component).
    """
    import numpy as np
    import core.config as config

    assert mu_key and sigma_key and fisher_inv_key and grad_key, \
        "mu_key/sigma_key/fisher_inv_key/grad_key are required"

    aid  = getattr(agent_i, "id", "?")
    ajid = getattr(agent_j, "id", "?")
    PRINT_SIGN = bool(params.get("DEBUG_PHI_SIGN", False))

    # --- resolve beta
    def _resolve_beta(_key):
        if not _key:
            return float(params.get("beta", getattr(config, "beta", 0.0)))
        cand = [_key, _key.replace("-", "_")]
        cand += (["beta"] if field == "phi" else ["beta_model"])
        for k in cand:
            if k in params: return float(params[k])
            if hasattr(config, k):
                try: return float(getattr(config, k))
                except Exception: pass
        return float(getattr(config, _key, 0.0))

    beta = _resolve_beta(beta_key)
    if beta <= 0.0:
        return

    # --- context & mask
    ctx = params.get("runtime_ctx", None) or params.get("ctx", None)
    tau = float(getattr(ctx, "support_tau", getattr(config, "support_tau",
                     getattr(config, "support_cutoff_eps", 1e-6))))
    Mi = (np.asarray(getattr(agent_i, "mask"), np.float32) > tau)
    Mj = (np.asarray(getattr(agent_j, "mask"), np.float32) > tau)
    joint_mask = (Mi & Mj)[..., None].astype(np.float32)
    if not np.any(joint_mask):
        return

    # --- fields
    mu_i    = np.asarray(getattr(agent_i, mu_key),    np.float32)
    sigma_i = np.asarray(getattr(agent_i, sigma_key), np.float32)
    mu_j    = np.asarray(getattr(agent_j, mu_key),    np.float32)
    which = "q" if field == "phi" else "p"

    # --- helpers expected in module scope
    # grad_kl_wrt_phi_i, apply_regularization_terms_lie_natural,
    # sanitize_sigma, safe_inv, safe_omega_inv,
    # E_grid, Jinv_grid, build_dexpinv_matrix,
    # d_exp_phi_exact, d_exp_phi_tilde_exact

    eps_num = float(getattr(config, "eps", 1e-6))

    # --- transports
    if ctx is not None:
        from transport.transport_cache import E_grid, Omega
        E_i = E_grid(ctx, agent_i, which=which)
        E_j = E_grid(ctx, agent_j, which=which)
        Omega_ij = Omega(ctx, agent_i, agent_j, which=which)
    else:
       
        phi_i_loc = np.asarray(getattr(agent_i, field), np.float32)
        phi_j_loc = np.asarray(getattr(agent_j, field), np.float32)
        E_i = exp_lie_algebra_irrep(phi_i_loc, generators)
        E_j = exp_lie_algebra_irrep(phi_j_loc, generators)
        Omega_ij = np.matmul(E_i, safe_omega_inv(E_j, eps=eps_num))
    Omega_T = np.swapaxes(Omega_ij, -1, -2)

    # --- push μ_j and Σ_j^{-1}
    mu_j_t = np.einsum("...ij,...j->...i", Omega_ij, mu_j, optimize=True)

    inv_key = sigma_key.replace("_field", "_inv")
    sigma_j_inv = getattr(agent_j, inv_key, None)

    if sigma_j_inv is not None:
        Oinv   = safe_omega_inv(Omega_ij, eps=eps_num)
        Oinv_T = np.swapaxes(Oinv, -1, -2)
        sigma_j_t_inv = np.einsum("...ik,...kl,...jl->...ij",
                                  Oinv_T, np.asarray(sigma_j_inv, np.float32), Oinv, optimize=True)
        # guard large magnitudes
        max_inv_norm = float(params.get("inverse_push_max_norm",
                                        getattr(config, "inverse_push_max_norm", 1e4)))
        if (not np.all(np.isfinite(sigma_j_t_inv))) or (np.max(np.abs(sigma_j_t_inv)) > max_inv_norm):
            sigma_j   = sanitize_sigma(np.asarray(getattr(agent_j, sigma_key), np.float32), eps=eps_num)
            sigma_j_t = np.einsum("...ik,...kl,...jl->...ij", Omega_ij, sigma_j, Omega_T, optimize=True)
            sigma_j_t = sanitize_sigma(sigma_j_t, eps=eps_num)
            sigma_j_t_inv = safe_inv(sigma_j_t, eps=eps_num)
    else:
        sigma_j   = sanitize_sigma(np.asarray(getattr(agent_j, sigma_key), np.float32), eps=eps_num)
        sigma_j_t = np.einsum("...ik,...kl,...jl->...ij", Omega_ij, sigma_j, Omega_T, optimize=True)
        sigma_j_t = sanitize_sigma(sigma_j_t, eps=eps_num)
        sigma_j_t_inv = safe_inv(sigma_j_t, eps=eps_num)

    # --- Jacobians
    if Q_all_i is None:
        if field == "phi":
            from core.omega import d_exp_phi_exact
            Q_all_i = d_exp_phi_exact(np.asarray(getattr(agent_i, field), np.float32), generators)
        else:
            from core.omega import d_exp_phi_tilde_exact
            Q_all_i = d_exp_phi_tilde_exact(np.asarray(getattr(agent_i, field), np.float32), generators)
    if exp_neg_phi_j is None:
        exp_neg_phi_j = safe_omega_inv(E_j, eps=eps_num)

    # --- raw parameter-covector gradient
    grad_param = grad_kl_wrt_phi_i(
        mu_i, sigma_i,
        None, None,
        phi_i = np.asarray(getattr(agent_i, field), np.float32),
        phi_j = None,
        Omega_ij = Omega_ij,
        generators = generators,
        exp_phi_i = E_i,
        exp_phi_j = E_j,
        mu_j_t = mu_j_t,
        sigma_j_t_inv = sigma_j_t_inv,
        eps = eps_num,
        exp_neg_phi_j = exp_neg_phi_j,
        Q_all = Q_all_i,
    ).astype(np.float32, copy=False)

    # --- J^{-T}: parameter covector -> body covector
    Jinv = Jinv_grid(ctx, agent_i, which=which) if ctx is not None else None
    if Jinv is None:
        Jinv = build_dexpinv_matrix(np.asarray(getattr(agent_i, field), np.float32))
    g_body_cov = np.einsum("...ji,...j->...i", Jinv, grad_param, optimize=True)  # (..., d)
    d = int(g_body_cov.shape[-1])

    # --- Fisher^{-1} in body (support full/diag/scalar)
    F_inv = getattr(agent_i, fisher_inv_key, None)
    v_body = g_body_cov
    if F_inv is not None:
        Fi = np.asarray(F_inv, np.float32)
        if np.all(np.isfinite(Fi)) and Fi.size > 0:
            # optional clamp to avoid blow-ups
            fisher_max_norm = float(params.get("fisher_max_norm",
                                               getattr(config, "fisher_max_norm", 1e3)))
            Finf = np.max(np.abs(Fi))
            if np.isfinite(Finf) and Finf > fisher_max_norm:
                Fi = Fi * (fisher_max_norm / (Finf + 1e-12))
            if Fi.ndim >= 2 and Fi.shape[-2:] == (d, d):
                v_body = np.einsum("...ab,...b->...a", Fi, g_body_cov, optimize=True)
            elif Fi.shape[-1:] == (d,):
                v_body = g_body_cov * Fi
            elif Fi.ndim == g_body_cov.ndim - 1:
                v_body = g_body_cov * Fi[..., None]

    # --- regularizer (already NATURAL/body), coerce to vector shape
    reg_nat = apply_regularization_terms_lie_natural(
        agent=agent_i,
        phi_field=np.asarray(getattr(agent_i, field), np.float32),
        generators=generators,
        field=field,
        params={**params, "step_tag": getattr(ctx, "global_step", None) if ctx is not None else None},
        agents=agents,
        ctx=ctx,
    )
    reg_nat = np.asarray(reg_nat, np.float32)
    if reg_nat.ndim == v_body.ndim - 1:
        reg_nat = reg_nat[..., None]
    elif reg_nat.shape[-1:] != (1,) and reg_nat.shape[-1:] != (d,):
        reg_nat = np.zeros_like(v_body, dtype=np.float32)

    v_body_total = v_body + reg_nat

    # --- J^{-1}: body vector -> parameter vector
    v_param = np.einsum("...ab,...b->...a", Jinv, v_body_total, optimize=True)
    v_param = np.nan_to_num(v_param, nan=0.0, posinf=0.0, neginf=0.0)

    # --- optional per-pixel norm clip for stability
    clip_norm = float(params.get("phi_grad_clip_norm",
                                 getattr(config, "phi_grad_clip_norm", 1e6)))
    vp_norm = np.linalg.norm(v_param, axis=-1, keepdims=True)
    scale   = np.minimum(1.0, clip_norm / (vp_norm + 1e-9))
    v_param = v_param * scale

    # --- accumulate (mask & weight)
    contrib = (beta * v_param) * joint_mask
    grad_accum = getattr(agent_i, grad_key, None)
    if grad_accum is None:
        grad_accum = np.zeros_like(contrib, dtype=np.float32)
    setattr(agent_i, grad_key, grad_accum + contrib)

    # --- single compact sign print (spatial average, per component)
    if PRINT_SIGN:
        # mean over spatial axes (all but last)
        spatial_axes = tuple(range(contrib.ndim - 1))
        g_mean = np.mean(contrib, axis=spatial_axes)  # shape (d,)
        def _sgn(x):
            return "+" if x > 0 else ("-" if x < 0 else "0")
        sign_str = "".join(_sgn(x) for x in g_mean.tolist())
        # also show tiny summary magnitude to detect blowups
        g_abs_mean = float(np.mean(np.abs(contrib)))
        print(f"[PHI-GRAD] field={field} ai={aid} aj={ajid} sign={sign_str} "
              f"mean={np.array2string(g_mean, precision=3, suppress_small=True)} "
              f"|g|_mean={g_abs_mean:.3e}", flush=True)



def accumulate_phi_morphism_self_feedback(
    agent_i,
    generators_src, generators_tgt,
    params,
    field="phi",                # or "phi_model"
    fisher_inv_key=None,
    grad_key=None,
    phi_self_func=None,
    phi_feedback_func=None,
):
    """
    Self/feedback gradients for φ or φ̃ with minimal logging.
    Prints one compact sign line of the spatially-averaged accumulated gradient.
    """
    import numpy as np
    import inspect
    import core.config as config

    # ---- config & toggles
    PRINT_SIGN = bool(params.get("DEBUG_PHI_SIGN", False))
    eps       = float(getattr(config, "eps", 1e-6))
    eps_sup   = float(getattr(config, "support_cutoff_eps", 1e-6))
    fisher_max_norm = float(params.get("fisher_max_norm",
                                       getattr(config, "fisher_max_norm", 1e3)))
    clip_norm = float(params.get("phi_grad_clip_norm",
                                 getattr(config, "phi_grad_clip_norm", 1e6)))

    # --- call helper respects callee signature (avoids passing unknown kwargs)
    def _call_sig_aware(fn, **kwargs):
        if fn is None:
            return None
        sig = inspect.signature(fn).parameters
        return fn(**{k: v for k, v in kwargs.items() if k in sig})

    alpha   = float(params.get("alpha",           getattr(config, "alpha",           1.0)))
    lambda_ = float(params.get("feedback_weight", getattr(config, "feedback_weight", 0.0)))
    if alpha <= 0.0 and lambda_ <= 0.0:
        return

    # ---- mask to (...,1)
    mask = (np.asarray(agent_i.mask) > eps_sup).astype(np.float32)
    if mask.ndim == 2:  # (H,W)
        mask = mask[..., None]
    elif mask.ndim == 3 and mask.shape[-1] != 1:
        mask = mask[..., :1]  # force 1-channel
    if not np.any(mask):
        # quiet—no grads printed
        return

    # ---- fields
    mu_q, sigma_q = agent_i.mu_q_field, agent_i.sigma_q_field
    mu_p, sigma_p = agent_i.mu_p_field, agent_i.sigma_p_field
    phi       = np.asarray(agent_i.phi,       np.float32)
    phi_tilde = np.asarray(agent_i.phi_model, np.float32)

    # ---- context, transports
    ctx = params.get("runtime_ctx", None)
    if ctx is not None:
        from transport.transport_cache import E_grid, Jinv_grid
        exp_phi       = E_grid(ctx, agent_i, which="q")
        exp_phi_tilde = E_grid(ctx, agent_i, which="p")
    else:
       
        exp_phi       = exp_lie_algebra_irrep(phi,       generators_tgt)
        exp_phi_tilde = exp_lie_algebra_irrep(phi_tilde, generators_src)
        Jinv_grid = None  # to appease linter

    # choose field + Jinv + generator roles
    if field == "phi":
        which = "q"
        base_field = phi
        Jinv = Jinv_grid(ctx, agent_i, which="q") if ctx is not None else build_dexpinv_matrix(base_field)
        G_src, G_tgt = generators_src, generators_tgt
    else:
        which = "p"
        base_field = phi_tilde
        Jinv = Jinv_grid(ctx, agent_i, which="p") if ctx is not None else build_dexpinv_matrix(base_field)
        G_src, G_tgt = generators_tgt, generators_src

    # Fisher^{-1} in body frame (vector)
    F_inv = getattr(agent_i, fisher_inv_key, None)

    def _apply_F_inv_body(vec_body):
        """Apply Fisher^{-1} in body; supports full/diag/scalar with soft clamp."""
        if F_inv is None:
            return vec_body
        Fi = np.asarray(F_inv, np.float32)
        if not (np.all(np.isfinite(Fi)) and Fi.size > 0):
            return vec_body
        # clamp ∞-norm
        Finf = float(np.max(np.abs(Fi)))
        if np.isfinite(Finf) and Finf > fisher_max_norm:
            Fi = Fi * (fisher_max_norm / (Finf + 1e-12))
        if Fi.ndim >= 2 and Fi.shape[-2:] == (3, 3):
            return np.einsum("...ab,...b->...a", Fi, vec_body, optimize=True)
        if Fi.ndim == vec_body.ndim - 1:   # scalar per-pixel (...,)
            return vec_body * Fi[..., None]
        if Fi.shape[-1:] == (3,):          # diag (...,3)
            return vec_body * Fi
        return vec_body

    def _accum_term(grad_param_eucl, weight):
        if grad_param_eucl is None or weight <= 0.0:
            return
        grad_param = np.asarray(grad_param_eucl, np.float32)
        # parameter covector -> body covector (Jinv^T)
        g_body_cov = np.einsum("...ji,...j->...i", Jinv, grad_param, optimize=True)
        # Fisher^{-1} in body frame
        v_body = _apply_F_inv_body(g_body_cov)
        # back to parameter vector (Jinv)
        v_param = np.einsum("...ab,...b->...a", Jinv, v_body, optimize=True)
        v_param = np.nan_to_num(v_param, nan=0.0, posinf=0.0, neginf=0.0)

        # norm clip per-pixel for stability
        vp_norm = np.linalg.norm(v_param, axis=-1, keepdims=True)
        scale   = np.minimum(1.0, clip_norm / (vp_norm + 1e-9))
        v_param = v_param * scale

        acc = getattr(agent_i, grad_key, None)
        if acc is None:
            acc = np.zeros_like(v_param, dtype=np.float32)
        setattr(agent_i, grad_key, acc + weight * v_param * mask)

        return weight * v_param * mask  # return contribution for sign print

    # --- SELF term
    self_contrib = None
    if alpha > 0.0 and phi_self_func is not None:
        if field == "phi":
            args = dict(
                mu_q=mu_q, sigma_q=sigma_q, mu_p=mu_p, sigma_p=sigma_p,
                Phi_tilde_0=getattr(agent_i, "Phi_tilde_0", None),
                phi=phi, phi_tilde=phi_tilde,
                exp_phi=exp_phi, exp_phi_tilde=exp_phi_tilde,
                eps=eps, generators_q=G_tgt, generators_p=G_src,
            )
        else:  # phi_model
            args = dict(
                mu_q=mu_q, sigma_q=sigma_q, mu_p=mu_p, sigma_p=sigma_p,
                Phi_tilde_0=getattr(agent_i, "Phi_tilde_0", None),
                phi=phi, phi_tilde=phi_tilde,
                exp_phi=exp_phi, exp_phi_tilde=exp_phi_tilde,
                eps=eps, generators_q=G_src, generators_p=G_tgt,
            )
        grad_self_param = _call_sig_aware(phi_self_func, **args)
        self_contrib = _accum_term(grad_self_param, alpha)

    # --- FEEDBACK term
    fb_contrib = None
    if lambda_ > 0.0 and phi_feedback_func is not None:
        if field == "phi":
            args = dict(
                mu_p=mu_p, sigma_p=sigma_p, mu_q=mu_q, sigma_q=sigma_q,
                Phi_0=getattr(agent_i, "Phi_0", None),
                phi=phi, phi_tilde=phi_tilde,
                exp_phi=exp_phi, exp_phi_tilde=exp_phi_tilde,
                eps=eps, generators_q=G_tgt,
            )
        else:  # phi_model
            args = dict(
                mu_p=mu_p, sigma_p=sigma_p, mu_q=mu_q, sigma_q=sigma_q,
                Phi_0=getattr(agent_i, "Phi_0", None),
                phi=phi, phi_tilde=phi_tilde,
                exp_phi=exp_phi, exp_phi_tilde=exp_phi_tilde,
                eps=eps, generators_p=G_src, generators_q=G_tgt,
            )
        grad_fb_param = _call_sig_aware(phi_feedback_func, **args)
        fb_contrib = _accum_term(grad_fb_param, lambda_)

    if PRINT_SIGN:
        aid = getattr(agent_i, "id", "?")
        # total contribution this call
        total = None
        if self_contrib is not None and fb_contrib is not None:
            total = self_contrib + fb_contrib
        elif self_contrib is not None:
            total = self_contrib
        elif fb_contrib is not None:
            total = fb_contrib

        def _print_sign(tag, tensor):
            if tensor is None:
                return
            spatial_axes = tuple(range(tensor.ndim - 1))
            g_mean = np.mean(tensor, axis=spatial_axes)
            def _sgn(x): return "+" if x > 0 else ("-" if x < 0 else "0")
            sign_str = "".join(_sgn(x) for x in g_mean.tolist())
            g_abs_mean = float(np.mean(np.abs(tensor)))
            print(f"[PHI-MORPH] field={field} ai={aid} term={tag} sign={sign_str} "
                  f"mean={np.array2string(g_mean, precision=3, suppress_small=True)} "
                  f"|g|_mean={g_abs_mean:.3e}", flush=True)

        _print_sign("self", self_contrib)
        _print_sign("feedback", fb_contrib)
        _print_sign("total", total)



#==============================================================================
#
#                    WITH RESPECT TO phi_i TERMS
#                       VALIDATED
#==============================================================================
def grad_kl_wrt_phi_i(
    mu_i, sigma_i,
    mu_j, sigma_j,
    phi_i, phi_j,
    Omega_ij,
    generators,
    exp_phi_i,
    exp_phi_j,
    mu_j_t,
    sigma_j_t_inv,
    eps=1e-8,
    exp_neg_phi_j=None,
    Q_all=None,
):
    """
    Vectorized over Lie index a. Returns (..., d) float32.
    Robustly handles Q_all stacking/broadcasting so that:
        Q_all.shape == (..., a, K, K)
        exp_neg_phi_j.shape == (..., K, K) broadcastable to Q_all's leading dims
    """
    import numpy as _np
    # local safety imports
    from core.numerical_utils import sanitize_sigma
    from core.omega import d_exp_phi_exact, safe_omega_inv

    # ---- helpers -------------------------------------------------------------
    def _stack_Q_all_right_before_KK(Q_list_or_array, K_expected=None):
        """
        Ensure Q has shape (..., a, K, K), i.e., Lie axis 'a' is the -3 axis.
        Accepts:
          - list/tuple of length d with arrays shaped (..., K, K)
          - array with 'a' axis anywhere before the trailing K,K
        """
        if isinstance(Q_list_or_array, (list, tuple)):
            sample = Q_list_or_array[0]
            nd = sample.ndim
            Q_arr = _np.stack(Q_list_or_array, axis=nd - 2)  # insert 'a' right before K,K
        else:
            Q_arr = _np.asarray(Q_list_or_array)

        if Q_arr.ndim < 3:
            raise ValueError(f"Q_all has too few dims: {Q_arr.shape}")
        K1, K2 = Q_arr.shape[-2], Q_arr.shape[-1]
        if K_expected is not None and (K1 != K_expected or K2 != K_expected):
            raise ValueError(f"Q_all trailing dims {K1}x{K2} != expected {K_expected}x{K_expected}")
        d = int(generators.shape[0])
        if Q_arr.shape[-3] != d:
            a_axis = None
            for ax in range(Q_arr.ndim - 2):
                if Q_arr.shape[ax] == d:
                    a_axis = ax
                    break
            if a_axis is None:
                raise ValueError(f"Could not locate Lie axis of size d={d} in Q_all shape {Q_arr.shape}")
            if a_axis != Q_arr.ndim - 3:
                Q_arr = _np.moveaxis(Q_arr, a_axis, Q_arr.ndim - 3)
        return Q_arr

    def _broadcast_KK_to_leading(mat, lead_shape, K):
        M = _np.asarray(mat)
        if M.ndim < 2 or M.shape[-2:] != (K, K):
            raise ValueError(f"Expected a (...,{K},{K}) matrix; got {M.shape}")
        if M.ndim == 2:
            M = M.reshape((1,) * len(lead_shape) + (K, K))
        if M.shape[:-2] != tuple(lead_shape):
            M = _np.broadcast_to(M, tuple(lead_shape) + (K, K))
        return M

    def _broadcast_vec_to_leading(vec, lead_shape, K):
        v = _np.asarray(vec)
        if v.ndim == 1:
            v = v.reshape((1,) * len(lead_shape) + (K,))
        elif v.shape[-1] != K:
            raise ValueError(f"Vector trailing dim mismatch {v.shape[-1]} vs {K}")
        if v.shape[:-1] != tuple(lead_shape):
            v = _np.broadcast_to(v, tuple(lead_shape) + (K,))
        return v

    # ---- inputs & shapes -----------------------------------------------------
    sigma_i = sanitize_sigma(sigma_i, eps=eps)
    K = int(sigma_i.shape[-1])

    # Q_all: d Jacobians of exp(phi_i); ensure (..., a, K, K)
    if Q_all is None:
        Q_all = d_exp_phi_exact(phi_i, generators)  # list of length d
    Q = _stack_Q_all_right_before_KK(Q_all, K_expected=K)  # (..., a, K, K)

    # exp(-phi_j): prefer cached; broadcast to Q's leading dims (excluding a,K,K)
    if exp_neg_phi_j is None:
        exp_neg_phi_j = safe_omega_inv(exp_phi_j, eps=eps, debug=False)
    exp_neg_phi_j = _broadcast_KK_to_leading(exp_neg_phi_j, lead_shape=Q.shape[:-3], K=K)

    # push-forward pieces
    delta_mu = mu_i - mu_j_t  # (..., K)

    # Also reconstruct μ_j in the source frame for mean-term consistency
    Oinv = safe_omega_inv(Omega_ij, eps=eps, debug=False)  # (..., K, K)
    mu_j_src = _np.einsum("...ij,...j->...i", Oinv, mu_j_t, optimize=True)  # (..., K)

    # ---- terms ---------------------------------------------------------------
    # dΩ/dφ_i^a = Q[a] @ e^{-φ_j}
    Q = _np.einsum("...aik,...kj->...aij", Q, exp_neg_phi_j, optimize=True)  # (..., a, K, K)
    Q_T = _np.swapaxes(Q, -1, -2)  # (..., a, K, K)

    # trace term: -0.5 * [ Tr(Sj_inv Q Σi) + Tr(Q^T Sj_inv Σi) ]
    t1 = _np.einsum("...ij,...ajk,...ki->...a", sigma_j_t_inv, Q, sigma_i, optimize=True)
    t2 = _np.einsum("...aij,...jk,...ki->...a", Q_T,           sigma_j_t_inv, sigma_i, optimize=True)
    trace_term = -0.5 * (t1 + t2)  # (..., a)

    # mean term:  - μ_j^T (Sj_inv (Q^T Δμ))   with μ_j in the source frame
    tmp1 = _np.einsum("...aij,...j->...ai", Q_T, delta_mu, optimize=True)          # Q^T Δμ
    tmp2 = _np.einsum("...ij,...aj->...ai", sigma_j_t_inv, tmp1, optimize=True)    # Sj_inv (...)
    mean_term = -_np.einsum("...i,...ai->...a", mu_j_src, tmp2, optimize=True)

    # Mahalanobis term: 0.5 Δμ^T ( Q^T Sj_inv + Sj_inv Q ) Δμ
    v1 = _np.einsum("...ij,...j->...i", sigma_j_t_inv, delta_mu, optimize=True)   # Sj_inv Δμ
    part1 = _np.einsum("...aij,...j->...ai", Q_T, v1, optimize=True)              # Q^T Sj_inv Δμ
    v2 = _np.einsum("...aij,...j->...ai", Q,  delta_mu, optimize=True)            # Q Δμ
    part2 = _np.einsum("...ij,...aj->...ai", sigma_j_t_inv, v2, optimize=True)    # Sj_inv Q Δμ
    mahal_term = 0.5 * _np.einsum("...i,...ai->...a", delta_mu, part1 + part2, optimize=True)

    grad = (trace_term + mean_term + mahal_term).astype(_np.float32, copy=False)  # (..., a)
    return grad





def grad_feedback_phi_direct(
    mu_p, sigma_p,
    mu_q, sigma_q,
    Phi_0,
    phi,                # (..., d)
    phi_tilde,          # (..., d)
    generators_q,       # (d, K_q, K_q)
    exp_phi,            # (..., K_q, K_q)
    exp_phi_tilde,      # (..., K_p, K_p)
    eps=1e-8,
    # NEW optional precomputes:
    Q_all=None,         # (..., d, K_q, K_q) = d_exp_phi_exact(phi, generators_q)
    exp_neg_phi=None    # (..., K_q, K_q) = e^{-φ}
):
    """
    Vectorized over 'a': returns (..., d)
    Φ = e^{φ̃} Φ₀ e^{-φ}, ∂Φ = - Φ · (e^{-φ} (∂ e^{φ}) e^{-φ})
    """
    d, K_q, _ = generators_q.shape
    

    if exp_neg_phi is None:
        exp_neg_phi = safe_omega_inv(exp_phi, eps=eps, debug=False)
    if Q_all is None:
        # stack as (..., d, K_q, K_q)
        Q_list = d_exp_phi_exact(phi, generators_q)
        Q_all = np.stack(Q_list, axis=-3)  # assuming list of d arrays

    # Φ and Φᵀ
    Phi = np.einsum("...ik,...kj,...jl->...il", exp_phi_tilde, Phi_0, exp_neg_phi)
    Phi_T = np.swapaxes(Phi, -1, -2)


    # A, A^{-1}
    A = np.einsum("...ik,...kl,...lj->...ij", Phi, sigma_q, Phi_T)
    A = sanitize_sigma(A, eps=eps)
    A_inv = safe_inv(A, eps=eps)

    # Δμ and v
    Phi_mu_q = np.einsum("...ij,...j->...i", Phi, mu_q)
    delta_mu = mu_p - Phi_mu_q
    v = np.einsum("...ij,...j->...i", A_inv, delta_mu)  # (..., K_p)

    # Q̄_a = e^{-φ} Q_a e^{-φ}
    Q_bar = np.einsum("...ik,...akl,...lj->...aij", exp_neg_phi, Q_all, exp_neg_phi)

    # dΦ_a = - Φ Q̄_a
    dPhi = -np.einsum("...ik,...akj->...aij", Phi, Q_bar)
    dPhi_T = np.swapaxes(dPhi, -1, -2)

    # dA_a
    dA = (
        np.einsum("...aik,...kl,...lj->...aij", dPhi, sigma_q, Phi_T) +
        np.einsum("...ik,...kl,...alj->...aij", Phi,  sigma_q, dPhi_T)
    )

    # Term 1
    dPhi_mu = np.einsum("...aij,...j->...ai", dPhi, mu_q)       # (..., a, K_p)
    term1 = -np.einsum("...ai,...i->...a", dPhi_mu, v)

    # Term 2
    term2 = 0.5 * np.einsum("...ij,...aij->...a", A_inv, dA)

    # M_a = A^{-1} dA_a A^{-1}
    M = np.einsum("...ij,...ajk,...kl->...ail", A_inv, dA, A_inv)

    # Term 3
    term3 = -0.5 * np.einsum("...i,...aij,...j->...a", delta_mu, M, delta_mu)

    # Term 4
    term4 = -0.5 * np.einsum("...aij,...ji->...a", M, sigma_p)

    grad = (term1 + term2 + term3 + term4).astype(np.float32, copy=False)
    return grad


def grad_feedback_phi_tilde_direct(
    mu_p, sigma_p,
    mu_q, sigma_q,
    Phi_0,
    phi, phi_tilde,
    generators_p, generators_q,
    exp_phi, exp_phi_tilde,
    eps=1e-8,
    # NEW optional precomputes:
    Q_tilde_all=None,       # (..., d, K_p, K_p) = d_exp_phi_tilde_exact(phi_tilde, generators_p)
    exp_neg_phi_tilde=None  # (..., K_p, K_p) = e^{-φ̃}
):
    """
    Vectorized over 'a': returns (..., d)
    Left-trivialize: dΦ_a = L_a Φ,  L_a = (∂e^{φ̃}/∂φ̃^a) e^{-φ̃}.
    """
    d, K_p, _ = generators_p.shape
    

    if exp_neg_phi_tilde is None:
        exp_neg_phi_tilde = safe_omega_inv(exp_phi_tilde, eps=eps, debug=False)
    if Q_tilde_all is None:
        Q_list = d_exp_phi_tilde_exact(phi_tilde, generators_p)
        Q_tilde_all = np.stack(Q_list, axis=-3)

    # Φ
    exp_neg_phi = safe_omega_inv(exp_phi, eps=eps, debug=False)
    Phi   = np.einsum("...ik,...kj,...jl->...il", exp_phi_tilde, Phi_0, exp_neg_phi)
    Phi_T = np.swapaxes(Phi, -1, -2)

  
    A = np.einsum("...ik,...kl,...lj->...ij", Phi, sigma_q, Phi_T)
    A = sanitize_sigma(A, eps=eps)
    A_inv = safe_inv(A, eps=eps)

    Phi_mu_q = np.einsum("...ij,...j->...i", Phi, mu_q)
    delta_mu = mu_p - Phi_mu_q
    v = np.einsum("...ij,...j->...i", A_inv, delta_mu)

    # L_a = Q̃_a e^{-φ̃}
    L = np.einsum("...aik,...kj->...aij", Q_tilde_all, exp_neg_phi_tilde)

    # dΦ_a = L_a Φ
    dPhi = np.einsum("...aik,...kj->...aij", L, Phi)
    dPhi_T = np.swapaxes(dPhi, -1, -2)

    dA = (
        np.einsum("...aik,...kl,...lj->...aij", dPhi, sigma_q, Phi_T) +
        np.einsum("...ik,...kl,...alj->...aij", Phi,  sigma_q, dPhi_T)
    )

    dPhi_mu = np.einsum("...aij,...j->...ai", dPhi, mu_q)
    term1 = -np.einsum("...ai,...i->...a", dPhi_mu, v)

    term2 = 0.5 * np.einsum("...ij,...aij->...a", A_inv, dA)

    M = np.einsum("...ij,...ajk,...kl->...ail", A_inv, dA, A_inv)

    term3 = -0.5 * np.einsum("...i,...aij,...j->...a", delta_mu, M, delta_mu)
    term4 = -0.5 * np.einsum("...aij,...ji->...a", M, sigma_p)

    grad = term1 + term2 + term3 + term4
    return grad



def grad_self_phi_direct(
    mu_q, sigma_q,
    mu_p, sigma_p,
    Phi_tilde_0,
    phi, phi_tilde,
    generators_q, generators_p,
    exp_phi, exp_phi_tilde,
    eps=1e-8,
    # NEW optional precomputes:
    Q_all=None,               # (..., d, K_q, K_q)
    exp_neg_phi_tilde=None    # (..., K_p, K_p)
):
    """
    Vectorized over 'a': returns (..., d)
    Φ̃ = e^{φ} Φ̃₀ e^{-φ̃},  ∂Φ̃_a = (∂e^{φ}/∂φ^a) Φ̃₀ e^{-φ̃}.
    """
    d, K_q, _ = generators_q.shape
    

    if exp_neg_phi_tilde is None:
        exp_neg_phi_tilde = safe_omega_inv(exp_phi_tilde, eps=eps, debug=False)
    if Q_all is None:
        Q_list = d_exp_phi_exact(phi, generators_q)
        Q_all = np.stack(Q_list, axis=-3)

    # Φ̃ and A
    Phi_tilde = np.einsum("...ik,...kj,...jl->...il", exp_phi, Phi_tilde_0, exp_neg_phi_tilde)
    Phi_tilde_T = np.swapaxes(Phi_tilde, -1, -2)


    A = np.einsum("...ik,...kl,...lj->...ij", Phi_tilde, sigma_p, Phi_tilde_T)
    A = sanitize_sigma(A, eps=eps)
    A_inv = safe_inv(A, eps=eps)

    delta_mu = mu_q - np.einsum("...ij,...j->...i", Phi_tilde, mu_p)

    # ∂Φ̃_a = Q_a Φ̃₀ e^{-φ̃}
    dPhi = np.einsum("...aik,...kj,...jl->...ail", Q_all, Phi_tilde_0, exp_neg_phi_tilde)
    dPhi_T = np.swapaxes(dPhi, -1, -2)

    dA = (
        np.einsum("...aik,...kl,...lj->...aij", dPhi, sigma_p, Phi_tilde_T) +
        np.einsum("...ik,...kl,...alj->...aij", Phi_tilde, sigma_p, dPhi_T)
    )

    # Term 1
    dPhi_mu_p = np.einsum("...aij,...j->...ai", dPhi, mu_p)
    term1 = -np.einsum("...ai,...i->...a", dPhi_mu_p, np.einsum("...ij,...j->...i", A_inv, delta_mu))

    # Term 2   (note: your earlier self/feedback sign conventions differ; keeping your current one)
    term2 = 0.5 * np.einsum("...ij,...aij->...a", A_inv, dA)

    # M = A^{-1} dA A^{-1}
    M = np.einsum("...ij,...ajk,...kl->...ail", A_inv, dA, A_inv)

    # Term 3
    term3 = -0.5 * np.einsum("...i,...aij,...j->...a", delta_mu, M, delta_mu)

    # Term 4
    term4 = -0.5 * np.einsum("...aij,...ji->...a", M, sigma_q)

    grad = term1 + term2 + term3 + term4
    return grad


def grad_self_phi_tilde_direct(
    mu_q, sigma_q,
    mu_p, sigma_p,
    Phi_tilde_0,
    phi, phi_tilde,
    generators_q, generators_p,
    exp_phi, exp_phi_tilde,
    eps=1e-8,
    # NEW optional precomputes:
    Q_tilde_all=None,         # (..., d, K_p, K_p)
    exp_neg_phi_tilde=None    # (..., K_p, K_p)
):
    """
    Vectorized over 'a': returns (..., d)
    Right-trivialize on model: R_a = (∂e^{φ̃}/∂φ̃^a) e^{-φ̃},  ∂Φ̃_a = - Φ̃ R_a.
    """
    d, K_p, _ = generators_p.shape
    
    if exp_neg_phi_tilde is None:
        exp_neg_phi_tilde = safe_omega_inv(exp_phi_tilde, eps=eps, debug=False)
    if Q_tilde_all is None:
        Q_list = d_exp_phi_tilde_exact(phi_tilde, generators_p)
        Q_tilde_all = np.stack(Q_list, axis=-3)

    Phi_tilde = np.einsum("...ik,...kj,...jl->...il", exp_phi, Phi_tilde_0, exp_neg_phi_tilde)
    Phi_tilde_T = np.swapaxes(Phi_tilde, -1, -2)

    
    A = np.einsum("...ik,...kl,...lj->...ij", Phi_tilde, sigma_p, Phi_tilde_T)
    A = sanitize_sigma(A, eps=eps)
    A_inv = safe_inv(A, eps=eps)

    mu_t = np.einsum("...ij,...j->...i", Phi_tilde, mu_p)
    delta_mu = mu_q - mu_t

    # R_a = Q̃_a e^{-φ̃}
    R = np.einsum("...aik,...kj->...aij", Q_tilde_all, exp_neg_phi_tilde)

    # dΦ̃_a = - Φ̃ R_a
    dPhi = -np.einsum("...ik,...akj->...aij", Phi_tilde, R)
    dPhi_T = np.swapaxes(dPhi, -1, -2)

    dA = (
        np.einsum("...aik,...kl,...lj->...aij", dPhi, sigma_p, Phi_tilde_T) +
        np.einsum("...ik,...kl,...alj->...aij", Phi_tilde, sigma_p, dPhi_T)
    )

    dPhi_mu_p = np.einsum("...aij,...j->...ai", dPhi, mu_p)
    term1 = -np.einsum("...ai,...i->...a", dPhi_mu_p, np.einsum("...ij,...j->...i", A_inv, delta_mu))

    term2 = 0.5 * np.einsum("...ij,...aij->...a", A_inv, dA)

    M = np.einsum("...ij,...ajk,...kl->...ail", A_inv, dA, A_inv)

    term3 = -0.5 * np.einsum("...i,...aij,...j->...a", delta_mu, M, delta_mu)
    term4 = -0.5 * np.einsum("...aij,...ji->...a", M, sigma_q)

    grad = term1 + term2 + term3 + term4
    return grad




# --- capped body-frame map for MASS term only -------------------------------
def _cap_jinv_for_mass(J, max_gain=10.0):
    """
    Cheap row-sum norm cap on Jinv so mass regularizer can't explode near θ≈π.
    Operates elementwise over (..., 3, 3).
    """
    
    eps = 1e-8
    rs = np.sum(np.abs(J), axis=-1)           # (..., 3)
    gain = np.max(rs, axis=-1, keepdims=True) # (..., 1)
    scale = np.minimum(1.0, float(max_gain) / np.maximum(gain, eps))  # (..., 1)
    return J * scale[..., None]               # (..., 3, 3)


# =========================
# Regularization (with toggleable φ–φ̃ coupling)
# =========================
def apply_regularization_terms_lie_natural(
    agent,
    phi_field,
    generators,
    field,
    params,
    agents=None,
    eps=1e-8,
    *,
    ctx=None,
):
    """
    Return NATURAL (body-frame) regularizer vector for φ or φ̃.
    """
    import numpy as np
    import core.config as config

    grad_nat = np.zeros_like(phi_field, dtype=np.float32)    # (..., d)
    d = int(grad_nat.shape[-1])

    # ---------------- weights: CONFIG defaults, then PARAMS override ----------------
    if field == "phi":
        curv_w_default   = float(getattr(config, "curvature_weight", 0.0))
        fisher_w_default = float(getattr(config, "fisher_geometry_weight", 0.0))
        mass_w_default   = float(getattr(config, "belief_mass", 0.0))
        cpl_w_default    = float(getattr(config, "phi_coupling_weight", 0.0))

        curv_w   = float(params.get("curvature_weight", curv_w_default))
        fisher_w = float(params.get("fisher_geometry_weight", fisher_w_default))
        mass_w   = float(params.get("mass_weight", params.get("belief_mass", mass_w_default)))
        cpl_w    = float(params.get("phi_coupling_weight", cpl_w_default))

        which = "q"
        gen_q = generators
        gen_p = getattr(agent, "generators_p", None)
    elif field == "phi_model":
        curv_w_default   = float(getattr(config, "model_curvature_weight", 0.0))
        fisher_w_default = float(getattr(config, "fisher_model_geometry_weight", 0.0))
        mass_w_default   = float(getattr(config, "model_mass", 0.0))
        cpl_w_default    = float(getattr(config, "phi_coupling_weight", 0.0))

        curv_w   = float(params.get("model_curvature_weight", params.get("curvature_weight", curv_w_default)))
        fisher_w = float(params.get("fisher_model_geometry_weight", fisher_w_default))
        mass_w   = float(params.get("mass_weight", params.get("model_mass", mass_w_default)))
        cpl_w    = float(params.get("phi_coupling_weight", cpl_w_default))

        which = "p"
        gen_p = generators
        gen_q = getattr(agent, "generators_q", None)
    else:
        raise ValueError(f"Invalid field: {field}")

    # quick exit
    if (curv_w == 0.0) and (fisher_w == 0.0) and (mass_w == 0.0) and (cpl_w == 0.0):
        return grad_nat

    # ---------------- mask over support ----------------
    eps_sup = float(getattr(config, "support_cutoff_eps", 1e-6))
    # mask has shape (...,) or (...,1) depending on upstream; standardize to (...,1)
    mask = (np.asarray(agent.mask) > eps_sup).astype(np.float32)
    if mask.ndim == grad_nat.ndim - 1:
        mask = mask[..., None]                     # (...,1)
    elif mask.ndim == grad_nat.ndim and mask.shape[-1] == 1:
        pass                                       # already (...,1)
    else:
        # force to (...,1)
        mask = np.asarray(mask, np.float32)
        mask = mask.reshape(grad_nat.shape[:-1] + (1,))
    mask_vec = mask                                # for vector-valued terms
    mask_sca = mask[..., 0]                        # for scalar-valued terms (if needed)

    # ---------------- Jacobians for CURRENT field ----------------
    Jinv = Jinv_grid(ctx, agent, which=which) if ctx is not None else None
    if Jinv is None:
        Jinv = build_dexpinv_matrix(np.asarray(phi_field, np.float32))

    def cov_param_to_body(g_param):
        """ map parameter covector (...,d) to body vector via Jinv, uncapped """
        g_param = np.asarray(g_param, np.float32)
        if g_param.ndim == grad_nat.ndim - 1:
            g_param = g_param[..., None] * np.ones((d,), dtype=np.float32)
        return np.einsum("...ji,...j->...i", Jinv, g_param, optimize=True)

    def cov_param_to_body_mass(g_param):
        """ same as above (kept separate in case of future capping) """
        g_param = np.asarray(g_param, np.float32)
        if g_param.ndim == grad_nat.ndim - 1:
            g_param = g_param[..., None] * np.ones((d,), dtype=np.float32)
        return np.einsum("...ji,...j->...i", Jinv, g_param, optimize=True)

    def _as_vec(x):
        """Coerce x to (..., d) for safe vector addition"""
        x = np.asarray(x, np.float32)
        if x.ndim == grad_nat.ndim - 1:
            return x[..., None]                    # (...,1)
        if x.ndim == grad_nat.ndim and x.shape[-1] in (1, d):
            return x
        # unknown shape → ignore by returning zeros_like grad_nat
        return np.zeros_like(grad_nat, dtype=np.float32)

    # ---------------- Curvature term (param covector → body) ----------------
    if curv_w > 0.0:
        g_param_curv = compute_curvature_gradient_analytical(
            np.asarray(phi_field, np.float32), generators
        )
        g_curv_body = cov_param_to_body(g_param_curv)            # (...,d)
        grad_nat += curv_w * g_curv_body * mask_vec              # (...,d)

    # ---------------- Fisher-geometry term (already NATURAL/body) ----------------
    if fisher_w > 0.0 and agents is not None:
        step_tag = params.get("step_tag", getattr(ctx, "global_step", None) if ctx is not None else None)
        g_body = compute_fisher_geometry_gradient(
            agent, agents, params, field=field, step_tag=step_tag, ctx=ctx
        )
        g_body = _as_vec(g_body)                                 # coerce to (...,1) or (...,d)
        # if it came back (...,1), broadcast on add:
        if g_body.shape[-1] == 1:
            g_body = np.broadcast_to(g_body, grad_nat.shape)
        grad_nat += fisher_w * g_body * mask_vec

    # ---------------- Mass term: L = (λ/2)||φ||^2 → ∂L/∂φ = λ φ ----------------
    if mass_w > 0.0:
        g_mass_body = cov_param_to_body_mass(mass_w * np.asarray(phi_field, np.float32))
        grad_nat += g_mass_body * mask_vec

    # ---------------- q↔p coupling term ----------------
    if cpl_w > 0.0:
        try:
            phi_q = np.asarray(getattr(agent, "phi", None), np.float32)
            phi_p = np.asarray(getattr(agent, "phi_model", None), np.float32)
            C = _build_c_mapping_q2p(getattr(agent, "Phi_tilde_0", None), gen_q, gen_p, eps=eps)
            if C is not None and phi_q is not None and phi_p is not None:
                r = phi_p - np.einsum("ab,...b->...a", C, phi_q, optimize=True)   # (...,d_p)
                if field == "phi":
                    # ∂L/∂φ_q = - C^T r  (param covector)
                    g_param = -np.einsum("ba,...a->...b", C, r, optimize=True)    # (...,d_q)
                else:
                    # ∂L/∂φ_p = r
                    g_param = r
                g_cpl_body = cov_param_to_body(g_param)
                grad_nat += cpl_w * g_cpl_body * mask_vec
        except Exception as e:
            if params.get("DEBUG_PHI", False):
                print(f"[DBGALIGN-WARN] reg coupling failed: {type(e).__name__}: {e}", flush=True)

    return np.asarray(grad_nat, np.float32)




# =========================
# Helpers (cached, lightweight)
# =========================

def _algebra_matrix_from_coeffs(phi_coeffs, generators):
    """
    phi_coeffs: (..., d)
    generators: (d, K, K)
    returns: (..., K, K) matrix in the algebra
    """
    # sum_a phi^a G^a
    # reshape phi to broadcast over K,K
    return np.einsum("...a,aij->...ij", phi_coeffs, generators)


def _gram_inv_cached(agent, key_name, generators, eps=1e-8):
    """
    Cache and return Gram^{-1} for the given generator set under Frobenius inner product.
    key_name: str attribute name on agent (e.g., "_Gram_q_inv" / "_Gram_p_inv")
    """
    G_inv = getattr(agent, key_name, None)
    if G_inv is not None:
        return G_inv
    d = generators.shape[0]
    Gram = np.empty((d, d), dtype=generators.dtype)
    for a in range(d):
        for b in range(d):
            Gram[a, b] = np.tensordot(generators[a], generators[b])
    Gram += eps * np.eye(d, dtype=Gram.dtype)
    G_inv = safe_inv(Gram)
    setattr(agent, key_name, G_inv)
    return G_inv


def _project_mat_to_algebra_coeffs(M, generators, G_inv):
    """
    Project matrix M (..., K, K) to algebra coefficients (..., d)
    by solving coeffs = G_inv @ h, where h_b = <G^b, M>_F.
    """
    d = generators.shape[0]
    # h_b = tr((G^b)^T M)
    h = np.stack([np.tensordot(generators[b], M) for b in range(d)], axis=-1)
    # coeffs = h @ G_inv^T (since last dim is basis index)
    # both ways are fine; using right-multiply for (..., d)
    coeffs = np.einsum("...b,bc->...c", h, G_inv)
    return coeffs


# --- helper: build C once and cache on agent ---
def _build_c_mapping_q2p(Phi_tilde_0, generators_q, generators_p, eps=1e-8):
    """
    Build linear map C (d_p x d_q) so that Ad_{Phi_tilde_0}(G_q^a) ≈ sum_b C_{ba} G_p^b,
    using Frobenius inner-product projection onto p-basis.
    """
    d_q, Kq, _ = generators_q.shape
    d_p, Kp, _ = generators_p.shape
    assert Phi_tilde_0.shape[-2:] == (Kq, Kp), "Phi_tilde_0 must be (K_q, K_p)"

    # Gram matrix of p-basis under Frobenius product
    Gp = np.zeros((d_p, d_p), dtype=generators_p.dtype)
    for b in range(d_p):
        for c in range(d_p):
            Gp[b, c] = np.tensordot(generators_p[b], generators_p[c])
    Gp += eps * np.eye(d_p, dtype=Gp.dtype)
    Gp_inv = safe_inv(Gp)

    # Conjugate each q-generator: H_a = Phi0 G_q^a Phi0^{-1}
    Phi0    = Phi_tilde_0
    Phi0_inv = safe_inv(Phi0, eps=eps)

    C = np.zeros((d_p, d_q), dtype=generators_p.dtype)
    for a in range(d_q):
        H_a = np.einsum("ik,kl,lj->ij", Phi0, generators_q[a], Phi0_inv)
        h = np.array([np.tensordot(generators_p[b], H_a) for b in range(d_p)], dtype=H_a.dtype)
        C[:, a] = Gp_inv @ h
    return C


