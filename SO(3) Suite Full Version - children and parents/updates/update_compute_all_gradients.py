from __future__ import annotations

"""
Created on Tue Aug 12 20:15:47 2025

@author: chris and christine
"""
import time
from joblib import Parallel, delayed
import numpy as np
import core.config as config
from core.omega import d_exp_phi_exact, d_exp_phi_tilde_exact

from core.numerical_utils import safe_omega_inv    
from transport.bundle_morphism_utils import mark_morphism_dirty
from core.omega import exp_lie_algebra_irrep
from transport.transport_cache import E_grid  
from updates.update_terms_phi import ( 
    accumulate_phi_morphism_self_feedback, accumulate_phi_alignment_gradient,
    grad_feedback_phi_direct, grad_feedback_phi_tilde_direct, grad_self_phi_direct,
    grad_self_phi_tilde_direct)

from updates.update_terms_mu_sigma import accumulate_mu_sigma_gradient
from transport.preprocess_utils import zero_all_gradients
from core.omega import retract_phi_principal
from core.numerical_utils import sanitize_sigma
import os
from contextlib import contextmanager
from updates.update_terms_xscale import accumulate_crossscale_phi_env, compute_crossscale_mu_sigma_env
from updates.update_refresh_utils import mark_dirty



@contextmanager
def _blas_single_threads(enabled: bool):
    if not enabled:
        yield
        return
    keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    old = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys: os.environ[k] = "1"
        yield
    finally:
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v


def _coerce_mask_shape(mask_src, shape_dst):
    """
    Make mask_src match shape_dst (H,W), using:
      1) exact reshape if sizes match,
      2) nearest-neighbor resample if not.
    Returns float32 mask in [0,1].
    """
    
    m = np.asarray(mask_src, dtype=np.float32)
    Ht, Wt = shape_dst
    if m.shape == (Ht, Wt):
        return m

    # reshape if number of elements matches
    if m.size == Ht * Wt:
        return m.reshape(Ht, Wt)

    # nearest neighbor resample (no deps)
    Hs, Ws = m.shape[:2]
    if Hs <= 0 or Ws <= 0:
        return np.zeros((Ht, Wt), np.float32)
    yi = (np.arange(Ht) * (Hs / float(Ht))).astype(np.int64)
    xi = (np.arange(Wt) * (Ws / float(Wt))).astype(np.int64)
    yi = np.clip(yi, 0, Hs - 1)
    xi = np.clip(xi, 0, Ws - 1)
    return m[yi[:, None], xi[None, :]].astype(np.float32)






# unified metadata for phi vs phi_tilde blocks
_FIELD_META = {
    "phi": {
        "which": "q",
        "beta_key": "beta",
        "mu_key": "mu_q_field",
        "sigma_key": "sigma_q_field",
        "fisher_inv_key": "inverse_fisher_phi",
        "grad_key": "grad_phi",
        "phi_attr": "phi",
        "qall_func": d_exp_phi_exact,
    },
    "phi_model": {
        "which": "p",
        "beta_key": "beta_model",
        "mu_key": "mu_p_field",
        "sigma_key": "sigma_p_field",
        "fisher_inv_key": "inverse_fisher_phi_model",
        "grad_key": "grad_phi_tilde",
        "phi_attr": "phi_model",
        "qall_func": d_exp_phi_tilde_exact,
    },
}

def _alignment_block(agent_i, neighbors, generators, params, *, field: str, ctx):
    """
    Compute same-level alignment gradients for 'phi' or 'phi_model'.
    """
    if not neighbors:
        return 0.0

    meta = _FIELD_META[field]
    t0 = time.perf_counter()

    # Q_all for agent_i once
    phi_i = getattr(agent_i, meta["phi_attr"])
    Q_all_i = meta["qall_func"](phi_i, generators)

    # loop neighbors
    for agent_j in neighbors:

        # Build exp(-φ_j) from central cache if available, else local
        if ctx is not None:
            Ej = E_grid(ctx, agent_j, which=meta["which"])
            exp_neg_phi_j = safe_omega_inv(Ej, eps=getattr(config, "eps", 1e-6), debug=False)
        else:
            phi_j = getattr(agent_j, meta["phi_attr"])
            Ej = exp_lie_algebra_irrep(phi_j, generators)
            exp_neg_phi_j = safe_omega_inv(Ej, eps=getattr(config, "eps", 1e-6), debug=False)

        # accumulate alignment grad
        accumulate_phi_alignment_gradient(
            agent_i, agent_j, generators, params,
            field=field,
            beta_key=meta["beta_key"],
            omega_cache_key=None,                   # ctx handles caching centrally
            mu_key=meta["mu_key"],
            sigma_key=meta["sigma_key"],
            fisher_inv_key=meta["fisher_inv_key"],
            grad_key=meta["grad_key"],
            Q_all_i=Q_all_i,
            exp_neg_phi_j=exp_neg_phi_j,
            agents=None,                            # pass if your impl needs it; else None is fine
        )
        
    return time.perf_counter() - t0

def _morphism_block(agent_i, generators_q, generators_p, params, *, field: str):
    meta = _FIELD_META[field]
    t0 = time.perf_counter()

    if field == "phi":
        accumulate_phi_morphism_self_feedback(
            agent_i, generators_p, generators_q, params,
            field="phi",
            fisher_inv_key=meta["fisher_inv_key"],
            grad_key=meta["grad_key"],
            phi_self_func=grad_self_phi_direct,
            phi_feedback_func=grad_feedback_phi_direct,
        )
    else:
        accumulate_phi_morphism_self_feedback(
            agent_i, generators_p, generators_q, params,
            field="phi_model",
            fisher_inv_key=meta["fisher_inv_key"],
            grad_key=meta["grad_key"],
            phi_self_func=grad_self_phi_tilde_direct,
            phi_feedback_func=grad_feedback_phi_tilde_direct,
        )

    return time.perf_counter() - t0


# -----------------------------------
# Disentangled gradient orchestrator
# -----------------------------------
def compute_all_gradients(agent_i, agents, all_agents, generators_q, generators_p, params, runtime_ctx=None):
    """
    Gradient orchestrator with a single cross-scale switch.

      Phase 0: thread runtime ctx
      Phase 1: μ, Σ (belief/model) — same-level only (lvl-0 math)
      Phase 2: warm caches for (agent_i + neighbors)
      Phase 3: φ alignment (belief + model), same-level only
      Phase 4: ALL cross-scale (μ/Σ + φ/φ̃) if enable_xscale=True
      Phase 5: morphism self + feedback (φ, φ̃)
      Phase 6: return gradients
    """
    
    # helper to add in-place (handles None/0)
    def _accum_add_inplace(dst, inc):
        import numpy as np
        if inc is None:
            return dst
        if dst is None or (isinstance(dst, (int, float)) and dst == 0):
            return np.asarray(inc, np.float32)
        return (np.asarray(dst, np.float32) + np.asarray(inc, np.float32)).astype(np.float32, copy=False)

    # Phase 0 — ctx
    params, ctx = _thread_ctx(params, runtime_ctx)

    # single switch for ALL cross-scale grads
    enable_xscale = bool(params.get("enable_xscale", getattr(config, "enable_xscale", False)))

    # reset grads, timers
    zero_all_gradients(agent_i)
    timings = {}

    # Phase 1 — μ, Σ (same-level only)
    t0 = time.perf_counter()
    accumulate_mu_sigma_gradient(agent_i, agents, params, mode="belief")
    timings["mu_sigma_belief"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    accumulate_mu_sigma_gradient(agent_i, agents, params, mode="model")
    timings["mu_sigma_model"] = time.perf_counter() - t0

    # Phase 2 — warm caches for neighbors
    neighbors = _get_neighbors(agent_i, agents)
 
    
    # Phase 3 — φ alignment (same-level only)
    if neighbors:
        dt = _alignment_block(agent_i, neighbors, generators_q, params, field="phi", ctx=ctx)
        timings["phi_alignment"] = dt

        dt = _alignment_block(agent_i, neighbors, generators_p, params, field="phi_model", ctx=ctx)
        timings["phi_tilde_alignment"] = dt
        

    # Phase 4 — ALL cross-scale under one switch
    if enable_xscale:
        try:
            
            t0 = time.perf_counter()

            # μ/Σ env (belief)
            dmu_env_q, dS_env_q = compute_crossscale_mu_sigma_env(
                agent_i, ctx=ctx, params=params, mode="belief"
            )
            agent_i.grad_mu_q    = _accum_add_inplace(agent_i.grad_mu_q,    dmu_env_q)
            agent_i.grad_sigma_q = _accum_add_inplace(agent_i.grad_sigma_q, dS_env_q)

            # μ/Σ env (model)
            dmu_env_p, dS_env_p = compute_crossscale_mu_sigma_env(
                agent_i, ctx=ctx, params=params, mode="model"
            )
            agent_i.grad_mu_p    = _accum_add_inplace(agent_i.grad_mu_p,    dmu_env_p)
            agent_i.grad_sigma_p = _accum_add_inplace(agent_i.grad_sigma_p, dS_env_p)

            # φ/φ̃ env
            accumulate_crossscale_phi_env(
                agent_i,
                generators_q=generators_q,
                generators_p=generators_p,
                params=params,
            )

            timings["xscale_all"] = time.perf_counter() - t0

        except Exception as e:
            if getattr(config, "debug_grad_timing", False):
                print(f"[xscale] disabled due to import/exec error: {e}")

    # Phase 5 — morphism self + feedback (same-level)
    dt = _morphism_block(agent_i, generators_q, generators_p, params, field="phi")
    timings["phi_morphism"] = dt

    dt = _morphism_block(agent_i, generators_q, generators_p, params, field="phi_model")
    timings["phi_tilde_morphism"] = dt

    if getattr(config, "debug_grad_timing", False):
        print(f"\n[GRAD TIMINGS] Agent {agent_i.id}")
        for k, v in timings.items():
            print(f"  {k:22s} : {v*1e3:7.2f} ms")

    # Phase 6 — return
    return (
        (agent_i.grad_mu_q, agent_i.grad_sigma_q),
        (agent_i.grad_mu_p, agent_i.grad_sigma_p),
        agent_i.grad_phi,
        agent_i.grad_phi_tilde,
        agent_i.grad_Phi,
        agent_i.grad_Phi_tilde,
    )


def apply_phi_updates(agent, grad_phi, grad_phi_m, mask, params, *, ctx=None):
    """
    Constant-step φ / φ̃ update (no adaptive eta).
    - Uses tau_phi / tau_phi_model as fixed learning rates.
    - Per-pixel vector-norm clipping (gradient_clip_phi / _phi_model).
    - Soft mask at edges (optional).
    - Stores clipped grads into agent.grad_phi / grad_phi_tilde for logging.
    """
        
    mark_dirty = lambda *_args, **_kw: None
    # --- capture raw grads for debugging/diags (as given)
    agent.grad_phi       = None if grad_phi   is None else np.array(grad_phi,   copy=True)
    agent.grad_phi_tilde = None if grad_phi_m is None else np.array(grad_phi_m, copy=True)

    # --- soft mask (optional)
    def _soft_mask(m):
        m = np.clip(np.asarray(m, np.float32), 0.0, 1.0)
        sigma = float(getattr(config, "mask_edge_soften_sigma", 0.0))
        if sigma <= 0.0:
            return m
        try:
            from scipy.ndimage import gaussian_filter
            return np.clip(gaussian_filter(m, sigma=sigma, mode="wrap"), 0.0, 1.0)
        except Exception:
            # lightweight 4-neighbor smoothing fallback
            out = m.copy()
            for _ in range(3):
                nb = (np.roll(out, 1, 0) + np.roll(out, -1, 0) +
                      np.roll(out, 1, 1) + np.roll(out, -1, 1)) * 0.25
                out = 0.5 * out + 0.5 * nb
            return np.clip(out, 0.0, 1.0)

    mexp = _soft_mask(mask)[..., None].astype(np.float32)  # (H,W,1)

    # --- fixed learning rates
    eta_phi    = float(params.get("tau_phi",        getattr(config, "tau_phi",        1e-3)))
    eta_phi_m  = float(params.get("tau_phi_model",  getattr(config, "tau_phi_model",  1e-3)))

    # --- per-pixel grad-norm clipping
    clip_phi   = float(getattr(config, "gradient_clip_phi",       1.0))
    clip_phi_m = float(getattr(config, "gradient_clip_phi_model", 1.0))

    def _clip_vecnorm(v, thr, like):
        if v is None:
            return np.zeros_like(like, dtype=np.float32)
        v = np.asarray(v, np.float32)
        if thr <= 0.0:
            return v
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        over = n > thr
        if np.any(over):
            v = v * (thr / np.maximum(n, 1e-8)).astype(np.float32)
        return v

    # --- clip to thresholds (and coerce None → zeros_like current fields)
    g_phi  = _clip_vecnorm(grad_phi,   clip_phi,   getattr(agent, "phi"))
    g_phim = _clip_vecnorm(grad_phi_m, clip_phi_m, getattr(agent, "phi_model"))
   
    
    # overwrite diag snapshots with the clipped versions (what we actually apply)
    agent.grad_phi       = np.array(g_phi,  copy=True)
    agent.grad_phi_tilde = np.array(g_phim, copy=True)

    # --- apply fixed steps (mask-aware)
    delta_phi   = np.nan_to_num(eta_phi   * g_phi,  nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    delta_phi_m = np.nan_to_num(eta_phi_m * g_phim, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    phi_new  = (np.asarray(agent.phi,       np.float32) - delta_phi   * mexp)
    phim_new = (np.asarray(agent.phi_model, np.float32) - delta_phi_m * mexp)

    # --- retract to principal SO(3) chart
    margin = float(getattr(config, "so3_principal_margin", 1e-4))
    agent.phi       = retract_phi_principal(phi_new,  margin=margin)
    agent.phi_model = retract_phi_principal(phim_new, margin=margin)

    # --- notify runtime that transports/metrics are dirty; no direct warming here
    if ctx is not None:
        # kl=True so KL fields get refreshed on next ensure_dirty/refresh pass
        mark_dirty(ctx, agent, kl=True)
    # keep these set so any legacy logging that prints "eta" won't break
    agent._eta_phi_prev   = eta_phi
    agent._eta_phi_m_prev = eta_phi_m




def _apply_children(agents, grads, params, frozen_levels, frozen_phi_levels, Gq, Gp, n_jobs, *, ctx=None):
    
    
    tau_mu_belief    = params.get("tau_mu_belief",    config.tau_mu_belief)
    tau_sigma_belief = params.get("tau_sigma_belief", config.tau_sigma_belief)
    tau_mu_model     = params.get("tau_mu_model",     config.tau_mu_model)
    tau_sigma_model  = params.get("tau_sigma_model",  config.tau_sigma_model)

    (q_updates, p_updates, phi_grads, phi_m_grads, _, _) = grads
    masks = [A.mask for A in agents]
    eps = float(getattr(config, "eps", 1e-8))

    def _apply_sigma_masked(old_S, dS, tau, mask_bool):
        # update only on masked region; symmetrize+sanitize only there to avoid drifting unmasked entries
        mask3 = mask_bool[..., None, None]
        S_new = old_S + tau * dS * mask3
        S_new = _safe_symmetrize(S_new)
        S_new_san = sanitize_sigma(S_new, eps=eps)
        return np.where(mask3, S_new_san, old_S)

    def _one(i):
        A = agents[i]
        level = getattr(A, "level", 0)
        mask = np.asarray(masks[i], np.float32)
        mask_mu    = mask[..., None]
        mask_sigma = mask[..., None, None]

        dmu_q, dsg_q = q_updates[i]
        dmu_p, dsg_p = p_updates[i]
        if dmu_q is None: dmu_q = np.zeros_like(A.mu_q_field,  dtype=np.float32)
        if dsg_q is None: dsg_q = np.zeros_like(A.sigma_q_field,dtype=np.float32)
        if dmu_p is None: dmu_p = np.zeros_like(A.mu_p_field,  dtype=np.float32)
        if dsg_p is None: dsg_p = np.zeros_like(A.sigma_p_field,dtype=np.float32)

        did_phi = False

        # --- field updates (μ, Σ) ---
        if level not in frozen_levels:
            A.mu_q_field = A.mu_q_field + (tau_mu_belief * dmu_q * mask_mu).astype(np.float32, copy=False)
            A.mu_p_field = A.mu_p_field + (tau_mu_model  * dmu_p * mask_mu).astype(np.float32, copy=False)

            A.sigma_q_field = _apply_sigma_masked(A.sigma_q_field, dsg_q, tau_sigma_belief, mask > 0.0)
            A.sigma_p_field = _apply_sigma_masked(A.sigma_p_field, dsg_p, tau_sigma_model,  mask > 0.0)

        # --- φ updates ---
        if level not in frozen_phi_levels:
            gp  = phi_grads[i];   gp  = np.zeros_like(A.phi,       dtype=np.float32) if gp  is None else gp
            gpm = phi_m_grads[i]; gpm = np.zeros_like(A.phi_model, dtype=np.float32) if gpm is None else gpm
            apply_phi_updates(A, gp, gpm, mask, params, ctx=ctx)
            did_phi = True

        # mark morphism stale only if φ changed
        if did_phi:
            mark_morphism_dirty(A)

        return did_phi

    did_flags = Parallel(n_jobs=n_jobs, backend="threading", batch_size="auto")(
        delayed(_one)(i) for i in range(len(agents))
    )



def _apply_parent_updates(parents, step, params, *, ctx=None):
    """
    Apply parent μ/Σ and φ/φ̃ updates with ramping; invalidate central caches
    when gauge fields change; and mark objects dirty for downstream rebuilds.
    """
    freeze = int(params.get("parent_freeze_steps", 0))
    ramp   = max(1, int(params.get("parent_ramp_steps", 1)))
    if step < freeze:
        return
    s = min(1.0, (step - freeze + 1) / ramp)

    tau_mu_belief    = params.get("tau_mu_belief",    0.0)
    tau_sigma_belief = params.get("tau_sigma_belief", 0.0)
    tau_mu_model     = params.get("tau_mu_model",     0.0)
    tau_sigma_model  = params.get("tau_sigma_model",  0.0)

    for P in parents:
        pmask = np.asarray(getattr(P, "mask", 0.0), dtype=np.float32)
        mask_mu    = pmask[..., None]
        mask_sigma = pmask[..., None, None]

        # q-bundle
        if getattr(P, "grad_mu_q", None) is not None:
            P.mu_q_field   = P.mu_q_field   + s * tau_mu_belief * P.grad_mu_q * mask_mu
        if getattr(P, "grad_sigma_q", None) is not None:
            P.sigma_q_field = P.sigma_q_field + s * tau_sigma_belief * P.grad_sigma_q * mask_sigma

        # p-bundle
        if getattr(P, "grad_mu_p", None) is not None:
            P.mu_p_field   = P.mu_p_field   + s * tau_mu_model * P.grad_mu_p * mask_mu
        if getattr(P, "grad_sigma_p", None) is not None:
            P.sigma_p_field = P.sigma_p_field + s * tau_sigma_model * P.grad_sigma_p * mask_sigma

        # φ / φ̃ updates — apply and let apply_phi_updates handle per-agent invalidation
        if getattr(P, "grad_phi", None) is not None or getattr(P, "grad_phi_tilde", None) is not None:
            apply_phi_updates(
                P,
                getattr(P, "grad_phi", None),
                getattr(P, "grad_phi_tilde", None),
                pmask,
                params,
                ctx=ctx,  
            )

            # mark dirty so downstream phases rebuild Λ / Φ
            try:
                
                if ctx is not None:
                    mark_dirty(ctx, P, kl=True)
            except Exception:
                setattr(P, "morphisms_dirty", True)




def _parent_intrascale_grad_pass(agents, parents, Gq, Gp, params, accum_into=True, *, ctx=None):
    if not parents:
        return

    # --- params + ctx
    params_par = dict(params or {})
    params_par["enable_crossscale"] = False
    if ctx is not None:
        params_par["runtime_ctx"] = ctx

    all_agents_par = agents + parents
    parents_by_id = {int(getattr(p, "id", i)): p for i, p in enumerate(parents)}

    
    # --- choose backend / workers sensibly
    backend = str(getattr(config, "parent_backend", "loky")).lower()  # {"loky","multiprocessing","threading"}
    prefer  = "threads" if backend == "threading" else "processes"
    n_jobs  = int(getattr(config, "n_jobs", 1))
    n_jobs  = max(1, min(n_jobs, len(parents)))  # don't spawn more workers than tasks

    def _safe_add(a, b):
        if a is None: return b
        if b is None: return a
        return a + b

    # --- run
    # Pin BLAS threads only if we use a process backend to avoid oversubscription
    with _blas_single_threads(enabled=(backend != "threading")):
        results = Parallel(
            n_jobs=n_jobs,
            backend=backend,
            prefer=prefer,
            batch_size="auto",
        )(
            delayed(compute_all_gradients)(
                P, parents_by_id, all_agents_par, Gq, Gp, params_par, runtime_ctx=ctx
            )
            for P in parents
        )

    # --- reduce
    for P, res in zip(parents, results):
        if res is None:
            continue
        (dmuq, dsgq), (dmup, dsgp), gphi, gphit, *_ = res

        # ensure grad slots exist
        if not hasattr(P, "grad_mu_q"):    P.grad_mu_q = None
        if not hasattr(P, "grad_sigma_q"): P.grad_sigma_q = None
        if not hasattr(P, "grad_mu_p"):    P.grad_mu_p = None
        if not hasattr(P, "grad_sigma_p"): P.grad_sigma_p = None
        if not hasattr(P, "grad_phi") or P.grad_phi is None:
            P.grad_phi = (np.zeros_like(P.phi, dtype=np.float32)
                          if getattr(P, "phi", None) is not None else None)
        if not hasattr(P, "grad_phi_tilde") or P.grad_phi_tilde is None:
            P.grad_phi_tilde = (np.zeros_like(P.phi_model, dtype=np.float32)
                                if getattr(P, "phi_model", None) is not None else None)

        if accum_into:
            P.grad_mu_q    = _safe_add(P.grad_mu_q,    dmuq)
            P.grad_sigma_q = _safe_add(P.grad_sigma_q, dsgq)
            P.grad_mu_p    = _safe_add(P.grad_mu_p,    dmup)
            P.grad_sigma_p = _safe_add(P.grad_sigma_p, dsgp)
            if gphi  is not None and P.grad_phi is not None:
                P.grad_phi       = P.grad_phi       + gphi
            if gphit is not None and P.grad_phi_tilde is not None:
                P.grad_phi_tilde = P.grad_phi_tilde + gphit
        else:
            P.grad_mu_q,  P.grad_sigma_q  = dmuq, dsgq
            P.grad_mu_p,  P.grad_sigma_p  = dmup, dsgp
            if gphi is not None:
                if P.grad_phi is None and getattr(P, "phi", None) is not None:
                    P.grad_phi = np.zeros_like(P.phi, dtype=np.float32)
                if P.grad_phi is not None:
                    P.grad_phi[...] = gphi
            if gphit is not None:
                if P.grad_phi_tilde is None and getattr(P, "phi_model", None) is not None:
                    P.grad_phi_tilde = np.zeros_like(P.phi_model, dtype=np.float32)
                if P.grad_phi_tilde is not None:
                    P.grad_phi_tilde[...] = gphit


def _compute_child_grads(agents, all_agents, Gq, Gp, params):
    if not agents:
        # keep shape expectations: return empty tuples of lists
        return ([], [], [], [], [], [])

    params_x = dict(params or {})
    ctx = params_x.get("runtime_ctx", None)   
    n_jobs = max(1, min(int(getattr(config, "n_jobs", 1)), len(agents)))

    def _safe_compute(Ai):
        try:
            return compute_all_gradients(
                Ai, agents, all_agents, Gq, Gp, params_x, runtime_ctx=ctx
            )
        except Exception:
            # fall back to zeros of proper shapes to avoid crashing the batch
            Kq = Ai.sigma_q_field.shape[-1]
            Kp = Ai.sigma_p_field.shape[-1]
            H, W = Ai.mask.shape[:2]
            zeros_mu_q = np.zeros((H, W, Kq), dtype=np.float32)
            zeros_mu_p = np.zeros((H, W, Kp), dtype=np.float32)
            zeros_Sq   = np.zeros((H, W, Kq, Kq), dtype=np.float32)
            zeros_Sp   = np.zeros((H, W, Kp, Kp), dtype=np.float32)
            zeros_phi  = np.zeros_like(Ai.phi,       dtype=np.float32)
            zeros_phim = np.zeros_like(Ai.phi_model, dtype=np.float32)
            print("safe-compute exception compute all grads-child grads FAIL \n FAIL \n FAIL \n\n\n ")
            return ((zeros_mu_q, zeros_Sq),
                    (zeros_mu_p, zeros_Sp),
                    zeros_phi,
                    zeros_phim,
                    None,
                    None)

    results = Parallel(
        n_jobs=n_jobs,
        backend="threading",
        prefer="threads",
        batch_size="auto",
    )([delayed(_safe_compute)(Ai) for Ai in agents])

    # unzip and convert each to lists for consistent downstream handling
    q_updates, p_updates, phi_grads, phi_m_grads, morphism_grads, morphism_tilde_grads = zip(*results)
    return (list(q_updates), list(p_updates), list(phi_grads), list(phi_m_grads),
            list(morphism_grads), list(morphism_tilde_grads))





def _safe_symmetrize(S):
    return 0.5 * (S + np.swapaxes(S, -1, -2))


# --- robust parent collection -----------------------------------------------

def _collect_parents_from_ctx(ctx, *, level: int = 1):
    """
    Prefer the level-aware registry on ctx; fall back to legacy mirrors
    only if needed. Returns a de-duplicated, order-preserving list.
    """
    parents = []

    # 1) Preferred: level-aware registry
    try:
        reg, _ = ctx.get_parent_registry(level)
        if isinstance(reg, dict):
            parents.extend([v for v in reg.values() if v is not None])
        elif reg is not None:
            parents.append(reg)
    except Exception:
        print("collect-parents-from-ctx failure\n\n")

    # 2) Legacy fallbacks (only if nothing collected)
    if not parents:
        pbr = getattr(ctx, "parents_by_region", None)
        if isinstance(pbr, dict):
            parents.extend([v for v in pbr.values() if v is not None])
        elif pbr is not None:
            parents.append(pbr)
        else:
            for name in ("parents", "parent"):
                obj = getattr(ctx, name, None)
                if obj is None:
                    continue
                if isinstance(obj, (list, tuple)):
                    parents.extend(obj)
                else:
                    parents.append(obj)

    # De-dup, preserve order
    seen = set()
    flat = []
    for p in parents:
        if p is None:
            continue
        try:
            pid = int(getattr(p, "id"))
        except Exception:
            pid = id(p)
        if pid not in seen:
            seen.add(pid)
            flat.append(p)
    return flat


def _final_apply_parents(ctx, params):
    parents = _collect_parents_from_ctx(ctx)
    if not parents:
        return
    _apply_parent_updates(parents, getattr(ctx, "global_step", 0), params, ctx=ctx)



# ---------------------------
# Local helpers (file-local)
# ---------------------------

def _thread_ctx(params, runtime_ctx):
    params = {} if params is None else dict(params)
    if runtime_ctx is not None:
        params["runtime_ctx"] = runtime_ctx
    return params, params.get("runtime_ctx", None)

def _get_neighbors(agent_i, agents):
    nbs = getattr(agent_i, "neighbors", []) or []
    if isinstance(agents, dict):
        lookup = {int(k): v for k, v in agents.items()}
    else:
        lookup = {int(getattr(a, "id", i)): a for i, a in enumerate(agents)}
    out = []
    for nb in nbs:
        try:
            j = int(nb.get("id"))
        except Exception:
            continue
        a = lookup.get(j)
        if a is not None:
            out.append(a)
    return out



