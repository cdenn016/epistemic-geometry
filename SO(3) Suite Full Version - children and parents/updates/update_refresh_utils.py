# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 13:30:00 2025

@author: chris and christine
"""
from __future__ import annotations


from transport.preprocess_utils import preprocess_all_agents
from core.gaussian_core import blended_spawn_metrics_child_vs_parents, directed_kl_weight_after_transport

from transport.preprocess_utils import ensure_transforms
from transport.bundle_morphism_utils import recompute_agent_morphisms
from core.gaussian_core import resolve_support_tau
import numpy as np

from typing import Optional
from transport.transport_cache import warm_E, Lambda_up, Lambda_dn,  warm_Jinv

from core.omega import pre_retract_fields




def refresh_spawn_alignment_caches(
    ctx,
    children,
    agents,
    *,
    G_q=None,
    G_p=None,
    tau_q=None,
    tau_p=None,
    alpha=None,          # 0→model-only, 1→belief-only
    sm_tau=None,         # softmin temperature across neighbors
    mask_tau=None,       # support threshold on masks
    use_cover_weight=True,
    kl_cap=None,         # kept for signature compat; not used
    debug=False,
):
    """
    Build *spawn-only* caches on each child (per pixel):
        spawn_nb_count, spawn_min_kl_q/p, spawn_softmin_kl_q/p,
        spawn_cover_weight (opt), spawn_A_final, kl_field, agreement_field.

    Notes
    -----
    - If parents exist and overlap, uses a *batched parent path* with Ω on overlap bboxes.
    - Else falls back to neighbor (child-child) logic using directed_kl_weight_after_transport.
    - Softmin is computed with temperature `sm_tau`; min-KL is the elementwise minimum.
    """
    
    import core.config as _cfg
    
    # ---- defaults / resolvers ----------------------------------------------------
    
    tau_q    = float(getattr(_cfg, "tau_align_q", 0.45))        if tau_q    is None else float(tau_q)
    tau_p    = float(getattr(_cfg, "tau_align_p", 0.45))        if tau_p    is None else float(tau_p)
    alpha    = float(getattr(_cfg, "blend_qp_alpha", 0.50))     if alpha    is None else float(alpha)
    sm_tau   = float(getattr(_cfg, "spawn_softmin_tau", 0.40))  if sm_tau   is None else float(sm_tau)
    
    mask_tau = resolve_support_tau(ctx, params=None, default=1e-3, name="support_tau")
    tiny     = float(getattr(_cfg, "log_eps", 1e-9))

    # ---- quick id->agent map -----------------------------------------------------
    id2A = {int(getattr(a, "id", -1)): a for a in (agents or []) if a is not None}

    # Helper to select candidate parents (level-based, with a fallback flag)
    def _select_parents_for(child):
        lvl_c = int(getattr(child, "level", 0))
        parents_lvl = [a for a in (agents or []) if a is not None and int(getattr(a, "level", 0)) > lvl_c]
        if parents_lvl:
            return parents_lvl
        # fallback: explicit flag if levels aren’t used
        return [a for a in (agents or []) if a is not None and bool(getattr(a, "is_parent", False))]

    # ---- per-child processing ----------------------------------------------------
    for ch in (children or []):
        if ch is None:
            continue

        # shapes / supports
        cmask = np.asarray(getattr(ch, "mask", 0.0), np.float32)
        H, W = cmask.shape[:2]
        present_i = (cmask > mask_tau)

        # fiber availability + generator guards
        mu_q = getattr(ch, "mu_q_field", None)
        mu_p = getattr(ch, "mu_p_field", None)
        Kq   = int(mu_q.shape[-1]) if mu_q is not None else 0
        Kp   = int(mu_p.shape[-1]) if mu_p is not None else 0
        have_q = (Kq > 0 and G_q is not None)
        have_p = (Kp > 0 and G_p is not None)

        if (Kq > 0 and G_q is None):
            raise ValueError("refresh_spawn_alignment_caches: missing G_q for q-fiber.")
        if (Kp > 0 and G_p is None):
            raise ValueError("refresh_spawn_alignment_caches: missing G_p for p-fiber.")

        # init outputs
        nb_count     = np.zeros((H, W), np.int32)
        min_kl_q     = np.full((H, W), np.inf, np.float32)
        min_kl_p     = np.full((H, W), np.inf, np.float32)
        sumexp_q     = np.zeros((H, W), np.float32)  # Σ exp(-KL_q / sm_tau)
        sumexp_p     = np.zeros((H, W), np.float32)  # Σ exp(-KL_p / sm_tau)
        cover_weight = np.zeros((H, W), np.float32) if use_cover_weight else None

        # ---------- FAST PATH: batched parents if present & overlapping -----------
        parents = _select_parents_for(ch)
        # quick overlap existence test
        has_parent_overlap = False
        if parents:
            cm = present_i
            for p in parents:
                pm = np.asarray(getattr(p, "mask", 0.0), np.float32) > mask_tau
                if np.any(cm & pm):
                    has_parent_overlap = True
                    break

        if parents and has_parent_overlap:
            res = blended_spawn_metrics_child_vs_parents(
                ctx, ch, parents, Gq=G_q, Gp=G_p,
                tau_q=tau_q, tau_p=tau_p, alpha=alpha,
                sm_tau=sm_tau, mask_tau=mask_tau
            )
            # publish
            ch.spawn_nb_count      = res["nb_count"]
            ch.spawn_min_kl_q      = res["min_kl_q"]
            ch.spawn_min_kl_p      = res["min_kl_p"]
            ch.spawn_softmin_kl_q  = res["softmin_q"]
            ch.spawn_softmin_kl_p  = res["softmin_p"]
            ch.spawn_cover_weight  = (res["cover_weight"] if use_cover_weight else None)
            ch.spawn_A_final       = res["A_final"]
            ch.kl_field            = res["KL_blend"]
            ch.agreement_field     = res["A_final"]
            
            continue

        # ---------- FALLBACK: child-child neighbor accumulation (vectorized) ------
        neighbors = getattr(ch, "neighbors", None) or []
        if not neighbors or not np.any(present_i):
            # populate empty caches and continue
            zf = np.zeros((H, W), np.float32)
            zi = np.zeros((H, W), np.int32)
            ch.spawn_nb_count      = zi
            ch.spawn_min_kl_q      = zf + np.inf
            ch.spawn_min_kl_p      = zf + np.inf
            ch.spawn_softmin_kl_q  = zf
            ch.spawn_softmin_kl_p  = zf
            ch.spawn_cover_weight  = zf if use_cover_weight else None
            ch.spawn_A_final       = zf
            ch.kl_field            = zf
            ch.agreement_field     = zf
            continue

        # We keep a Python loop over neighbors (usually small); pixel math is vectorized.
        for nb in neighbors:
            # robust neighbor id extraction (dict or object)
            nid = int(nb.get("id", -1)) if isinstance(nb, dict) else int(getattr(nb, "id", -1))
            j = id2A.get(nid)
            if j is None:
                continue

            jmask = np.asarray(getattr(j, "mask", 0.0), np.float32)
            present_j = (jmask > mask_tau)
            ov = (present_i & present_j)
            if not np.any(ov):
                continue

            iy, ix = np.nonzero(ov)
            # count this neighbor wherever overlap holds
            np.add.at(nb_count, (iy, ix), 1)

            # --- q-fiber -----------------------------------------------------------
            if have_q:
                wq = directed_kl_weight_after_transport(
                    ch, j,
                    getattr(ch, "mu_q_field", None), getattr(ch, "sigma_q_field", None),
                    getattr(j,  "mu_q_field", None), getattr(j,  "sigma_q_field", None),
                    fiber="q", H=H, W=W, K=Kq, ov=ov, tau=tau_q,
                    ctx=ctx, G=G_q,
                )
                # KL from weight field at tau_q: KL = -tau * ln w
                kq = (-tau_q * np.log(np.maximum(wq[iy, ix], tiny))).astype(np.float32)
                np.minimum.at(min_kl_q, (iy, ix), kq)
                # softmin at sm_tau: exp(-KL/sm_tau)
                np.add.at(sumexp_q, (iy, ix), np.exp(-kq / max(sm_tau, tiny)).astype(np.float32))

            # --- p-fiber -----------------------------------------------------------
            if have_p:
                wp = directed_kl_weight_after_transport(
                    ch, j,
                    getattr(ch, "mu_p_field", None), getattr(ch, "sigma_p_field", None),
                    getattr(j,  "mu_p_field", None), getattr(j,  "sigma_p_field", None),
                    fiber="p", H=H, W=W, K=Kp, ov=ov, tau=tau_p,
                    ctx=ctx, G=G_p,
                )
                kp = (-tau_p * np.log(np.maximum(wp[iy, ix], tiny))).astype(np.float32)
                np.minimum.at(min_kl_p, (iy, ix), kp)
                np.add.at(sumexp_p, (iy, ix), np.exp(-kp / max(sm_tau, tiny)).astype(np.float32))

            # --- optional coverage weighting --------------------------------------
            if use_cover_weight:
                np.add.at(cover_weight, (iy, ix), jmask[iy, ix])

        # ---- finalize: softmin, blend, agreement ---------------------------------
        valid = (nb_count > 0)

        softmin_q = np.zeros_like(sumexp_q, np.float32)
        softmin_p = np.zeros_like(sumexp_p, np.float32)
        if have_q:
            softmin_q[valid] = (-sm_tau * np.log(np.maximum(sumexp_q[valid], tiny))).astype(np.float32)
        if have_p:
            softmin_p[valid] = (-sm_tau * np.log(np.maximum(sumexp_p[valid], tiny))).astype(np.float32)

        # Blend q/p according to alpha into a single KL field for spawn logic
        if have_q and have_p:
            alpha_eff = float(alpha)
            tau_blend = alpha_eff * tau_q + (1.0 - alpha_eff) * tau_p
            KL_blend  = (alpha_eff * softmin_q + (1.0 - alpha_eff) * softmin_p).astype(np.float32)
        elif have_q:
            tau_blend = float(tau_q)
            KL_blend  = softmin_q.astype(np.float32)
        elif have_p:
            tau_blend = float(tau_p)
            KL_blend  = softmin_p.astype(np.float32)
        else:
            tau_blend = 1.0
            KL_blend  = np.zeros((H, W), np.float32)

        # Agreement from blended KL
        A_final = np.zeros_like(KL_blend, np.float32)
        if np.any(valid & present_i):
            A_final[valid] = np.exp(-KL_blend[valid] / max(tau_blend, tiny)).astype(np.float32)
        A_final = np.clip(A_final, 0.0, 1.0)

        # mask outside child support; set min-KLs to +inf where invalid
        KL_blend  = np.where(present_i, KL_blend,  0.0).astype(np.float32)
        A_final   = np.where(present_i, A_final,   0.0).astype(np.float32)
        softmin_q = np.where(present_i, softmin_q, 0.0).astype(np.float32)
        softmin_p = np.where(present_i, softmin_p, 0.0).astype(np.float32)

        if have_q:
            min_kl_q = np.where(valid & present_i, min_kl_q, np.inf).astype(np.float32)
        else:
            min_kl_q[:] = np.inf
        if have_p:
            min_kl_p = np.where(valid & present_i, min_kl_p, np.inf).astype(np.float32)
        else:
            min_kl_p[:] = np.inf

        if use_cover_weight:
            cover_weight = np.where(present_i, cover_weight, 0.0).astype(np.float32)

        # final NaN guards (rare but cheap)
        KL_blend  = np.nan_to_num(KL_blend,  copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        A_final   = np.nan_to_num(A_final,   copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        softmin_q = np.nan_to_num(softmin_q, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        softmin_p = np.nan_to_num(softmin_p, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        min_kl_q  = np.nan_to_num(min_kl_q,  copy=False, nan=np.inf, posinf=np.inf)
        min_kl_p  = np.nan_to_num(min_kl_p,  copy=False, nan=np.inf, posinf=np.inf)

        # ---- publish on child -----------------------------------------------------
        ch.spawn_nb_count      = nb_count
        ch.spawn_min_kl_q      = min_kl_q
        ch.spawn_min_kl_p      = min_kl_p
        ch.spawn_softmin_kl_q  = softmin_q
        ch.spawn_softmin_kl_p  = softmin_p
        ch.spawn_cover_weight  = cover_weight if use_cover_weight else None
        ch.spawn_A_final       = A_final
        # convenience aliases used by plots / downstream
        ch.kl_field            = KL_blend
        ch.agreement_field     = A_final

        



def ensure_spawn_alignment_caches(
    children,
    agents,
    runtime_ctx,
    *,
    generators_q=None,
    generators_p=None,
    use_cover_weight=True,
    debug=False,
):
    """
    Minimal spawn-alignment cache builder.

    Responsibility:
      - Select subset of children that actually need refresh.
      - Call refresh_spawn_alignment_caches once for that subset.
      - Record bookkeeping fields.

    NOTE: Pre-warming of E/Jinv grids is handled centrally in `ensure_dirty(...)`.
          This function intentionally does NOT pre-warm anything.
    """
    import numpy as np
    import core.config as _cfg

    if not children:
        return

    # Which children actually need a refresh?
    need = [ch for ch in children if ch is not None and should_refresh_spawn_cache(ch, runtime_ctx)]
    if not need:
        return

    # Build/refresh caches for the subset
    refresh_spawn_alignment_caches(
        runtime_ctx,
        need,
        agents,
        G_q=generators_q,
        G_p=generators_p,
        tau_q=float(getattr(_cfg, "tau_align_q", 0.45)),
        tau_p=float(getattr(_cfg, "tau_align_p", 0.45)),
        alpha=float(getattr(_cfg, "blend_qp_alpha", 0.50)),
        sm_tau=float(getattr(_cfg, "spawn_softmin_tau", 0.40)),
        mask_tau=float(getattr(_cfg, "support_tau", getattr(_cfg, "support_cutoff_eps", 1e-3))),
        use_cover_weight=bool(use_cover_weight),
        debug=bool(debug),
    )

    # Record cache build metadata once
    step_tag = int(getattr(runtime_ctx, "global_step", 0))
    cachever = int(getattr(runtime_ctx, "cache_version", 0))
    for ch in need:
        try:
            ch._spawn_cache_built_at_step = step_tag
            ch._spawn_cache_built_at_cachever = cachever
            ch._mask_prev_for_spawn = (
                np.asarray(getattr(ch, "mask", None), dtype=np.float32).copy()
                if getattr(ch, "mask", None) is not None else None
            )
            setattr(ch, "_dirty_fields", False)
            setattr(ch, "_dirty_kl", False)
        except Exception:
            print("ENSURE SPAWN ALIGNMENT CACHE EXCEPTION")
            pass





def ensure_dirty(ctx, generators_q, generators_p, params, scope="all"):
    """
    Refresh transforms for dirty items; pre-warm central caches; clear flags; bump cache version.
    Policy:
      - Ensure bases/φ-fields are sane (no per-agent Φ/Φ̃ writes).
      - Pre-warm E, Jinv, and Φ/Φ̃ in CacheHub (lazy facades).
      - Bump cache version once if anything changed.
    """
    # local imports only for the helpers we need
    import core.config as CFG
    try:
        from transport.transport_cache import warm_E, warm_Jinv, Phi as tc_Phi
    except Exception:
        # Safe no-ops if transport facade is unavailable in some minimal tests
        warm_E = lambda *_a, **_k: None
        warm_Jinv = lambda *_a, **_k: None
        tc_Phi = lambda *_a, **_k: None

    # ------------- collect dirty targets -------------
    targets = []
    if scope in ("all", "children"):
        targets += take_dirty(ctx, getattr(ctx, "children_latest", []), which="agents")
    if scope in ("all", "parents"):
        targets += take_dirty(ctx, list_parents(ctx), which="agents")
    targets = [t for t in targets if t is not None]
    if not targets:
        return targets

    pre_retract_fields(targets)

    did_any = False

    # 1) Ensure φ/φ̃ transforms + shape-correct Φ/Φ̃ exist (idempotent)
    try:
        method = (params or {}).get("intertwiner_method", "casimir")
        data   = (params or {}).get("data_for_alignment", None)
        ensure_transforms(
            targets,
            generators_q=generators_q,
            generators_p=generators_p,
            ctx=ctx,
            intertwiner_method=method,
            data_for_alignment=data,
        )
        did_any = True
    except Exception as e:
        print(f"[ensure_dirty] ensure_transforms failed: {e}\n\n\n\n\n")


    # 2) Pre-warm central caches (idempotent per step)
    try:
        # Exponential grids + dexp^{-1}
        warm_E(ctx, targets, whiches=("q", "p"))
        warm_Jinv(ctx, targets, whiches=("q", "p"))
        # Morphisms Φ/Φ̃ (per-pixel fields, stored in CacheHub — not on agents)
        for a in targets:
            try:
                _ = tc_Phi(ctx, a, kind="q_to_p")
                _ = tc_Phi(ctx, a, kind="p_to_q")
            except Exception:
                aid = getattr(a, "id", None)
                print(f"[ensure_dirty] Phi prewarm failed for agent id={aid}\n\n\n\n\n\n")
        did_any = True
    except Exception as e:        
        print(f"[ensure_dirty] cache prewarm failed: {e}\n\n\n\n")

    # 3) Clear per-agent morphism-dirty flags (we don’t store Φ/Φ̃ on agents)
    for a in targets:
        if hasattr(a, "morphisms_dirty"):
            a.morphisms_dirty = False

    # 4) Bump cache version so downstream users know something changed
    if did_any:
        try:
            bump_cache(ctx, 1)
        except Exception:
            pass

    return targets












def refresh_kl_for_dirty_children(ctx, universe, eps, Gq=None, Gp=None):
    dirty_kids = list_nonnull(take_dirty(ctx, getattr(ctx, "children_latest", []), which="kl"))
    if not dirty_kids:
        return
    _refresh_child_kl_fields_subset(
        dirty_kids, list_nonnull(universe), eps=eps,
        build_spawn=True, ctx=ctx, Gq=Gq, Gp=Gp
    )

def refresh_kl_for_dirty_parents(ctx, universe, eps, Gq=None, Gp=None):
    try:
        parents = list_parents(ctx)
    except Exception:
        parents = []
    if not parents:
        return
    dirty_par = list_nonnull(take_dirty(ctx, parents, which="kl"))
    if not dirty_par:
        return
    _refresh_child_kl_fields_subset(
        dirty_par, list_nonnull(universe), eps=eps,
        build_spawn=True, ctx=ctx, Gq=Gq, Gp=Gp
    )






def _ensure_dirty_sets(ctx):
    # normalize the dirty container once
    d = getattr(ctx, "dirty", None)
    if d is None:
        class _D: pass
        d = _D()
        d.agents = set()
        d.kl = set()
        setattr(ctx, "dirty", d)
    else:
        if not hasattr(d, "agents"): d.agents = set()
        if not hasattr(d, "kl"):     d.kl = set()
    return d




def mark_dirty(ctx, *agents, kl=False):
    """Mark agents as needing refresh; optionally mark their KL dirty too."""
    d = _ensure_dirty_sets(ctx)
    for a in agents:
        if a is None: 
            continue
        aid = int(getattr(a, "id", id(a)))
        d.agents.add(aid)
        if kl:
            d.kl.add(aid)
        # local flags for fast checks
        setattr(a, "_dirty_fields", True)
        if kl:
            setattr(a, "_dirty_kl", True)



def take_dirty(ctx, all_agents, *, which="agents"):
    """
    Pop and return objects whose ids are marked dirty.
    which ∈ {"agents","kl"}.
    """
    d = _ensure_dirty_sets(ctx)
    ids = d.agents if which == "agents" else d.kl

    # inline _collect_by_ids logic (since it's removed from runtime_context)
    if not ids:
        subset = []
    else:
        wanted = {int(i) for i in ids}
        subset = []
        for a in all_agents:
            try:
                aid = int(getattr(a, "id", id(a)))
                if aid in wanted:
                    subset.append(a)
            except Exception:
                continue

    ids.clear()
    return subset
      



# --- refresh passes -----------------------------------------------------------

def refresh_agents_light(agents, params, Gq, Gp, *, ctx=None):
    """
    Light refresh: sanitize (soft), warm exp grids into ctx.cache, build Λ caches,
    and standardize mask shapes.
    """
    if not agents: 
        return
    pre = dict(params)
    pre["sanitize_strict"] = pre.get("sanitize_strict_pre", False)
    pre["exp_n_jobs"] = 1

    preprocess_all_agents(agents, pre, Gq, Gp, ctx=ctx)
    # mask shape hygiene (once)
   
    H, W = np.asarray(agents[0].mask).shape[:2]
    _standardize_all_masks_once(agents, (H, W))  # keep your existing helper




def _rebuild_dirty_morphisms(objs, Gq, Gp, *, ctx=None):
    """Recompute rectangular, gauge-covariant morphisms for dirty agents."""
    if not objs:
        return
    try:
        from transport.transport_cache import Phi as tc_Phi
    except Exception:
        tc_Phi = None
        
    for a in objs:
        if a is None:
            continue
        need = bool(getattr(a, "morphisms_dirty", False))
        
        if not need:
            continue
        if ctx is not None and tc_Phi is not None:
            try:
                _ = tc_Phi(ctx, a, kind="q_to_p")
                _ = tc_Phi(ctx, a, kind="p_to_q")
            except Exception:
                # best-effort prewarm; facade will rebuild lazily on first use
                print("\n rebuilod-dirty-morphism fail Phin")
                pass
        # Clear flag either way; downstream ensure_dirty will warm as needed.
        if hasattr(a, "morphisms_dirty"):
            a.morphisms_dirty = False



def _refresh_child_kl_fields_subset(
    children_subset,
    all_agents,
    eps,
    build_spawn=False,
    *,
    ctx=None,
    Gq=None,
    Gp=None,
):
    """
    Recompute per-pixel KL/agree caches for a subset of agents.

    - If Gq/Gp are provided, uses them for q/p fibers respectively.
    - If Gq/Gp are None, the spawn cache builder will skip the missing fiber(s)
      (agreement still computed from whichever fiber is available).
    """
    if not children_subset or not build_spawn:
        return

    import core.config as _cfg

    refresh_spawn_alignment_caches(
        ctx,
        children_subset,
        all_agents,
        G_q=Gq,
        G_p=Gp,
        tau_q=float(getattr(_cfg, "tau_align_q", 0.45)),
        tau_p=float(getattr(_cfg, "tau_align_p", 0.45)),
        alpha=float(getattr(_cfg, "blend_qp_alpha", 0.50)),
        sm_tau=float(getattr(_cfg, "spawn_softmin_tau", 0.40)),
        mask_tau=float(getattr(_cfg, "support_tau", getattr(_cfg, "support_cutoff_eps", 1e-3))),
        use_cover_weight=True,
        debug=False,
    )




def should_refresh_spawn_cache(child, runtime_ctx, *, mask_tol=None) -> bool:
    """
    Decide whether to rebuild a child's spawn/agree caches.

    Heuristics:
      - no cache built yet
      - fields or KL marked dirty
      - mask changed beyond `mask_tol`
      - runtime_ctx.cache_version advanced since last build
    """
  
    try:
        import core.config as _cfg
    except Exception:
        _cfg = None

    # Early outs
    if getattr(child, "spawn_A_final", None) is None:
        return True
    if getattr(child, "_dirty_fields", False) or getattr(child, "_dirty_kl", False):
        return True

    # Resolve tolerance (no knobs)
    if mask_tol is None:
        # Prefer explicit per-parent setting; fall back to a generic small tol
        mask_tol = 5e-3
        if _cfg is not None:
            try:
                mask_tol = float(getattr(_cfg, "parent_mask_change_tol",
                                  getattr(_cfg, "spawn_mask_change_tol", 5e-3)))
            except Exception:
                mask_tol = 5e-3
    else:
        mask_tol = float(mask_tol)

    # Compare against previous mask snapshot used for spawn cache
    prev = getattr(child, "_mask_prev_for_spawn", None)
    if prev is None:
        return True

    now = np.asarray(child.mask, dtype=np.float32)
    if now.shape != np.asarray(prev).shape:
        return True

    if float(np.max(np.abs(now - np.asarray(prev, dtype=np.float32)))) > mask_tol:
        return True

    # Cache version bump invalidates spawn caches
    built_ver = int(getattr(child, "_spawn_cache_built_at_cachever", -1))
    cur_ver   = int(getattr(runtime_ctx, "cache_version", 0))
    return built_ver != cur_ver




def _standardize_all_masks_once(all_agents, HW):
    from parents.parent_utils import ensure_hw
    for a in all_agents:
        if hasattr(a, "mask"):
            m = getattr(a, "mask", None)
            if m is not None:
                mm = ensure_hw(m, HW)  
                if mm is not m:
                    setattr(a, "mask", mm)




def list_parents(ctx, *, level: int = 1):
    """Return current parents as a flat list from the level-aware registry."""
    reg, _ = ctx.get_parent_registry(level)  # raises if ctx lacks the helper
    if isinstance(reg, dict):
        return [p for p in reg.values() if p is not None]
    return [reg] if reg is not None else []


def list_nonnull(xs):
    return [x for x in (xs or []) if x is not None]

def bump_cache(ctx, n: int = 1) -> None:
    """Monotonic counter to invalidate spawn/derived caches that track it."""
    setattr(ctx, "cache_version", int(getattr(ctx, "cache_version", 0)) + int(n))






