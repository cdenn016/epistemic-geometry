# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 10:32:58 2025

@author: chris and christine
"""
import core.config as config
import time
import numpy as np
from parents.detector import detect_regions_minimal, _parse_kl_tau 

from parents.parent_spawn import run_cg_for_parents
from parents.parent_spawn import _apply_spawn_from_regions
from runtime_context import ensure_runtime_defaults

from collections import defaultdict

from core.gaussian_core import resolve_support_tau

from parents.parent_semantics import update_parent_lifecycle, reconcile_parent_masks_and_assignments
from updates.update_compute_all_gradients import (
    _apply_children, _compute_child_grads,
    _parent_intrascale_grad_pass, _final_apply_parents
    )
from updates.update_rules_utils import (  
                                compute_and_set_auto_agree_gate )
from updates.update_rules_utils import ( build_alignment_aggregates, build_child_parent_responsibilities,
                                )
from updates.update_rules_utils import  enforce_min_children_per_parent, sync_child_parent_links_from_reconciler

from updates.update_refresh_utils import (
    refresh_agents_light,    
    ensure_spawn_alignment_caches,
    mark_dirty, ensure_dirty, list_nonnull,
    refresh_kl_for_dirty_parents, list_parents, refresh_kl_for_dirty_children
)


def CFG(name, default=None):
    """Unified config accessor with explicit failure if config is absent."""
    return getattr(config, name, default)

class PhaseTimer:
    """Context-manager timer that aggregates per-phase wall time on runtime_ctx."""
    def __init__(self, ctx):
        self.ctx = ctx
        if not hasattr(ctx, "timings") or getattr(ctx, "timings") is None:
            ctx.timings = {}

    def __call__(self, name: str):
        ctx = self.ctx
        class _T:
            def __enter__(_s):
                _s.t0 = time.perf_counter()
                return _s
            def __exit__(_s, *_):
                dt = time.perf_counter() - _s.t0
                ctx.timings[name] = ctx.timings.get(name, 0.0) + dt
        return _T()





# --- synchronous step (simplified, timed) ------------------------------------
def synchronous_step(agents, generators_q, generators_p, params=None, n_jobs=1, runtime_ctx=None):
    
    _CHILD_LEVEL = 0
    _PARENT_LEVEL = 1

    def CFG(key, default=None):
        return (params or {}).get(key, getattr(config, key, default))

    # resolve once per step and persist on ctx
    _ = resolve_support_tau(runtime_ctx, params, default=1e-3, name="support_tau")

    assert runtime_ctx is not None, "synchronous_step: pass runtime_ctx"
    assert agents and (len(agents) > 0), "synchronous_step: no agents"

    params = dict(params or {})
    params["runtime_ctx"] = runtime_ctx

    # Use modern defaults; avoid populating legacy mirrors here.
    runtime_ctx = ensure_runtime_defaults(runtime_ctx)

    if not hasattr(runtime_ctx, "cache") or runtime_ctx.cache is None:
        try:
            from runtime_context import CacheHub
            runtime_ctx.cache = CacheHub()
        except Exception:
            class _NoopCache:
                def clear_on_step(self, *_args, **_kw): pass
                def ns(self, *_args, **_kw):
                    class _D(dict):
                        def clear(self): super().clear()
                    return _D()
            runtime_ctx.cache = _NoopCache()

    def CFG_TAU(name="support_tau", fallback="support_cutoff_eps", default=1e-3):
        v = CFG(name, None)
        if v is None: v = CFG(fallback, default)
        try: return float(v)
        except Exception: return float(default)

    step_now = int(getattr(runtime_ctx, "global_step", 0))
    runtime_ctx.cache.clear_on_step(step_now)

    # right after ctx.cache.clear_on_step(step_now) in synchronous_step
    d = getattr(runtime_ctx, "dirty", None)
    if hasattr(d, "reset_step"):
        d.reset_step(step_now)  # WorkFlags
    # also clear id-sets for this step (harmless if they’re empty)
    try:
        d.agents.clear(); d.kl.clear()
    except Exception:
        print("clear failed")
        pass



    T = PhaseTimer(runtime_ctx)

    frozen_levels     = set(params.get("frozen_levels", []))
    frozen_phi_levels = set(params.get("frozen_phi_levels", []))

    # ---------- Phase 0: pre -------------------------------------------------
    with T("pre"):
        parents    = list_parents(runtime_ctx)
        children   = list_nonnull(agents)
        if not children:
            # Register / advance book-keeping minimally and exit
            runtime_ctx.register_agents(_CHILD_LEVEL, [])
            runtime_ctx.register_agents(_PARENT_LEVEL, parents or [])
            runtime_ctx.global_step += 1
            return agents
        all_agents = children + (parents or [])
        refresh_agents_light(all_agents, params, generators_q, generators_p, ctx=runtime_ctx)

    _wire_hierarchy_maps(runtime_ctx, children, parents)

    # ---------- Phase 1: child grads/apply ----------------------------------
    with T("child_grad"):
        grads = _compute_child_grads(children, all_agents, generators_q, generators_p, params)

    with T("child_apply"):
        _apply_children(children, grads, params, frozen_levels, frozen_phi_levels,
                        generators_q, generators_p, n_jobs, ctx=runtime_ctx)
        for a in children:
            mark_dirty(runtime_ctx, a, kl=True)
        runtime_ctx.children_latest = children

    
    
    with T("ensure_children"):
        ensure_dirty(runtime_ctx, generators_q, generators_p, params, scope="children")
    with T("refresh_child_kl"):
        # all_agents may change later; for child KL this early snapshot is fine
        refresh_kl_for_dirty_children(runtime_ctx, universe=all_agents,
                                      eps=CFG("eps", 1e-6), Gq=generators_q, Gp=generators_p)

    # ---------- Phase 2: spawn caches ---------------------------------------
    with T("spawn_caches"):
        ensure_spawn_alignment_caches(children, children, runtime_ctx,
                                      generators_q=generators_q, generators_p=generators_p,
                                      use_cover_weight=True,
                                      debug=bool(CFG("debug_spawn_log_details", False)))

    # ---------- Phase 3: aggregates + auto gate ------------------------------
    with T("aggregates"):
        try:
            build_alignment_aggregates(runtime_ctx, children)
            parents = list_parents(runtime_ctx)
            if parents:
                build_alignment_aggregates(runtime_ctx, parents)
            if bool(CFG("detector_agree_auto", True)):
                compute_and_set_auto_agree_gate(
                    runtime_ctx,
                    percentile=float(CFG("detector_agree_percentile", 65.0)),
                    ema=float(CFG("detector_agree_ema", 0.2)),
                    use_p_if_q_missing=True,
                )
        except Exception as e:
            if bool(CFG("debug_strict", False)):
                print(f"[AGG] skipped: {e}")

    with T("spawn_detect"):
        H, W = np.asarray(children[0].mask, np.float32).shape[:2]
        # level-aware detector state
        ds = runtime_ctx.get_detector_state(_CHILD_LEVEL)

        every = int(CFG("detector_period", 1))
        run_now = (int(getattr(runtime_ctx, "global_step", 0)) % max(1, every) == 0)

        if run_now:
            regions, cand_now = detect_regions_minimal(
                children, H=H, W=W,
                level=_CHILD_LEVEL,
                min_kids=int(CFG("seed_min_kids", 2)),
                min_area=int(CFG("emerge_min_area", 8)),
                kl_tau=_parse_kl_tau(CFG("emerge_tau", 0.28)),
                weight_tau=CFG("parent_weight_tau", 0.6),
                erode_iters=CFG("detector_edge_erode", 0),
                runtime_ctx=runtime_ctx,
                agree_gate_tau=CFG("agree_gate_tau", None),
                return_cand_map=True,
            )
            beta = float(CFG("detector_cand_ema", 0.7))
            prev = getattr(ds, "ema_cand", None)
            if (prev is None) or (getattr(prev, "shape", None) != (H, W)):
                prev = np.zeros((H, W), np.float32)
            ds.ema_cand = beta * prev + (1.0 - beta) * np.asarray(cand_now, np.float32)
            cand_map = ds.ema_cand
        else:
            regions = {}
            cand_map = getattr(ds, "ema_cand", np.zeros((H, W), np.float32))

        active_regions = regions
        regs_to_spawn = _gate_spawns(active_regions, runtime_ctx)

    with T("spawn_apply"):
        if regs_to_spawn:
            changed_parents = _apply_spawn_from_regions(
                runtime_ctx, regs_to_spawn, children, generators_q, generators_p, params, level=_CHILD_LEVEL
            ) or []
        else:
            changed_parents = []

    # ---------- Phase 4: parent masks ---------------------------------------
    with T("parent_masks"):
        step_now = int(getattr(runtime_ctx, "global_step", 0))
        # use the level-aware parent registry; also mirror for legacy utilities
        parent_registry, _next_pid = runtime_ctx.get_parent_registry(_PARENT_LEVEL)
        
        prev_masks = {int(pid): P.mask.copy() for pid, P in (parent_registry or {}).items()}

        H, W = (children[0].mask.shape[:2] if children else (0, 0))
        reconcile_parent_masks_and_assignments(runtime_ctx, children, step=step_now, periodic=True, H=H, W=W, level=1)

        change_tol = float(CFG("parent_mask_change_tol", 5e-3))
        changed_masks = []
        for pid, P in (parent_registry or {}).items():
            pm = prev_masks.get(int(pid))
            if pm is None:
                changed_masks.append(P); continue
            import numpy as _np
            if float(_np.max(_np.abs(_np.asarray(P.mask, _np.float32) - _np.asarray(pm, _np.float32)))) >= change_tol:
                changed_masks.append(P)

    # ---------- Phase 5: mirror assignments + guards -------------------------
    with T("assign_sync_from_reconciler"):
        sync_child_parent_links_from_reconciler(children, parent_registry)
        enforce_min_children_per_parent(parent_registry,
                                        min_kids=int(CFG("parent_min_kids", 2)),
                                        respect_grace=True)

    with T("assign_build_responsibilities"):
        build_child_parent_responsibilities(
            children, parent_registry,
            mode=str(CFG("parent_resp_mode", "pixel")).lower()
        )

    # ---------- Phase 6: lifecycle ------------------------------------------
    with T("lifecycle"):
        _ = update_parent_lifecycle(runtime_ctx, int(getattr(runtime_ctx, "global_step", 0)),level=_PARENT_LEVEL)
        # delete flagged parents in the level-aware registry and persist
        for pid, P in list(parent_registry.items()):
            if getattr(P, "_delete", False):
                parent_registry.pop(pid, None)
        runtime_ctx.set_parent_registry(_PARENT_LEVEL, parent_registry, _next_pid)

    # ---------- Phase 7: prepare transforms before CG ------------------------
    with T("parents_prepare"):
        if changed_masks:
            for p in changed_masks:
                mark_dirty(runtime_ctx, p)
        ensure_dirty(runtime_ctx, generators_q, generators_p, params, scope="parents")

    # --- Phase 8: coarsegrain ------------------------------------------------
    with T("parent_coarsegrain"):
        try:
            cg_mode = str(CFG("use_parent_cg", False)).lower()
            if cg_mode in ("false", "0", "off"):
                target_parents = []
            else:
                cg_tau    = CFG_TAU()
                periodic  = int(CFG("parent_cg_period", 0))
                run_all   = (cg_mode == "periodic") and (periodic > 0) and (step_now % periodic == 0)
                if run_all:
                    target_parents = list((parent_registry or {}).values())
                else:
                    target_parents = (changed_masks if cg_mode in ("true", "1", "on", "yes") else [])
            if target_parents:
                _ = run_cg_for_parents(runtime_ctx, target_parents, children, generators_q, generators_p,
                                       eps=float(CFG("cg_eps", 1e-6)),
                                       mask_tau=cg_tau,
                                       backend=str(CFG("cg_backend", "loky")).lower(),
                                       n_jobs=int(CFG("cg_jobs", 1)),
                                       batch_size=CFG("cg_batch", 1),
                                       prefer=CFG("cg_prefer", None),
                                       omp_thr=int(CFG("cg_omp_thr", 1)))
                for p in target_parents:
                    mark_dirty(runtime_ctx, p)
                ensure_dirty(runtime_ctx, generators_q, generators_p, params, scope="parents")
        except Exception as e:
            if bool(CFG("debug_strict", False)):
                print(f"[CG] skipped: {e}")

    # ---------- Phase 9: refresh KL AFTER CG ---------------------------------
    with T("parents_kl_refresh"):
        try:
            # Recompute universe so we include any new/removed parents
            cur_parents = list_parents(runtime_ctx)
            cur_all_agents = children + (cur_parents or [])
            refresh_kl_for_dirty_parents(runtime_ctx, universe=cur_all_agents, eps=CFG("eps", 1e-6))
        except Exception:
            print("KL REFRESH Phase 9: ")

    # ---------- Phase 10: parent-level grads/updates -------------------------
    with T("parent_grads_apply"):
        parents = list_parents(runtime_ctx)
        if parents:
            mode = str(params.get("parent_update_mode", CFG("parent_update_mode", "off"))).lower()
            period = int(params.get("parent_update_period", CFG("parent_update_period", 50)))
            if (mode in ("on", "true", "1", "yes")) or (mode == "periodic" and period > 0 and step_now % period == 0):
                ensure_dirty(runtime_ctx, generators_q, generators_p, params, scope="parents")
                _parent_intrascale_grad_pass(children, parents, generators_q, generators_p, params,
                                             accum_into=True, ctx=runtime_ctx)
                _final_apply_parents(runtime_ctx, params)
                for p in parents:
                    mark_dirty(runtime_ctx, p)
                ensure_dirty(runtime_ctx, generators_q, generators_p, params, scope="parents")
                try:
                    cur_parents = list_parents(runtime_ctx)
                    cur_all_agents = children + (cur_parents or [])
                    refresh_kl_for_dirty_parents(runtime_ctx, universe=cur_all_agents, eps=CFG("eps", 1e-6))
                except Exception:
                    pass
            elif bool(CFG("parent_diag_when_updates_off", False)):
                try:
                    cur_parents = list_parents(runtime_ctx)
                    cur_all_agents = children + (cur_parents or [])
                    refresh_kl_for_dirty_parents(runtime_ctx, universe=cur_all_agents, eps=CFG("eps", 1e-6))
                except Exception:
                    print("Phase 10 failure: ")

    # ---------- Phase 11: register/mirror + bookkeeping ----------------------
    with T("register_mirror"):
        runtime_ctx.register_agents(_CHILD_LEVEL, children) if children else None
        parents = list_parents(runtime_ctx)
        runtime_ctx.register_agents(_PARENT_LEVEL, parents)  if parents  else None

        if children and parents:
            ensure_dirty(runtime_ctx, generators_q, generators_p, params, scope="all")
            child_ids  = [int(getattr(ch, "id")) for ch in children]
            parent_ids = [int(getattr(p, "id")) for p in parents]
            pid_set    = set(parent_ids)
            labels = np.full(len(child_ids), -1, dtype=int)
            for i, ch in enumerate(children):
                pids = getattr(ch, "parent_ids", ()) or ()
                labels[i] = next((int(pid) for pid in pids if int(pid) in pid_set), -1)
            runtime_ctx.set_crossscale_labels(_CHILD_LEVEL, _PARENT_LEVEL, child_ids, parent_ids, labels=labels)
        
        runtime_ctx.children_latest = children
        # expose latest parents from the level-aware registry
        runtime_ctx.parents_latest  = parent_registry
        runtime_ctx.global_step    += 1

    # ---------- Phase 12: timing report --------------------------------------
    if bool(CFG("debug_phase_timing", True)):
        print("[timings]")
        for name, dt in sorted(getattr(runtime_ctx, "timings", {}).items()):
            print(f"  {name:20s} {dt:8.3f}s")
        if getattr(runtime_ctx, "counters", None):
            print("[counts]", runtime_ctx.counters)

    return agents




# update_refresh_utils.py
def invalidate_all_after_update(ctx):
    # clear everything that depends on φ/μ/Σ
    for ns in (
        "exp",       # E-grid e^{Φ}
        "omega",     # exp/log helpers (if any)
        "jinv",      # dexp^{-1}
        "morphism",  # Φ, Φ̃
        "fisher",    # natgrad
        "kl",        # if you cache KLs
        "align",     # alignment aggregates
        "spawn",     # spawn/detector caches
        "aggregates",
        "misc",
    ):
        try:
            ctx.cache.invalidate_ns(ns)
        except AttributeError:
            try:
                ctx.cache.ns(ns).clear()
            except Exception:
                pass





def _wire_hierarchy_maps(ctx, children, parents):
    """
    Populate ctx.parents_for_agent and ctx.children_for_agent.
    Works with whatever fields are available; prefers explicit links,
    
    """
    parents_for = defaultdict(list)
    children_for = defaultdict(list)

    # normalize containers
    children = list(children or [])
    parents  = list(parents  or [])

    if not parents:
        # level-aware fallback: use parent registry at level 1 (standard)
        try:
            reg, _ = ctx.get_parent_registry(1)
            parents = list((reg or {}).values())
        except Exception:
            parents = []

    # index by id for quick lookups
    by_id = {}
    for a in children + parents:
        try:
            by_id[int(getattr(a, "id"))] = a
        except Exception:
            pass

    # 1) use explicit links if present
    for p in parents:
        kids = getattr(p, "children", None)
        if kids:
            for c in kids:
                cid = int(getattr(c, "id", getattr(c, "child_id", -1)))
                ch = by_id.get(cid, c)
                if ch is None: continue
                parents_for[cid].append(p)
                children_for[int(getattr(p, "id"))].append(ch)

    for c in children:
        pars = getattr(c, "parents", None)
        if pars:
            cid = int(getattr(c, "id"))
            for p in pars:
                pid = int(getattr(p, "id", getattr(p, "parent_id", -1)))
                par = by_id.get(pid, p)
                if par is None: continue
                parents_for[cid].append(par)
                children_for[pid].append(c)

        # 2) fall back to id lists
        pid_list = getattr(c, "parent_ids", None) or getattr(c, "parents_ids", None)
        if pid_list:
            cid = int(getattr(c, "id"))
            for pid in pid_list:
                pid = int(pid)
                par = by_id.get(pid)
                if par:
                    parents_for[cid].append(par)
                    children_for[pid].append(c)

 
    # de-dup
    for d in (parents_for, children_for):
        for k, lst in d.items():
            seen = set(); uniq = []
            for a in lst:
                aid = int(getattr(a, "id", -1))
                if aid in seen: continue
                seen.add(aid); uniq.append(a)
            d[k] = uniq

    ctx.parents_for_agent  = dict(parents_for)
    ctx.children_for_agent = dict(children_for)




def _gate_spawns(active_regions, ctx):
    
   
    if not active_regions:
        return []
    conf_tau = CFG("spawn_confidence_tau", None)
    if conf_tau is None and bool(CFG("spawn_conf_auto", True)):
        ds = getattr(ctx, "detector_state", None)
        conf_tau = float(getattr(ds, "conf_tau_eff", 0.60)) if ds is not None else 0.60
    conf_tau = 0.60 if conf_tau is None else float(conf_tau)
    regs = [r for r in active_regions if float(getattr(r, "confidence", 1.0)) >= conf_tau]
    regs.sort(key=lambda r: float(getattr(r, "confidence", 1.0)), reverse=True)
    return regs[: int(CFG("parent_max_new_per_step", 3))]




