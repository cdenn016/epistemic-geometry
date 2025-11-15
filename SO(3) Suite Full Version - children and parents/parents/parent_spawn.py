# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 09:17:47 2025

@author: chris and christine
"""
# -*- coding: utf-8 -*-
"""
Parent spawning & coarse-grain pipeline (CacheHub-friendly).
No legacy mirrors, no per-agent transport caches.
"""

import os

import numpy as np
import core.config as config

from joblib import Parallel, delayed

from transport.bundle_morphism_utils import init_parent_morphisms, have_parent_morphisms
from parents.parent_utils import (norm_active_regions, dedup_peaks_by_childset, spawn_spacing_radius,
    prepare_parent_match_state, append_center_from_mask, centroid_toroidal, dist2_toroidal,
    jaccard_sets, match_score_for_parent, respect_spawn_spacing,    
    existing_cores_and_centroids, filter_spawn_regions, select_cg_targets_stable, update_parent_weights, 
    apply_mask_and_centroid, assert_parent_invariants, ensure_parent_registry,
     cc_label, compute_cover2_map, merge_parents_into_registry, seed_proposal_or_skip, 
     novelty_or_skip, continuity_try_rescue, conservative_child_set, _spawn_parent)

from core.gaussian_core import (
    _overlap_pairs, directed_kl_weight_after_transport,
    apply_evidence_modulators, curvature_boltzmann_from_children,
)
from transport.bundle_coarsegrain_init import coarsegrain_parent_from_children
from core.gaussian_core import resolve_support_tau
from transport.transport_cache import warm_E



def _apply_spawn_from_regions(ctx, active_regions, agents, Gq, Gp, params, *, level: int = 0):
    """
    Mutates parents:
      - spawn/update parents for active regions (PIXELWISE proposals path)
      - coarse-grain μ,Σ only for **stable** (pre-existing) parents that actually need it
    """
    

    if not agents:
        return []

    # init + stable snapshot
    existing, next_parent_id, prev_ids, H, W = ensure_parent_registry(ctx, agents, level)

    # prefilter
    regions_to_spawn = list(active_regions or [])
    

    cover2 = None
    try:
        cover2 = compute_cover2_map(agents, float(getattr(config, "support_tau", 0.08)))
    except Exception:
        cover2 = None

    spawn_min_cover2_px = int(getattr(config, "spawn_min_cover2_px", 2))
    spawn_nms_iou       = float(getattr(config, "spawn_nms_iou", 0.25))
    spawn_min_dist_px   = float(getattr(config, "spawn_min_dist_px", 2.0))
    abs_core            = float(getattr(config, "core_abs_tau", 0.20))
    rel_core            = float(getattr(config, "core_rel_tau", 0.50))

    existing_cores, existing_centroids = existing_cores_and_centroids(
        existing, abs_core, rel_core
    )

    kept = filter_spawn_regions(
        regions_to_spawn, cover2, spawn_min_cover2_px,
        existing_cores, existing_centroids, spawn_nms_iou, spawn_min_dist_px
    )
    

    # spawn/update (ctx-aware)
    parents_changed, next_pid = spawn_or_update_parents(
        kept,
        agents,
        existing,
        next_parent_id,
        field_generators=(Gq, Gp),
        agree_map=(getattr(ctx, "Aq_agg", None), getattr(ctx, "Ap_agg", None)),
        ctx=ctx,  # enable downstream caches if used
    )

    # registry merge + bookkeeping
    parents_changed, step_now = merge_parents_into_registry(ctx, parents_changed, level)

    # CG only on stable parents that need it
    min_core_px = int(getattr(config, "cg_min_core_px", 3))
    cg_tau = resolve_support_tau(ctx, params)

    items_all, cg_items = select_cg_targets_stable(
        parents_changed, prev_ids,
        step_now=step_now, abs_core=abs_core, rel_core=rel_core,
        min_core_px=min_core_px, mask_tau=cg_tau
    )



    # backend knobs
    _cfg_backend = str(getattr(config, "cg_backend", "threading")).lower()
    if _cfg_backend in ("threads", "threading"):
        backend, prefer = "threading", "threads"
    elif _cfg_backend in ("process", "processes", "loky", "multiprocessing"):
        backend, prefer = "loky", "processes"
    elif _cfg_backend in ("seq", "sequential"):
        backend, prefer = "sequential", None
    else:
        backend, prefer = "threading", "threads"

    n_jobs       = int(getattr(config, "cg_n_jobs", -1))
    batch_size   = getattr(config, "cg_batch_size", "auto")
    omp_thr      = int(getattr(config, "cg_omp_threads", 1))
    cg_eps       = float(getattr(config, "cg_eps", 1e-6))
    auto_seq_cut = int(getattr(config, "cg_auto_seq_cutoff", 2))
    if len(cg_items) <= auto_seq_cut:
        backend, prefer = "sequential", None

    cg_ok, cg_fail = run_cg_for_parents(
        ctx, cg_items, agents, Gq, Gp,
        eps=cg_eps, mask_tau=cg_tau, backend=backend,
        n_jobs=n_jobs, batch_size=batch_size, prefer=prefer, omp_thr=omp_thr
    )


    return list(existing.values())






def spawn_or_update_parents(active_regions,
                            agents,
                            parents_by_id,
                            next_parent_id,
                            *,
                            field_generators,
                            agree_map=None,
                            ctx=None):   # ← accept ctx
    """
    Build pixelwise proposals (dual-fiber, directed-KL only), restrict to active regions,
    dedup, then match/spawn. Returns (list(parents_by_id.values()), next_parent_id).
    """
    
    if not agents:
        return list(parents_by_id.values()), next_parent_id

    H, W = np.asarray(agents[0].mask, np.float32).shape[:2]
    dtype = agents[0].mu_q_field.dtype
    Kq   = int(agents[0].mu_q_field.shape[-1])
    Kp   = int(agents[0].mu_p_field.shape[-1])
    Gq, Gp = field_generators

    # require generators when fibers exist
    if Kq > 0 and Gq is None:
        raise ValueError("spawn_or_update_parents: missing Gq for q-fiber.")
    if Kp > 0 and Gp is None:
        raise ValueError("spawn_or_update_parents: missing Gp for p-fiber.")

    # --- normalize detector regions → union mask to restrict spawn area -----
    region_masks = norm_active_regions(active_regions, H, W)
    region_or = None
    if region_masks:
        region_or = np.zeros((H, W), np.float32)
        for R in region_masks:
            region_or = np.maximum(region_or, np.asarray(R, np.float32))
        region_or = (region_or > 0.5)

    # --- parse agree_map and combine Aq/Ap with geometric blend -------------
    Aq = Ap = None
    if agree_map is not None:
        if isinstance(agree_map, (tuple, list)) and len(agree_map) >= 1:
            Aq = agree_map[0]
            if len(agree_map) >= 2:
                Ap = agree_map[1]
        elif isinstance(agree_map, dict):
            Aq = agree_map.get("q")
            Ap = agree_map.get("p")
        else:
            Aq = agree_map

    alpha = float(getattr(config, "blend_qp_alpha", 0.50))
    A = None
    if (Aq is not None) or (Ap is not None):
        def _prep_A(a, H, W):
            if a is None: return None
            a = np.asarray(a, np.float32)
            if a.shape != (H, W):
                if a.shape[0] >= H and a.shape[1] >= W:
                    a = a[:H, :W]
                else:
                    py, px = max(0, H - a.shape[0]), max(0, W - a.shape[1])
                    a = np.pad(a, ((0, py), (0, px)), mode="edge")[:H, :W]
            return np.clip(a, 0.0, 1.0).astype(np.float32, copy=False)

        Aq_ = _prep_A(Aq, H, W) if Aq is not None else None
        Ap_ = _prep_A(Ap, H, W) if Ap is not None else None
        if (Aq_ is not None) and (Ap_ is not None):
            eps = 1e-12
            A = np.exp((1.0 - alpha) * np.log(np.maximum(Aq_, eps)) +
                       alpha * np.log(np.maximum(Ap_, eps)))
        else:
            A = Aq_ if Aq_ is not None else Ap_

    # --- warm exp(φ) in central transport cache (Ω built on demand) ---------
    try:
        
        whiches = tuple(w for w, k in (("q", Kq), ("p", Kp)) if k > 0)
        if whiches:
            warm_E(ctx, agents, whiches=whiches)
    except Exception:
        pass

    # --- dual-fiber, directed-KL proposals (ctx + explicit generators) ------
    kwargs = dict(
        G_q=Gq,
        G_p=Gp,
        mask_tau=float(getattr(config, "support_tau", 0.10)),
        tau_q=float(getattr(config, "tau_align_q", 0.30)),
        tau_p=float(getattr(config, "tau_align_p", 0.30)),
        alpha=alpha,
        min_pair_px=int(getattr(config, "pxspawn_min_pair_px", 4)),
        agree_map=A,
        agree_power=float(getattr(config, "pxspawn_agree_power", 1.0)),
        min_component_px=int(getattr(config, "parent_min_area", 10)),
        smooth_sigma=float(getattr(config, "pxspawn_smooth_sigma", 0.0)),
        per_pixel_norm=bool(getattr(config, "pxspawn_per_pixel_norm", False)),
        debug=bool(getattr(config, "debug_spawn_log", False)),
        domain_mask=region_or,
    )

    proposals = proposals_from_dual_fibers(ctx, agents, H, W, **kwargs)

    if region_or is not None and proposals:
        rim = region_or.astype(np.float32)
        for p in proposals:
            p["mask"] = np.clip(np.asarray(p["mask"], np.float32) * rim, 0.0, 1.0)

    if proposals:
        try:
            proposals = dedup_peaks_by_childset(
                proposals,
                radius_px=int(getattr(config, "parent_peak_dedup_radius_px", 3)),
                j_thr=float(getattr(config, "parent_peak_dedup_child_jaccard", 0.92)),
                subset_thr=float(getattr(config, "parent_peak_dedup_subset_thr", 0.90)),
                H=H, W=W,
                per_x=bool(getattr(config, "periodic_x", False)),
                per_y=bool(getattr(config, "periodic_y", False)),
            )
        except Exception:
            pass

    if not proposals and not parents_by_id:
        return list(parents_by_id.values()), next_parent_id

    peaks = []
    for p in (proposals or []):
        q = dict(p)
        if "proposal" not in q and "mask" in q:
            q["proposal"] = np.asarray(q["mask"], np.float32)
        peaks.append(q)

    # --- match or spawn (PASS ctx AS KEYWORD) --------------------------------
    min_interseed_px = spawn_spacing_radius(
        True, int(getattr(config, "parent_min_interseed_px", 1))
    )
    next_parent_id = _match_or_spawn(
        peaks, parents_by_id, next_parent_id, agents,
        (H, W, dtype, Kq, Kp), field_generators,
        match_iou_thr=float(getattr(config, "parent_match_iou_existing", 0.10)),
        child_jac_thr=float(getattr(config, "parent_match_child_jaccard_thr", 0.85)),
        support_tau=float(getattr(config, "support_tau", 0.08)),
        max_new_per_step=int(getattr(config, "parent_max_new_per_step", 50)),
        min_interseed_px=min_interseed_px,
        ctx=ctx,  # ← kw-only arg
    )

    for P in parents_by_id.values():
        if not have_parent_morphisms(P):
            init_parent_morphisms(P, generators_q=Gq, generators_p=Gp)

    return list(parents_by_id.values()), next_parent_id



def proposals_from_dual_fibers(
    ctx,
    children,
    H,
    W,
    *,
    G_q=None,
    G_p=None,
    mask_tau=0.10,
    tau_q=0.30,
    tau_p=0.30,
    alpha=0.5,
    min_pair_px=8,
    agree_map=None,
    agree_power=1.0,
    min_component_px=20,
    smooth_sigma=0.0,
    per_pixel_norm=False,
    debug=False,
    domain_mask=None,
):
    """
    Gauge-covariant proposals using ONLY directed KL after transport:
      d_q = KL( N_i(q) || T_{j->i}(N_j) ),  d_p = KL( N_i(p) || T_{j->i}(N_j) )
      w   = exp(-d_q/tau_q)^(1-alpha) * exp(-d_p/tau_p)^alpha
      E(x)= Σ_{(i,j)} m_i m_j  w  1_{overlap}

    Notes
    -----
    - Uses ctx-aware transport/KL via directed_kl_weight_after_transport(..., ctx=ctx, G=G_*).
    - Ω/exp caching is handled centrally by the runtime transport cache (no per-agent caches).
    """
    import numpy as _np
    import core.config as _cfg

    if not children:
        return []

    # --- optional smoothing helper ------------------------------------------
    try:
        from scipy.ndimage import gaussian_filter as _gauss_filter
    except Exception:
        _gauss_filter = None

    from core.numerical_utils import resize_nn

    # --- fields & sizes ------------------------------------------------------
    M   = [_np.asarray(getattr(c, "mask", 0.0), _np.float32) for c in children]
    muq = [_np.asarray(c.mu_q_field, _np.float32) for c in children]
    Sq  = [_np.asarray(c.sigma_q_field, _np.float32) for c in children]
    mup = [_np.asarray(c.mu_p_field, _np.float32) for c in children]
    Sp  = [_np.asarray(c.sigma_p_field, _np.float32) for c in children]

    Kq  = int(muq[0].shape[-1]) if muq and muq[0].ndim >= 1 else 0
    Kp  = int(mup[0].shape[-1]) if mup and mup[0].ndim >= 1 else 0

    if Kq > 0 and G_q is None:
        raise ValueError("proposals_from_dual_fibers: missing G_q for q-fiber.")
    if Kp > 0 and G_p is None:
        raise ValueError("proposals_from_dual_fibers: missing G_p for p-fiber.")

    # --- candidate pairs by overlap -----------------------------------------
    pairs = _overlap_pairs(children, H, W, tau=float(mask_tau), min_pair_px=int(min_pair_px))
    if not pairs:
        return []

    # --- agreement / curvature priors ---------------------------------------
    def _prep_A(A):
        if A is None:
            return None
        A = _np.asarray(A, _np.float32)
        if A.shape[:2] != (H, W):
            try:
                A = resize_nn(A, (H, W)).astype(_np.float32)
            except Exception:
                return None
        return _np.clip(A, 0.0, 1.0)

    A = _prep_A(agree_map)

    curv_fiber = str(getattr(_cfg, "curvature_fiber", "q"))
    curv_G = G_q if curv_fiber == "q" else G_p
    BF = curvature_boltzmann_from_children(
        children, H, W,
        gamma=float(getattr(_cfg, "curvature_gamma", 0.0)),
        fiber=curv_fiber,
        curvature_map=getattr(_cfg, "curvature_map", None) if hasattr(_cfg, "curvature_map") else None,
        ctx=ctx,
        G=curv_G,
    )

    # --- domain: detector mask ∧ (cover ≥ 2) --------------------------------
    cover2 = _np.zeros((H, W), _np.int16)
    for m in M:
        cover2 += (m >= float(mask_tau)).astype(_np.int16)
    D = (cover2 >= 2)
    if domain_mask is not None:
        D = D & _np.asarray(domain_mask, bool)

    E = _np.zeros((H, W), _np.float32)
    pair_contribs = []

    t_q = max(1e-6, float(tau_q))
    t_p = max(1e-6, float(tau_p))
    a   = float(alpha)

    # --- main pair loop: ctx-aware KL weights after transport ----------------
    for (i, j, ov_raw) in pairs:
        if not ov_raw.any():
            continue
        Ai, Aj = children[i], children[j]
        ov = (ov_raw & D)
        if not ov.any():
            continue

        # Directed KL weights via central cache (ctx) and explicit generators
        wq = directed_kl_weight_after_transport(
            Ai, Aj, muq[i], Sq[i], muq[j], Sq[j],
            fiber="q", H=H, W=W, K=Kq, ov=ov, tau=t_q,
            ctx=ctx, G=G_q,
        ) if Kq > 0 else _np.ones((H, W), _np.float32)

        wp = directed_kl_weight_after_transport(
            Ai, Aj, mup[i], Sp[i], mup[j], Sp[j],
            fiber="p", H=H, W=W, K=Kp, ov=ov, tau=t_p,
            ctx=ctx, G=G_p,
        ) if Kp > 0 else _np.ones((H, W), _np.float32)

        w = (wq ** (1.0 - a)) * (wp ** a)
        w = _np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        w *= ov.astype(_np.float32)

        # evidence modulators (agreement prior & curvature); respects support
        wij = apply_evidence_modulators(
            w, M[i], M[j], ov, A=A, agree_power=agree_power, BF=BF
        )
        if float(wij.max()) <= 0.0:
            continue

        E += wij
        pair_contribs.append((i, j, wij))

    # enforce domain & early exit
    E *= D.astype(_np.float32)
    if float(E.max()) <= 0.0:
        return []

    # --- components on adaptive threshold -----------------------------------
    pos = E[E > 0.0]
    thr = float(_np.quantile(pos, 0.20)) if pos.size else 0.0
    B = (E > thr).astype(_np.uint8)

    comp_ids, n_comp = cc_label(B)  # parent_utils.cc_label: (H,W)->(labels, n)

    proposals = []
    for cid in range(1, int(n_comp) + 1):
        comp = (comp_ids == cid)
        area = int(comp.sum())
        if area < int(min_component_px):
            continue

        m = (E * comp.astype(_np.float32))
        mmax = float(m.max())
        if mmax > 0.0:
            m = (m / mmax).astype(_np.float32)

        if smooth_sigma and _gauss_filter is not None:
            m = _gauss_filter(m, sigma=float(smooth_sigma)).astype(_np.float32)
            m = _np.clip(m, 0.0, 1.0)

        ys, xs = _np.nonzero(comp)
        cy = float(ys.mean()) if ys.size else 0.0
        cx = float(xs.mean()) if xs.size else 0.0

        # child weights from pair contributions over this component
        weights = {}
        for (ii, jj, wij) in pair_contribs:
            wloc = float(wij[comp].sum())
            if wloc <= 0.0:
                continue
            ci = int(getattr(children[ii], "id", ii))
            cj = int(getattr(children[jj], "id", jj))
            weights[ci] = weights.get(ci, 0.0) + wloc
            weights[cj] = weights.get(cj, 0.0) + wloc
        totw = sum(weights.values()) or 1.0
        weights = {int(k): float(v / totw) for k, v in weights.items()}

        proposals.append({
            "cy": cy, "cx": cx,
            "mask": m,
            "child_ids": tuple(sorted(weights.keys())),
            "child_weights": weights,
            "score": float(E[comp].sum()) / max(1.0, area),
        })

    # optional per-pixel normalization across proposals
    if per_pixel_norm and proposals:
        stack = _np.stack([p["mask"] for p in proposals], axis=0)
        denom = _np.maximum(1e-6, stack.sum(axis=0))
        for p in proposals:
            p["mask"] = (p["mask"] / denom).astype(_np.float32)

    if debug:
        print(f"[PX-SPAWN/DKL] proposals={len(proposals)} comps={int(n_comp)} "
              f"Emax={E.max():.3f} alpha={a} gamma={float(getattr(_cfg,'curvature_gamma',0.0))}")
        for p in sorted(proposals, key=lambda d: -d.get('score', 0.0))[:min(8, len(proposals))]:
            print(f"  - kids={tuple(sorted(p.get('child_ids', ())))} "
                  f"area={(p['mask']>0.01).sum()} peak={float(p['mask'].max()):.3f} score={p['score']:.4f}")

    return proposals



# ----------------------------- refactor entry -----------------------------
def _match_or_spawn(
    peaks,
    parents_by_id,
    next_parent_id,
    agents,
    shape_info,
    generators,
    *,
    match_iou_thr,
    child_jac_thr,
    support_tau,
    max_new_per_step,
    min_interseed_px,
    ctx=None,  # optional runtime_ctx
):
    """
    Match proposals to existing parents or spawn new ones.
    Thread-safe w.r.t. ctx (for CacheHub-backed spawns).
    """

    # runtime context
    _RC = ctx if ctx is not None else globals().get("runtime_ctx", None)
    step_now = int(getattr(_RC, "global_step", 0)) if _RC is not None else 0

    # shapes, generators, periodicity
    H, W, dtype, Kq, Kp = shape_info
    Gq, Gp = generators
    per_x = bool(getattr(config, "periodic_x", True))
    per_y = bool(getattr(config, "periodic_y", True))

    # existing parent state
    exists_support, exists_kids = prepare_parent_match_state(parents_by_id, support_tau)

    # start chosen-centers with existing parents
    chosen_centers = []
    for pid, supp in exists_support.items():
        append_center_from_mask(chosen_centers, supp, pid, H, W, per_x=per_x, per_y=per_y)

    # thresholds / knobs
    local_tau      = float(getattr(config, "parent_proposal_local_tau", 0.01))
    min_local_px   = int(getattr(config, "parent_min_seed_area_px", 1))
    mask_tau_child = float(getattr(config, "support_tau", 1e-3))
    min_overlap_px = int(getattr(config, "parent_assign_min_overlap_px", 8))
    min_seed_px    = int(getattr(config, "parent_min_seed_px", max(4, min_overlap_px // 2)))
    keep_thr_cont  = float(getattr(config, "parent_continuity_child_jacc_keep_thr", 0.60))
   
    min_novel_px   = int(getattr(config, "parent_min_novel_seed_px", 6))
    min_novel_frac = float(getattr(config, "parent_min_novel_seed_frac", 0.20))

    # counters
    updated = spawned = 0
    dropped_local = dropped_spacing = dropped_budget = 0
    dropped_childov = dropped_seed = 0

    new_spawned = 0
    # highest-score-first
    for peak in sorted(peaks or [], key=lambda d: -float(d.get("score", 0.0))):
        if new_spawned >= int(max_new_per_step):
            dropped_budget += 1
            break

        prop = peak.get("proposal", peak.get("mask"))
        cy = float(peak.get("cy", 0.0)); cx = float(peak.get("cx", 0.0))

        # FIX: pass agents (not 'children')
        gated = seed_proposal_or_skip(
            peak,
            H=H, W=W, agents=agents,
            local_tau=local_tau, min_local_px=min_local_px,
            mask_tau_child=mask_tau_child, support_tau=support_tau, min_seed_px=min_seed_px,
        )
        if gated is None:
            continue
        prop, seed_bool, seed_px = gated

        # 2) novelty
        ok, novel_px, novel_frac = novelty_or_skip(
            seed_bool, exists_support=exists_support, H=H, W=W,
            min_novel_px=min_novel_px, min_novel_frac=min_novel_frac
        )
        if not ok:
            if bool(getattr(config, "debug_spawn_log_details", True)):
                print(f"  -> DROP (not novel) novel_px={novel_px} "
                      f"seed_px={seed_px} frac={novel_frac:.2f} "
                      f"min_px={min_novel_px} min_frac={min_novel_frac}")
            dropped_spacing += 1
            continue

        # 3) continuity rescue (attach to nearby prior parent if consistent)
        cont_r2      = float(getattr(config, "parent_continuity_radius_px", max(1, min_interseed_px))) ** 2
        cont_iou_thr = float(getattr(config, "parent_continuity_iou_thr", 0.15))
        cont_pid = continuity_try_rescue(
            cy, cx, seed_bool, exists_support=exists_support, H=H, W=W,
            per_x=per_x, per_y=per_y, cont_r2=cont_r2, cont_iou_thr=cont_iou_thr
        )
        if cont_pid is not None:
            P = parents_by_id[int(cont_pid)]
            supp = apply_mask_and_centroid(
                P, prop, dtype=dtype, support_tau=support_tau, H=H, W=W,
                chosen_centers=chosen_centers, pid=cont_pid,
                per_x=per_x, per_y=per_y, fallback_cyx=(cy, cx)
            )
            # keep/refresh child weights conservatively
            cw = peak.get("child_weights", None)
            if isinstance(cw, dict) and cw:
                old = set(getattr(P, "child_ids", ()))
                new = set(int(k) for k in cw.keys())
                try:
                    j = jaccard_sets(old, new) if old else 1.0
                except Exception:
                    j = 1.0 if not old else 0.0
                if j >= keep_thr_cont:
                    P.child_weights = {int(k): float(v) for k, v in cw.items()}
                    P.child_ids = tuple(sorted(P.child_weights.keys()))
            P._region_proposal = prop
            exists_support[cont_pid] = (P.mask > support_tau)
            updated += 1
            if bool(getattr(config, "debug_spawn_log_details", True)):
                print(f"  -> CONT-MATCH pid={cont_pid} seed_px={seed_px}")
            continue

        # 4) conservative kid set
        gkids_seed = conservative_child_set(
            agents, seed_bool, mask_tau_child=mask_tau_child, min_overlap_px=min_overlap_px
        )
        gkids_peak = set(peak.get("child_ids", ()))
        gkids = gkids_seed if not gkids_peak else (gkids_seed & gkids_peak)
        if len(gkids) < 2:
            if bool(getattr(config, "debug_spawn_log_details", True)):
                print(f"  -> DROP (child overlap) |kids|={len(gkids)} < 2 (ov_px≥{min_overlap_px})")
            dropped_childov += 1
            continue

        # 5) MATCH existing (nearby radius + joint score: IoU × child-Jaccard)
        R2 = float(max(1, min_interseed_px)) ** 2
        best_pid, best_sc = None, (-1.0, -1.0)
        for pid, supp in exists_support.items():
            cy0, cx0 = centroid_toroidal(supp, H, W, per_y=per_y, per_x=per_x)
            close = (dist2_toroidal(cy, cx, cy0, cx0, H, W, per_y, per_x) <= R2)
            sc = match_score_for_parent(seed_bool, gkids, supp, exists_kids[pid],
                                        close=close,
                                        match_iou_thr=match_iou_thr,
                                        child_jac_thr=child_jac_thr)
            if sc is not None and sc > best_sc:
                best_sc, best_pid = sc, pid

        if best_pid is not None:
            P = parents_by_id[int(best_pid)]
            supp = apply_mask_and_centroid(
                P, prop, dtype=dtype, support_tau=support_tau, H=H, W=W,
                chosen_centers=chosen_centers, pid=best_pid,
                per_x=per_x, per_y=per_y, fallback_cyx=(cy, cx)
            )
            cw = peak.get("child_weights")
            if isinstance(cw, dict) and cw:
                # weighted replace with EMA or init uniform over gkids if empty
                replaced = update_parent_weights(
                    P, cw,
                    replace_thr=float(getattr(config, "parent_match_childset_replace_thr", 0.75)),
                    weight_ema=float(getattr(config, "parent_child_weight_ema", 0.20)),
                )
                if not replaced and not getattr(P, "child_weights", {}):
                    P.child_ids = tuple(sorted(gkids))
                    P.child_weights = {int(cid): 1.0 / len(gkids) for cid in gkids}
            else:
                P.child_ids = tuple(sorted(gkids))
                P.child_weights = {int(cid): 1.0 / len(gkids) for cid in gkids}

            P._region_proposal = prop
            P.emerged = True
            exists_support[best_pid] = (P.mask > support_tau)
            exists_kids[best_pid]    = set(P.child_ids)
            updated += 1
            if bool(getattr(config, "debug_spawn_log_details", True)):
                print(f"  -> MATCH pid={best_pid} iou={best_sc[0]:.3f} "
                      f"jacc={(best_sc[1]-1.0) if best_sc[1] > 1 else best_sc[1]:.3f} "
                      f"seed_px={seed_px}")
            continue

        # 6) SPAWN new (respect spacing and kid-set duplicates)
        if not respect_spawn_spacing(
            cy, cx, gkids, chosen_centers, exists_kids,
            H, W, per_x, per_y,
            min_interseed_px=min_interseed_px,
            jac_thr=float(getattr(config, "spacing_child_jaccard_thr", 0.90)),
            subset_thr=float(getattr(config, "spacing_child_subset_thr", 0.90)),
        ):
            if bool(getattr(config, "debug_spawn_log_details", True)):
                print("  -> DROP (spacing/child-set near-duplicate)")
            dropped_spacing += 1
            continue

        pid = int(next_parent_id); next_parent_id += 1

        # ctx-aware spawn (threads CacheHub via _RC)
        P, supp = _spawn_parent(
            _RC, agents, pid=pid, H=H, W=W, Kq=Kq, Kp=Kp, Gq=Gq, Gp=Gp,
            prop=prop, dtype=dtype, support_tau=support_tau,
            gkids=gkids, peak_child_weights=peak.get("child_weights"),
            step_now=step_now, per_x=per_x, per_y=per_y, chosen_centers=chosen_centers
        )

        # register new parent locally
        parents_by_id[pid] = P
        if not assert_parent_invariants(P, H, W):
            raise RuntimeError(f"New parent {pid} failed invariants at spawn")
        exists_support[pid] = supp
        exists_kids[pid]    = set(P.child_ids)
        spawned += 1
        new_spawned += 1

    if bool(getattr(config, "debug_spawn_log", False)):
        print(f"[SPAWN] updated={updated} spawned={spawned} "
              f"d_local={dropped_local} d_space={dropped_spacing} "
              f"d_childov={dropped_childov} d_seed={dropped_seed} d_budget={dropped_budget}")
    return next_parent_id




def run_cg_for_parents(
    ctx,
    cg_items,
    agents,
    Gq,
    Gp,
    *,
    eps,
    mask_tau,
    backend,
    n_jobs,
    batch_size,
    prefer=None,
    omp_thr=1,
):
    """
    Fan-out coarse-graining jobs over parents in `cg_items`.

    - `backend="threading"` mutates parents in-place.
    - `backend="loky"` returns updates from worker shells and applies them here.

    Notes:
      * `ctx` is explicitly threaded into each worker (no implicit globals).
      * We gate BLAS thread counts for 'loky' to avoid oversubscription.
    """
    # oversub throttle for processes
    if backend == "loky":
        thr = str(int(omp_thr))
        os.environ.setdefault("OMP_NUM_THREADS", thr)
        os.environ.setdefault("OPENBLAS_NUM_THREADS", thr)
        os.environ.setdefault("MKL_NUM_THREADS", thr)
        os.environ.setdefault("NUMEXPR_NUM_THREADS", thr)

    parallel_kwargs = dict(n_jobs=n_jobs, backend=backend, batch_size=batch_size)
    if prefer:
        parallel_kwargs["prefer"] = prefer

    results = Parallel(**parallel_kwargs)(
        delayed(_cg_worker_apply_or_prepare)(
            P,
            agents,                     # children
            eps=float(eps),
            mask_tau=float(mask_tau),
            Gq=Gq,
            Gp=Gp,
            backend=backend,
            ctx=ctx,                    # <-- pass ctx explicitly
        )
        for P in cg_items
    )

    cg_ok, cg_fail = 0, 0
    for (status, pid, payload), P in zip(results, cg_items):
        if status == "ok":
            if backend == "loky" and isinstance(payload, dict):
                for name, val in payload.items():
                    if val is not None:
                        setattr(P, name, val)

            if getattr(config, "debug_strict", True):
                for name in ("sigma_q_field", "sigma_p_field"):
                    S = getattr(P, name, None)
                    if S is not None:
                        S = np.asarray(S)
                        assert (
                            S.ndim == 4 and S.shape[-1] == S.shape[-2]
                        ), f"[CG->ASSIGN] {name} bad shape {S.shape} for parent {getattr(P,'id','?')}"

            setattr(P, "morphisms_dirty", True)
            if hasattr(P, "_pending_core_hash"):
                P._last_cg_core_hash = P._pending_core_hash
                delattr(P, "_pending_core_hash")
            setattr(P, "_needs_cg", False)
            cg_ok += 1
        else:
            print(f"[WARN] coarsegrain/init failed for parent {pid}: {payload}")
            cg_fail += 1

    return cg_ok, cg_fail





# ---------- CG worker (ctx-aware) ----------
def _cg_worker_apply_or_prepare(
    P,
    children,
    *,
    eps,
    mask_tau,
    Gq,
    Gp,
    backend,
    ctx=None,
):
    """
    Wrapper for coarsegrain_parent_from_children.
      threading: mutate P in place, return ("ok", pid, None)
      loky:      compute into a shell, return ("ok", pid, updates_dict)
    """
    try:
        if ctx is None:
            ctx = getattr(P, "ctx", None)

        if backend == "threading":
            coarsegrain_parent_from_children(
                P, children, eps=eps, mask_tau=mask_tau,
                G_q=Gq, G_p=Gp, ctx=ctx
            )
            return ("ok", int(getattr(P, "id", -1)), None)

        # loky path: compute on a lightweight shell; send arrays back
        class _PShell: ...
        Ps = _PShell()
        for name in ("id","mask","mu_q_field","sigma_q_field","mu_p_field","sigma_p_field","phi","phi_model"):
            setattr(Ps, name, getattr(P, name, None))

        coarsegrain_parent_from_children(
            Ps, children, eps=eps, mask_tau=mask_tau,
            G_q=Gq, G_p=Gp, ctx=ctx
        )

        updates = {k: getattr(Ps, k, None)
                   for k in ("mu_q_field","sigma_q_field","mu_p_field","sigma_p_field","phi","phi_model")}
        return ("ok", int(getattr(P, "id", -1)), updates)

    except Exception as e:
        return ("fail", int(getattr(P, "id", -1)), f"{e}")

