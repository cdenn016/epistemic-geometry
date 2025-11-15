# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 16:38:18 2025

@author: chris and christine
"""

from __future__ import annotations
from typing import Dict, List, Iterable
import numpy as np
import core.config as config
from parents.parent_utils import cc_label  # toroidal CC

from core.gaussian_core import resolve_support_tau
from parents.parent_utils import conservative_child_set_from_seed



# ─────────────────────────────
# Main reconciler (refactored)
# ─────────────────────────────

def reconcile_parent_masks_and_assignments(runtime_ctx, children, step: int, *, periodic: bool, H: int, W: int, level: int = 1) -> None:
    """
    Enforce:
      (1) per-step child assignment by IoU with hysteresis,
      (2) seed vs active masks (freeze),
      (3) coalition-aware growth (strong->weak),
      (4) continuity-first CC under PBC,
      (5) soft-mask dynamics with step cap,
      (6) hard gate to prevent global 'glow' outside multi-child support,
      (7) publish a permissive cg_support for coarse-graining.
    """
    # Level-aware registry (fallback to legacy mirror only if needed)
    try:
        reg, _ = runtime_ctx.get_parent_registry(level)
        parents: Dict[int, object] = (reg or {})
    except Exception:
        parents: Dict[int, object] = (getattr(runtime_ctx, "parents_by_region", {}) or {})
    if not parents:
        return

    # knobs
    active_thr_cfg   = getattr(config, "parent_active_level", None)  # None => auto
    active_thr_q     = float(getattr(config, "parent_active_level_q", 60.0))
    active_thr_rel   = float(getattr(config, "parent_active_level_rel", 0.60))
    active_thr_min   = float(getattr(config, "parent_active_level_min", 0.06))
    add_iou_thr      = float(getattr(config, "parent_assign_iou_add_thr", 0.15))
    keep_iou_thr     = min(float(getattr(config, "parent_assign_iou_keep_thr", 0.05)), 0.75 * add_iou_thr)
    min_overlap_px   = int(getattr(config, "parent_assign_min_overlap_px", 8))
    orphan_grace     = int(getattr(config, "parent_orphan_grace_steps", 10))
    freeze_steps     = int(getattr(config, "parent_freeze_steps", 3))
    seed_erode_iters = int(getattr(config, "parent_seed_erode_iters", 1))
    min_cc_px_base   = int(getattr(config, "parent_active_min_cc_px", 8))
    active_iou_min   = float(getattr(config, "parent_active_iou_min", 0.30))
    hysteresis_keep  = float(getattr(config, "parent_hysteresis_keep_frac", 0.75))

    cover_N   = int(getattr(config, "coalition_cover_n", 2))
    coal_hi   = float(getattr(config, "coalition_high_tau", 0.18))
    coal_lo   = float(getattr(config, "coalition_low_tau", 0.06))
    agree_map = getattr(runtime_ctx, "Ap_agg" if bool(getattr(config, "model_only", False)) else "Aq_agg", None)

    
    support_tau = resolve_support_tau(runtime_ctx, None, default=1e-3, name="support_tau")
    child_tau_coal = max(float(getattr(config, "parent_coalition_tau", support_tau)), 0.08)


    # precompute per-child interior masks (robust IoUs)
    child_ids = [int(getattr(c, "id", i)) for i, c in enumerate(children)]
    child_bool_cache = {}
    for i, cid in enumerate(child_ids):
        cm = _to_bool(children[i].mask, active_thr_min)
        cm = _erode4(cm, iters=int(getattr(config, "assign_child_erode_iters", 1)), periodic=periodic)
        child_bool_cache[cid] = cm

    # cover count ≥N using permissive tau
    child_tau = float(getattr(config, "support_tau", 0.01))
    try:
        M_stack = np.stack([(np.asarray(c.mask, np.float32) >= child_tau).astype(np.uint8) for c in children], axis=0)
        cover_count = M_stack.sum(axis=0)
    except Exception:
        cover_count = None

    # optional coalition growth map
    coalition = None
    if isinstance(agree_map, np.ndarray) and (cover_count is not None):
        strong = (cover_count >= cover_N) & (agree_map >= coal_hi)
        weak   = (cover_count >= cover_N) & (agree_map >= coal_lo)
        if weak.any() or strong.any():
            coalition = _grow8(weak, strong)

    for pid, P in list(parents.items()):
        # init persistent fields
        if not hasattr(P, "active_mask"):
            P.active_mask = _to_bool(P.mask, active_thr_min)
        else:
            P.active_mask = _to_bool(P.active_mask, active_thr_min)

        if not hasattr(P, "seed_mask") or P.seed_mask is None or not np.any(P.seed_mask):
            seed0 = _erode4(P.active_mask, seed_erode_iters, periodic=periodic)
            if not seed0.any():
                seed0 = P.active_mask.copy()
            P.seed_mask = seed0.astype(np.float32)

        if not hasattr(P, "assigned_child_ids") or P.assigned_child_ids is None:
            P.assigned_child_ids = set()

        if not hasattr(P, "freeze_until"):
            P.freeze_until = int(step) + freeze_steps

        if not hasattr(P, "orphan_steps"):
            P.orphan_steps = 0

        # adaptive threshold + active area
        if active_thr_cfg is None:
            active_thr = _adaptive_active_threshold(P, q_pct=active_thr_q, rel=active_thr_rel, min_thr=active_thr_min)
        else:
            P._active_thr_prev = float(active_thr_cfg)
            active_thr = float(active_thr_cfg)

        active_bool = _to_bool(P.mask, active_thr)
        active_area = int(active_bool.sum())
        min_cc_px   = max(min_cc_px_base, int(0.04 * max(1, active_area)))

        # assignment (hysteresis)
        _assign_children_hysteresis(P, child_bool_cache, active_bool=active_bool,
                                    add_iou_thr=add_iou_thr, keep_iou_thr=keep_iou_thr,
                                    min_overlap_px=min_overlap_px)

        # bootstrap assignments if empty during freeze
        if (len(P.assigned_child_ids) == 0) and (step <= int(P.freeze_until)):
            try:
                
                boot = conservative_child_set_from_seed(children,
                                                        _to_bool(P.seed_mask, 0.5),
                                                        mask_tau=active_thr_min,
                                                        min_overlap_px=min_overlap_px)
                P.assigned_child_ids = set(int(x) for x in boot) or set()
            except Exception:
                pass

        # orphan detection post-freeze
        if (len(P.assigned_child_ids) == 0) and (active_area < min_cc_px) and (int(step) > int(P.freeze_until)):
            P.orphan_steps += 1
        else:
            P.orphan_steps = 0
        if P.orphan_steps >= orphan_grace:
            setattr(P, "_delete", True)
            continue

        # pick continuity-first target CC (if assignments exist, bias to their union)
        if len(P.assigned_child_ids) > 0:
            union_assigned = _union_of_children(children, P.assigned_child_ids, H, W, active_thr_min)
            labels, nlab = cc_label(active_bool, periodic=bool(periodic))
            if nlab > 1:
                target_bool = _choose_target_cc(P, active_bool, union_assigned,
                                                min_cc_px=min_cc_px, periodic=periodic)
            else:
                target_bool = active_bool

            # IoU gate vs assigned union + smart restore
            iou_u = _iou(target_bool, union_assigned)
            if iou_u < active_iou_min:
                prev = target_bool.copy()
                target_bool = target_bool & union_assigned
                dropped = prev & (~target_bool)
                if dropped.any():
                    restore_cap = int(np.ceil((1.0 - hysteresis_keep) * dropped.sum()))
                    if restore_cap > 0:
                        M_prev = np.asarray(P.mask, np.float32)
                        yy, xx = np.where(dropped)
                        vals = M_prev[yy, xx]
                        idx_sorted = np.argsort(-vals)[:restore_cap]
                        rr = np.zeros_like(prev, dtype=bool)
                        rr[yy[idx_sorted], xx[idx_sorted]] = True
                        target_bool |= rr
        else:
            target_bool = _largest_cc(active_bool, periodic=periodic) if active_area >= min_cc_px else active_bool

        # merge coalition CC that touches current/seed (natural growth)
        if isinstance(coalition, np.ndarray) and coalition.any():
            anchor = target_bool | _to_bool(P.seed_mask, 0.5)
            lab, ncoal = cc_label(coalition, periodic=bool(periodic))
            best, score = 0, -1
            for lbl in range(1, ncoal + 1):
                comp = (lab == lbl)
                s = int((comp & anchor).sum())
                if s > score:
                    score = s; best = lbl
            if best > 0:
                target_bool |= (lab == best)

        # freeze seed protection
        if step <= int(P.freeze_until):
            target_bool |= _to_bool(P.seed_mask, 0.5)

        # soft-update + hard gate (anti-glow)
        M_new = _apply_soft_update(P, target_bool, periodic=periodic)
        if cover_count is not None:
            M_new = _hard_gate_mask(P, M_new, children, cover_count, H=H, W=W, child_tau_coal=child_tau_coal)

        # always publish a permissive CG support for coarse-graining
        _publish_cg_support(P, children, cover_count, H=H, W=W)

        # minimal area safeguard during freeze
        if (M_new >= active_thr).sum() < min_cc_px and step <= int(P.freeze_until):
            M_new = np.maximum(M_new, np.asarray(P.seed_mask, np.float32))

        # commit only if meaningful change
        commit_eps = float(getattr(config, "parent_commit_eps", 1e-4))
        M_prev = np.asarray(P.mask, np.float32)
        if float(np.max(np.abs(M_new - M_prev))) >= commit_eps:
            P.mask = M_new.astype(np.float32)
        else:
            P.mask = M_prev.astype(np.float32)

        P.active_mask = (P.mask >= active_thr)










def update_parent_lifecycle(runtime_ctx, step: int, *, level: int = 1) -> int:
    """
    Bookkeeping only:
      - newborn freeze window
      - orphan detection (no assigned kids + tiny active area)
      - mark for deletion via P._delete
    Returns the number of parents marked for deletion this call.
    """
    

    # Level-aware registry (fallback to legacy only if needed)
    try:
        reg, _ = runtime_ctx.get_parent_registry(level)
        parents = (reg or {})
    except Exception:
        parents = getattr(runtime_ctx, "parents_by_region", {}) or {}
    if not parents:
        return 0

    # knobs (single-sourced; keep them few and obvious)
    active_thr_min     = float(getattr(config, "parent_active_level_min", 0.06))
    min_cc_px_base     = int(getattr(config, "parent_active_min_cc_px", 8))
    orphan_grace_steps = int(getattr(config, "parent_orphan_grace_steps", 10))
    freeze_steps       = int(getattr(config, "parent_freeze_steps", 2))

    deleted = 0
    for pid, P in list(parents.items()):
        # init lifecycle fields
        if not hasattr(P, "freeze_until"):
            P.freeze_until = int(step) + freeze_steps
        if not hasattr(P, "orphan_steps"):
            P.orphan_steps = 0
        if not hasattr(P, "assigned_child_ids") or P.assigned_child_ids is None:
            P.assigned_child_ids = set()

        # compute a simple "active" area from the current mask
        M = np.asarray(getattr(P, "mask", 0.0), np.float32)
        active_bool  = (M >= active_thr_min)
        active_area  = int(active_bool.sum())
        # scale min-cc with size (4% of area, clamped by base)
        min_cc_px    = max(min_cc_px_base, int(0.04 * max(1, active_area)))

        # orphan rule after newborn freeze
        if (int(step) > int(P.freeze_until)) and (len(P.assigned_child_ids) == 0) and (active_area < min_cc_px):
            P.orphan_steps += 1
        else:
            P.orphan_steps = 0

        if P.orphan_steps >= orphan_grace_steps:
            setattr(P, "_delete", True)
            deleted += 1

    return deleted










# ─────────────────────────────
# Small, file-local helpers
# ─────────────────────────────



def _to_bool(m, thr) -> np.ndarray:
    return (np.asarray(m, np.float32) >= float(thr))



def _iou(a, b) -> float:
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    inter = int((a & b).sum()); union = int(a.sum() + b.sum() - inter)
    return inter / max(1, union)



def _erode4(mask: np.ndarray, iters: int = 1, periodic: bool = False) -> np.ndarray:
    m = np.asarray(mask, bool)
    for _ in range(max(0, int(iters))):
        nbr = (np.roll(m,1,0) & np.roll(m,-1,0) & np.roll(m,1,1) & np.roll(m,-1,1))
        if not periodic:
            nbr[0,:]=False; nbr[-1,:]=False; nbr[:,0]=False; nbr[:,-1]=False
        m = m & nbr
    return m



def _grow8(weak: np.ndarray, strong: np.ndarray) -> np.ndarray:
    weak   = np.asarray(weak, bool)
    strong = np.asarray(strong, bool)
    act = strong.copy()
    while True:
        nbr = (np.roll(act,1,0)|np.roll(act,-1,0)|np.roll(act,1,1)|np.roll(act,-1,1)|
               np.roll(np.roll(act,1,0),1,1)|np.roll(np.roll(act,1,0),-1,1)|
               np.roll(np.roll(act,-1,0),1,1)|np.roll(np.roll(act,-1,0),-1,1))
        add = weak & (~act) & nbr
        if not add.any(): break
        act |= add
    return act



def _largest_cc(mask_bool: np.ndarray, *, periodic: bool) -> np.ndarray:
    labels, n = cc_label(mask_bool, periodic=bool(periodic))
    if n <= 1:
        return np.asarray(mask_bool, bool)
    best = 0
    best_area = -1
    for k in range(1, n + 1):
        area = int((labels == k).sum())
        if area > best_area:
            best_area = area; best = k
    return (labels == best)



def _union_of_children(children: Iterable[object], cids: Iterable[int], H: int, W: int, thr: float) -> np.ndarray:
    u = np.zeros((H, W), bool)
    want = set(int(x) for x in (cids or []))
    if not want:
        return u
    for c in children:
        cid = int(getattr(c, "id", -1))
        if cid in want:
            u |= _to_bool(getattr(c, "mask", 0.0), thr)
    return u



def _adaptive_active_threshold(P, *, q_pct: float, rel: float, min_thr: float) -> float:
    vals = np.asarray(getattr(P, "mask", 0.0), np.float32)
    vals = vals[vals > 1e-6]
    if vals.size:
        q = float(np.percentile(vals, q_pct))
        raw_thr = max(min_thr, rel * q)
    else:
        raw_thr = min_thr
    beta = float(getattr(config, "parent_active_thr_beta", 0.30))
    prev = float(getattr(P, "_active_thr_prev", raw_thr))
    thr  = (1.0 - beta) * prev + beta * raw_thr
    P._active_thr_prev = float(thr)
    return float(thr)



def _assign_children_hysteresis(P, child_bool_cache: Dict[int, np.ndarray],
                                *, active_bool: np.ndarray,
                                add_iou_thr: float, keep_iou_thr: float,
                                min_overlap_px: int) -> None:
    new_assign = set()
    for cid, cm in child_bool_cache.items():
        if cm is None or not cm.any(): continue
        inter = int((active_bool & cm).sum())
        if inter >= min_overlap_px and _iou(active_bool, cm) >= add_iou_thr:
            new_assign.add(cid)

    kept = set()
    for cid in list(getattr(P, "assigned_child_ids", set())):
        cm = child_bool_cache.get(cid)
        if cm is None or not cm.any(): continue
        inter = int((active_bool & cm).sum())
        if inter >= min_overlap_px and _iou(active_bool, cm) >= keep_iou_thr:
            kept.add(cid)

    P.assigned_child_ids = kept | new_assign



def _choose_target_cc(P, active_bool: np.ndarray, union_assigned: np.ndarray,
                      *, min_cc_px: int, periodic: bool) -> np.ndarray:
    labels, nlab = cc_label(active_bool, periodic=bool(periodic))
    if nlab <= 1:
        return active_bool
    prev_anchor = active_bool | _to_bool(getattr(P, "seed_mask", 0.0), 0.5)
    best_lbl, best_score = 0, (-1, -1, -1)
    for lbl in range(1, nlab + 1):
        comp = (labels == lbl)
        if comp.sum() < min_cc_px: continue
        ov_u = int((comp & union_assigned).sum())
        ov_p = int((comp & prev_anchor).sum())
        score = (ov_u, ov_p, int(comp.sum()))
        if score > best_score:
            best_score, best_lbl = score, lbl

    margin_u = int(getattr(config, "cc_switch_margin_u", 3))
    persist  = int(getattr(config, "cc_switch_persist", 2))
    last_lbl   = int(getattr(P, "_last_cc_lbl", 0))
    

    if last_lbl in range(1, nlab + 1):
        comp_last = (labels == last_lbl)
        ov_u_last = int((comp_last & union_assigned).sum())
        if (best_score[0] < ov_u_last + margin_u):
            best_lbl, best_score = last_lbl, (ov_u_last,
                                              int((comp_last & prev_anchor).sum()),
                                              int(comp_last.sum()))
            P._cc_hold = 0
        else:
            P._cc_hold = int(getattr(P, "_cc_hold", 0)) + 1
            if P._cc_hold < persist:
                best_lbl, best_score = last_lbl, (ov_u_last,
                                                  int((comp_last & prev_anchor).sum()),
                                                  int(comp_last.sum()))
            else:
                P._cc_hold = 0

    P._last_cc_lbl   = int(best_lbl)
    P._last_cc_score = tuple(best_score)
    return (labels == best_lbl) if best_lbl != 0 else _largest_cc(active_bool, periodic=periodic)



def _apply_soft_update(P, target_bool: np.ndarray, *, periodic: bool) -> np.ndarray:
    sigma     = float(getattr(config, "parent_feather_sigma", 0.8))
    eta_up    = float(getattr(config, "parent_eta_up", 0.20))
    eta_down  = float(getattr(config, "parent_eta_down", 0.15))
    delta_cap = float(getattr(config, "parent_delta_cap", 0.12))

    target_f = target_bool.astype(np.float32)
    if sigma > 0.0:
        try:
            from scipy.ndimage import gaussian_filter as _gf
            target_f = _gf(target_f, sigma=sigma, mode="wrap" if periodic else "nearest")
        except Exception:
            pass

    target_f = np.clip(target_f, 0.0, 1.0)
    M_prev = np.asarray(getattr(P, "mask", 0.0), np.float32)
    delta  = target_f - M_prev
    eta    = np.where(delta > 0.0, eta_up, np.where(delta < 0.0, eta_down, 0.0)).astype(np.float32)
    raw    = M_prev + eta * delta
    d      = np.clip(raw - M_prev, -delta_cap, delta_cap)
    return np.clip(M_prev + d, 0.0, 1.0)




def _hard_gate_mask(P, M_new: np.ndarray, children: Iterable[object], cover_count: np.ndarray,
                    *, H: int, W: int, child_tau_coal: float) -> np.ndarray:
    # require ≥2-cover; if assigned kids exist, gate to their union; else allow seed halo
    cover_ge2 = (cover_count >= 2) if cover_count is not None else None
    if cover_ge2 is None:
        return M_new

    assigned = getattr(P, "assigned_child_ids", set()) or set()
    if len(assigned) > 0:
        u_assigned = _union_of_children(children, assigned, H, W, child_tau_coal)
        hard_gate = (cover_ge2 & u_assigned).astype(np.float32)
    else:
        seed = (_to_bool(getattr(P, "seed_mask", 0.0), 0.5))
        halo = seed.copy()
        for _ in range(2):
            nbr = ( np.roll(halo,1,0) | np.roll(halo,-1,0) |
                    np.roll(halo,1,1) | np.roll(halo,-1,1) |
                    np.roll(np.roll(halo,1,0),1,1) | np.roll(np.roll(halo,1,0),-1,1) |
                    np.roll(np.roll(halo,-1,0),1,1) | np.roll(np.roll(halo,-1,0),-1,1) )
            halo |= nbr
        hard_gate = (cover_ge2 & halo).astype(np.float32)

    out = np.asarray(M_new, np.float32) * hard_gate
    out[out < 1e-5] = 0.0
    return out




def _publish_cg_support(P, children: Iterable[object], cover_count: np.ndarray, *,
                        H: int, W: int) -> None:
    """
    Publish a permissive support for coarse-graining:
      - union(assigned) @ support_tau
      - relaxed seed halo when unassigned
      - include current mask mass > support_tau

    Uses np.maximum for float unions (no bitwise ops on floats).
    """
    
    try:
        
        cg_child_tau = resolve_support_tau(None, None, default=0.01, name="support_tau")

        assigned = getattr(P, "assigned_child_ids", set()) or set()

        # union(assigned) at permissive tau (boolean)
        if len(assigned) > 0:
            u_assigned_perm = _union_of_children(children, assigned, H, W, cg_child_tau)  # bool
        else:
            u_assigned_perm = None

        # small seed halo (boolean)
        seed_bool = _to_bool(getattr(P, "seed_mask", 0.0), 0.5)
        halo = seed_bool.copy()
        for _ in range(int(getattr(config, "cg_seed_halo_iters", 2))):
            nbr = ( np.roll(halo,1,0) | np.roll(halo,-1,0) |
                    np.roll(halo,1,1) | np.roll(halo,-1,1) |
                    np.roll(np.roll(halo,1,0),1,1) | np.roll(np.roll(halo,1,0),-1,1) |
                    np.roll(np.roll(halo,-1,0),1,1) | np.roll(np.roll(halo,-1,0),-1,1) )
            halo |= nbr  # boolean OR

        # cover thresholds (booleans or None)
        need_norm = int(getattr(config, "coalition_cover_n", 2))
        need_boot = int(getattr(config, "cg_bootstrap_cover_n", 1))
        need_boot = max(1, min(need_boot, need_norm))

        cov_norm = (cover_count >= need_norm) if (cover_count is not None) else None
        cov_boot = (cover_count >= need_boot) if (cover_count is not None) else None

        # float32 accumulator; use np.maximum for unions
        cg_support = np.zeros((H, W), np.float32)

        if (u_assigned_perm is not None) and (cov_norm is not None):
            term = (u_assigned_perm & cov_norm).astype(np.float32)
            cg_support = np.maximum(cg_support, term)
        elif (u_assigned_perm is not None):
            cg_support = np.maximum(cg_support, u_assigned_perm.astype(np.float32))

        if len(assigned) == 0:
            if cov_boot is not None:
                term = (halo & cov_boot).astype(np.float32)
                cg_support = np.maximum(cg_support, term)
            else:
                cg_support = np.maximum(cg_support, halo.astype(np.float32))

        # include current parent mask mass > support_tau
        cg_support = np.maximum(cg_support, _to_bool(getattr(P, "mask", 0.0), cg_child_tau).astype(np.float32))

        P.cg_support = np.clip(cg_support, 0.0, 1.0)
    except Exception:
        # fallback: permissive threshold on current mask
        P.cg_support = (_to_bool(getattr(P, "mask", 0.0), float(getattr(config, "support_tau", 0.01)))).astype(np.float32)

















