# --- top of file -------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
Utilities for parent/child assignment, mask dynamics, spawn caches, and lifecycle.
Refactor: consolidated knobs, deleted legacy paths, added invariants & typing.
"""
import numpy as np
import core.config as config



# -------------------------- Main function  --------------------------



def build_alignment_aggregates(runtime_ctx, children, *, kl_cap=10000000.0):
    """
    Writes these onto runtime_ctx (H×W float32 each):
        - Aq_agg, KLq_agg  (agreement/KL from belief alignment)
        - Ap_agg, KLp_agg  (agreement/KL from model  alignment)

    Robust to missing caches and avoids any ambiguous array truth checks.
    """
    

    # ---------------- config ----------------
    try:
       
        tau_q = float(getattr(config, "tau_align_q", 0.45))
        tau_p = float(getattr(config, "tau_align_p", 0.45))
        kl_cap = float(getattr(config, "signal_kl_cap", kl_cap))
        reduce_mode = str(getattr(config, "agg_reduce_mode", "p95")).lower()  # "mean"|"max"|"p95"
        prefer_agreement_cache = bool(getattr(config, "agg_prefer_agreement_cache", True))
        ema = float(getattr(config, "agg_map_ema", 0.25))  # 0..1; weight for NEW value
    except Exception:
        tau_q = tau_p = 0.45
        reduce_mode = "mean"
        prefer_agreement_cache = True
        ema = 0.25

    eps = 1e-20

    # ---------------- shapes ----------------
    if not children or len(children) == 0:
        # nothing to do
        return

    
    H = getattr(runtime_ctx, "H", None)
    W = getattr(runtime_ctx, "W", None)
    for ch in children:
        if ch is None:
            continue
        m = getattr(ch, "mask", None)
        if isinstance(m, np.ndarray) and m.ndim >= 2:
            H = H if H is not None else m.shape[0]
            W = W if W is not None else m.shape[1]
            if (H is not None) and (W is not None):
        
                break
    if (H is None) or (W is None):
        return  # no valid shapes available

    # local crop/pad helper (renamed to avoid clashes with 3-arg version elsewhere)
    def _resize_hw_local(a):
        a = np.asarray(a, np.float32)
        if a.shape[:2] == (H, W):
            return a
        out = np.zeros((H, W), np.float32)
        h = min(H, a.shape[0]); w = min(W, a.shape[1])
        out[:h, :w] = a[:h, :w]
        return out

    # ---------------- collect per-child maps ----------------
    Aq_list, KLq_list = [], []
    Ap_list, KLp_list = [], []

    for ch in children:
        if ch is None:
            continue

        # Prefer cached agreement (already ∈ [0,1]) if configured
        A_q = getattr(ch, "spawn_A_final", None) if prefer_agreement_cache else None
        if isinstance(A_q, np.ndarray):
            Aq_list.append(np.clip(_resize_hw_local(A_q), 0.0, 1.0))
        else:
            # otherwise try KL caches for q
            KLq = getattr(ch, "spawn_softmin_kl_q", None)
            if KLq is None:
                KLq = getattr(ch, "cached_alignment_kl", None)
            if isinstance(KLq, np.ndarray):
                KLq_list.append(np.clip(_resize_hw_local(KLq), 0.0, float(kl_cap)))

        # Only build Ap if model-side maps are actually used (saves compute)
        if bool(getattr(config, "build_model_agreement_maps", False)):
            A_p = getattr(ch, "spawn_A_final_p", None) if prefer_agreement_cache else None
            if isinstance(A_p, np.ndarray):
                Ap_list.append(np.clip(_resize_hw_local(A_p), 0.0, 1.0))
            else:
                KLp = getattr(ch, "spawn_softmin_kl_p", None)
                if KLp is None:
                    KLp = getattr(ch, "cached_model_alignment_kl", None)
                if isinstance(KLp, np.ndarray):
                    KLp_list.append(np.clip(_resize_hw_local(KLp), 0.0, float(kl_cap)))

    # ---------------- reduction helpers (NO ambiguous truth) ----------------
    def _reduce_agree(A_list, KL_list, tau):
        has_A = isinstance(A_list, list) and (len(A_list) > 0)
        has_K = isinstance(KL_list, list) and (len(KL_list) > 0)

        if has_A:
            A_stack = np.stack(A_list, axis=0)  # (N,H,W)
        elif has_K:
            KL_stack = np.stack(KL_list, axis=0)  # (N,H,W)
            A_stack = np.exp(-KL_stack / max(tau, 1e-8)).astype(np.float32)
        else:
            return None, None

        if reduce_mode == "mean":
            A_new = np.mean(A_stack, axis=0)
        elif reduce_mode == "max":
            A_new = np.max(A_stack, axis=0)
        elif reduce_mode in ("p95", "p-95", "q95"):
            A_new = np.percentile(A_stack, 95.0, axis=0)
        else:
            A_new = np.mean(A_stack, axis=0)

        A_new = np.clip(np.nan_to_num(A_new, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        KL_new = (-max(tau, 1e-8) * np.log(np.maximum(A_new, eps))).astype(np.float32)
        return A_new.astype(np.float32), KL_new

    Aq_new, KLq_new = _reduce_agree(Aq_list, KLq_list, tau_q)
    Ap_new, KLp_new = _reduce_agree(Ap_list, KLp_list, tau_p)

    # ---------------- EMA smoothing (maps) ----------------
    if (ema > 0.0) and (ema < 1.0):
        Aq_prev = getattr(runtime_ctx, "Aq_agg", None)
        if (Aq_prev is not None) and (Aq_new is not None):
            Aq_prev = _resize_hw_local(Aq_prev)
            Aq_new = (1.0 - float(ema)) * Aq_prev + float(ema) * Aq_new
            KLq_new = (-max(tau_q, 1e-8) * np.log(np.maximum(Aq_new, eps))).astype(np.float32)

        Ap_prev = getattr(runtime_ctx, "Ap_agg", None)
        if (Ap_prev is not None) and (Ap_new is not None):
            Ap_prev = _resize_hw_local(Ap_prev)
            Ap_new = (1.0 - float(ema)) * Ap_prev + float(ema) * Ap_new
            KLp_new = (-max(tau_p, 1e-8) * np.log(np.maximum(Ap_new, eps))).astype(np.float32)

    # ---------------- write to ctx ----------------
    if Aq_new is not None:
        setattr(runtime_ctx, "Aq_agg", Aq_new)
        setattr(runtime_ctx, "KLq_agg", KLq_new)
    if Ap_new is not None:
        setattr(runtime_ctx, "Ap_agg", Ap_new)
        setattr(runtime_ctx, "KLp_agg", KLp_new)

#=============================================================================#


def compute_and_set_auto_agree_gate(runtime_ctx, *, percentile=65.0, ema=0.2, use_p_if_q_missing=True):
    """
    Sets runtime_ctx.detector_state.agree_gate_tau_eff to a smoothed percentile
    of A_agg in [0,1]. If no aggregates exist, sets it to None (gate off).
    """
    
    ds = getattr(runtime_ctx, "detector_state", None)
    if ds is None:
        return

    A = getattr(runtime_ctx, "Aq_agg", None)
    if (A is None) and use_p_if_q_missing:
        A = getattr(runtime_ctx, "Ap_agg", None)

    if A is None:
        ds.agree_gate_tau_eff = None
        try:
            print("[AGG] auto agree gate: no aggregates, gate OFF")
        except Exception:
            pass
        return

    A = np.asarray(A, np.float32)
    # focus percentile on *supported* pixels so background zeros don't depress the gate
    support_tau = float(getattr(getattr(runtime_ctx, "config", None) or __import__("config"), "support_tau", 1e-2))
    cover = None
    try:
        kids = getattr(runtime_ctx, "children_latest", None)
        if kids:
            stack = np.stack([(np.asarray(k.mask, np.float32) > support_tau).astype(np.uint8) for k in kids], axis=0)
            cover = (stack.sum(axis=0) > 0)
    except Exception:
        cover = None

    good = np.isfinite(A)
    if cover is not None and cover.shape == A.shape:
        good = good & cover
    if not good.any():
        ds.agree_gate_tau_eff = None
        try:
            print("[AGG] auto agree gate: aggregates non-finite, gate OFF")
        except Exception:
            pass
        return

    thr = float(np.percentile(A[good], float(percentile)))
    thr = float(np.clip(thr, 0.0, 1.0))
    prev = getattr(ds, "agree_gate_tau_eff", None)
    if (prev is None) or (not np.isfinite(prev)):
        new_tau = thr
    else:
        new_tau = (1.0 - float(ema)) * float(prev) + float(ema) * thr

    ds.agree_gate_tau_eff = float(np.clip(new_tau, 0.0, 1.0))
    try:
        print(f"[AGG] auto agree gate p{int(percentile)} ema={ema} tau={new_tau:.3f}")
    except Exception:
        pass



# --- put this in parent_update_helpers.py (or anywhere you call it from) ---

def sync_child_parent_links_from_reconciler(children, parents_by_id):
    """
    Mirror reconciler assignments into the lightweight fields used downstream.
    - parent.child_ids from P.assigned_child_ids
    - parent.child_weights set to uniform (critical for CG mass)
    - child.parent_ids / child.parent_id set via reverse index
    """
    # parents → child_ids (+ uniform weights so CG has nonzero mass)
    for P in (parents_by_id or {}).values():
        assigned = getattr(P, "assigned_child_ids", None) or set()
        P.child_ids = tuple(int(cid) for cid in sorted(assigned))
        P.child_weights = {int(cid): 1.0 for cid in P.child_ids}

    # build reverse map
    id2parents = {}
    for pid, P in (parents_by_id or {}).items():
        for cid in getattr(P, "child_ids", ()):
            id2parents.setdefault(int(cid), []).append(int(pid))

    # children → parent_ids / parent_id
    for ch in (children or []):
        cid = int(getattr(ch, "id", -1))
        plist = id2parents.get(cid, [])
        ch.parent_ids = plist
        ch.parent_id  = (plist[0] if plist else -1)
        ch.labels = {}  # legacy compat




def build_child_parent_responsibilities(children, parents_by_id, *,
                                        mode="pixel",   # "pixel" or "global"
                                        eps=1e-6):
    """
    Compute child↔parent responsibilities so a child can belong to MULTIPLE parents.

    For each child c and parent P that lists c in P.child_ids:
      - mode="pixel":  r_{P|c}(y,x)  ∝  P.mask(y,x) over all such parents, normalized per-pixel.
      - mode="global": r_{P|c}       ∝  mean_{child-support} P.mask, normalized per-child (scalar).

    Effects (writes onto objects):
      child._parent_ids        = [pid,...]
      child._parent_resp_mask  = {pid: H×W float32}     (if mode="pixel")
      child._parent_resp       = {pid: float}           (if mode="global")
      parent._child_resp       = {cid: float}           (global scalar for diagnostics)
    """
    

    if not children or not parents_by_id:
        return

    # quick shape
    H, W = np.asarray(children[0].mask, np.float32).shape[:2]

    # build reverse index: parent -> set(child_ids) already exists via P.child_ids
    # we also want, for each child, the set of parents that include it
    child_to_parents = {int(getattr(ch, "id", -1)): [] for ch in children}
    for pid, P in (parents_by_id or {}).items():
        for cid in (getattr(P, "child_ids", []) or []):
            cid = int(cid)
            if cid in child_to_parents:
                child_to_parents[cid].append(int(pid))

    # prefetch parent masks
    pid2mask = {int(pid): np.asarray(getattr(P, "mask", np.zeros((H, W), np.float32)), np.float32)
                for pid, P in (parents_by_id or {}).items()}

    # config-ish knobs (fallbacks ok)
    try:
        
        min_px = int(getattr(config, "xscale_min_overlap_px", 4))
        temp   = float(getattr(config, "parent_resp_temp", 1.0))  # 1.0=no change; <1 sharpen, >1 soften
        support_tau = float(getattr(config, "support_tau", 1e-2))
    except Exception:
        min_px, temp, support_tau = 4, 1.0, 1e-2

    for ch in children:
        cid = int(getattr(ch, "id", -1))
        pids = child_to_parents.get(cid, [])
        setattr(ch, "_parent_ids", list(pids))

        if len(pids) == 0:
            setattr(ch, "_parent_resp_mask", {})
            setattr(ch, "_parent_resp", {})
            continue

        cmask = np.clip(np.asarray(getattr(ch, "mask", np.zeros((H, W), np.float32)), np.float32), 0.0, 1.0)

        if mode == "pixel":
            # stack parent masks and normalize per-pixel over parents that include this child
            stack = np.stack([pid2mask[int(pid)] for pid in pids], axis=0)  # (P,H,W)
            # restrict to child support to reduce speckle
            stack = stack * cmask[None, :, :]
            # drop parents with tiny overlap to avoid noisy links
            overlaps = (stack > support_tau).sum(axis=(1,2))
            keep_idx = np.where(overlaps >= max(1, min_px))[0]
            if keep_idx.size == 0:
                setattr(ch, "_parent_resp_mask", {}); setattr(ch, "_parent_resp", {}); continue
            stack = stack[keep_idx]; pids = [pids[i] for i in keep_idx]

            # optional temperature for crisper/softer splits
            if temp != 1.0:
                # avoid negatives then power-scale
                stack = np.clip(stack, 0.0, None) ** (1.0 / max(1e-6, float(temp)))
            denom = np.maximum(np.sum(stack, axis=0), eps)  # (H,W)
            resp_masks = [(stack[i] / denom).astype(np.float32) for i in range(len(pids))]

            resp_map = {int(pid): resp_masks[i] for i, pid in enumerate(pids)}
            setattr(ch, "_parent_resp_mask", resp_map)
            setattr(ch, "_parent_resp", {})

            # for diagnostics: global scalar responsibility per parent (mean over child support)
            for i, pid in enumerate(pids):
                mean_w = float(np.mean(resp_masks[i][cmask > 0.0])) if np.any(cmask > 0.0) else 0.0
                P = parents_by_id.get(int(pid))
                if P is not None:
                    d = getattr(P, "_child_resp", None) or {}
                    d[cid] = mean_w
                    setattr(P, "_child_resp", d)

        else:  # mode == "global"
            # global weight ∝ mean parent mask over the child's support
            weights = []
            for pid in pids:
                pm = pid2mask[int(pid)]
                if np.any(cmask > 0.0):
                    w = float(np.mean(pm[cmask > 0.0]))
                else:
                    w = 0.0
                

                weights.append(max(w, 0.0))
            wsum = float(sum(weights))
            if wsum <= 0.0:
                setattr(ch, "_parent_resp", {}); setattr(ch, "_parent_resp_mask", {}); continue
            # temperature (optional) for global too
            if temp != 1.0:
                weights = [w ** (1.0 / max(1e-6, float(temp))) for w in weights]
                wsum = float(sum(weights))
            norm = [float(w) / max(wsum, 1e-6) for w in weights]

            resp = {int(pid): float(norm[i]) for i, pid in enumerate(pids)}
            setattr(ch, "_parent_resp", resp)
            setattr(ch, "_parent_resp_mask", {})

            for i, pid in enumerate(pids):
                P = parents_by_id.get(int(pid))
                if P is not None:
                    d = getattr(P, "_child_resp", None) or {}
                    d[cid] = float(norm[i])
                    setattr(P, "_child_resp", d)


def enforce_min_children_per_parent(parents_by_id, *, min_kids=2, respect_grace=True):
    """If assignment collapses a parent to < min_kids, revert to previous ids during grace."""
    for P in (parents_by_id or {}).values():
        try:
            cur = tuple(getattr(P, "child_ids", ()) or ())
            if len(cur) >= int(min_kids):
                continue
            prev = tuple(getattr(P, "_prev_child_ids", ()) or ())
            if respect_grace and int(getattr(P, "_grace", 0)) > 0 and len(prev) >= int(min_kids):
                P.child_ids = prev
        except Exception:
            pass





def _as_parent_dict(obj):
    """
    Normalize parents -> {pid: Parent}. Accepts dict, list, iterable, or None.
    Uses the parent's .id if present; else falls back to enumerate index.
    """
    import collections.abc as _abc
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return {int(getattr(p, "id", pid)): p for pid, p in obj.items()}
    if isinstance(obj, _abc.Iterable):
        return {int(getattr(p, "id", i)): p for i, p in enumerate(obj)}
    # Single parent object?
    return {int(getattr(obj, "id", 0)): obj}




