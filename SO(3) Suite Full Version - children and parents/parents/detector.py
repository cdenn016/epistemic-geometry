import numpy as np
import core.config as config
from typing import Dict, List, Optional, Any

from dataclasses import dataclass, field
# at top of detector.py
from parents.detector_utils import (
    compute_cover_weight, aggregate_child_agreement,
    apply_weight_gate, erode_core_bool, region_confidence
    )
from parents.parent_utils import _label4
from types import SimpleNamespace
from core.gaussian_core import resolve_support_tau



@dataclass
class Region:
    id: int
    mask: np.ndarray
    state: str
    age: int = 0
    t_activated: Optional[int] = None
    child_ids: List[int] = field(default_factory=list)




@dataclass
class DetectorState:
    regions: Dict[int, Region] = field(default_factory=dict)
    next_id: int = 0
    step: int = 0
    # new: live threshold + last-step diagnostics (used by update_rules & debug)
    agree_gate_tau_eff: Optional[float] = None
    last_stats: Dict[str, Any] = field(default_factory=dict)





def detect_regions_minimal(
    children, *, H, W,
    level: int = 0,
    min_kids: int,
    min_area: int,
    kl_tau=None,                     # None → skip KL/Agree gate
    weight_tau=0.6,                  # or config.parent_weight_tau (None => auto)
    erode_iters=None,                # or config.detector_edge_erode
    runtime_ctx=None,
    agree_gate_tau=None,             # or config.agree_gate_tau / runtime auto
    periodic=None,                   # DEPRECATED (ignored): domain is ALWAYS periodic
    return_cand_map: bool = True,
):
    """
    Minimal, unified region detector on a **toroidal (periodic) grid**:
      1) require >= min_kids active children per pixel
      2) interior gate by mean child-mask weight (adaptive if weight_tau=None)
      3) optional local gate via blended agreement from per-child softmin KLs (kl_tau)
      4) optional global lvl-0 agreement gate (Aq_agg/Ap_agg >= agree_gate_tau)
      5) 4-neigh erosion (periodic boundary conditions)
      6) CC + area filter (periodic boundary conditions)
      7) region-level probabilistic confidence; cand_map is confidence-weighted

    Returns: (regions_dict, cand_map) if return_cand_map else regions_dict
    """
    # --------------------------- knobs ----------------------------------------
    # ALWAYS periodic; ignore any caller-provided 'periodic' and any config flags
    _PERIODIC = True

    if erode_iters is None:
        erode_iters = int(getattr(config, "detector_edge_erode", 0))

    cfg_weight_tau = weight_tau if (weight_tau is not None) else getattr(config, "parent_weight_tau", None)

    # dynamic global agree gate (runtime EMA if enabled)
    if agree_gate_tau is None:
        if runtime_ctx is not None and bool(getattr(config, "detector_agree_auto", False)):
            try:
                ds = runtime_ctx.get_detector_state(level)
                agree_gate_tau = getattr(ds, "agree_gate_tau_eff", None)
            except Exception:
                agree_gate_tau = None
        if agree_gate_tau is None:
            agree_gate_tau = getattr(config, "agree_gate_tau", None)
            
    alpha         = float(getattr(config, "blend_qp_alpha", 0.5))
    tau_q         = float(getattr(config, "tau_align_q", 0.45))
    tau_p         = float(getattr(config, "tau_align_p", 0.45))
    tau_blend     = alpha * tau_q + (1.0 - alpha) * tau_p
    
    mask_tau_mask = resolve_support_tau(runtime_ctx, None, default=1e-3, name="support_tau")


    prior_px      = float(getattr(config, "detector_prior_parent_px", 0.10))  # NEW

    # ----------------- per-child evidence gather ------------------------------
    cover, weight, kid_act = compute_cover_weight(children, H, W, mask_tau_mask, config)
    kid_agree = aggregate_child_agreement(children, kid_act, H, W, alpha, tau_q, tau_p, tau_blend)

    # 1) require enough active kids + 2) interior preference
    keep, mean_weight, thr = apply_weight_gate(cover, weight, min_kids, cfg_weight_tau)

    # 3) Local KL/Agreement gate (only if requested and evidence present)
    if kl_tau is not None and any(A is not None for A in kid_agree):
        A_sum   = np.zeros((H, W), np.float32)
        A_count = np.zeros((H, W), np.int32)
        for a, A in zip(kid_act, kid_agree):
            if A is None:
                continue
            inside = a & np.isfinite(A)
            if inside.any():
                A_sum[inside]   += A[inside]
                A_count[inside] += 1
        A_mean = np.zeros((H, W), np.float32)
        validA = A_count > 0
        A_mean[validA] = A_sum[validA] / A_count[validA].astype(np.float32)
        A_mean = np.clip(A_mean, 0.0, 1.0)
        # map agreement → KL with blended τ and gate
        KL_eff = np.full((H, W), np.inf, np.float32)
        KL_eff[validA] = -tau_blend * np.log(np.maximum(A_mean[validA], 1e-20)).astype(np.float32)
        keep &= (KL_eff <= float(kl_tau))

    # 4) Global agreement gate (lvl-0 aggregates)
    if (agree_gate_tau is not None) and (runtime_ctx is not None):
        A_global = getattr(runtime_ctx, "Aq_agg", None)
        if A_global is None:
            A_global = getattr(runtime_ctx, "Ap_agg", None)
        if A_global is not None and getattr(A_global, "shape", None) == (H, W):
            keep &= (np.asarray(A_global, np.float32) >= float(agree_gate_tau))
    else:
        A_global = None  # ensure defined for later

    pre_px = int(np.asarray(keep).sum())
    core = keep.astype(bool)

    # 5) erosion (toroidal-safe) — ALWAYS periodic
    core = erode_core_bool(core, int(erode_iters), _PERIODIC)
    post_px = int(core.sum())

    # 6) Toroidal-aware CC + area filter — ALWAYS periodic
    labels = _label4(core, periodic=_PERIODIC)
    nlab = int(labels.max())

    regions: Dict[int, Any] = {}
    rid = 0

    # conservative child-set stamping: require min overlap PX inside region
    try:
        min_overlap_px = int(getattr(config, "parent_assign_min_overlap_px", 8))
    except Exception:
        min_overlap_px = 8

    kid_act_bool = [np.asarray(a, bool) for a in kid_act]

    conf_vals = []  # NEW: for diagnostics

    for lbl in range(1, nlab + 1):
        comp = (labels == lbl)
        area = int(comp.sum())
        if area < int(min_area):
            continue

        # Build region
        R = _make_region(comp.astype(np.float32), rid)

        # Stamp region-level child_ids (conservative, respects your schema)
        child_ids_here: List[int] = []
        for idx, a in enumerate(kid_act_bool):
            if int((a & comp).sum()) >= min_overlap_px:
                try:
                    child_ids_here.append(int(getattr(children[idx], "id", idx)))
                except Exception:
                    pass
        try:
            R.child_ids = child_ids_here
        except Exception:
            pass

        # --- NEW: probabilistic confidence
        rc = region_confidence(
            comp,
            A_global=A_global,
            cover=cover,
            mean_weight=mean_weight,
            prior_p=prior_px,
            eps=1e-6,
        )
        try:
            R.confidence = float(rc["conf"])
            R.conf_parts = rc["parts"]
            conf_vals.append(R.confidence)
        except Exception:
            pass

        regions[rid] = R
        rid += 1

    # -------------- persist lightweight diagnostics ---------------------------
    try:
        if runtime_ctx is not None:
            ds = runtime_ctx.get_detector_state(level)
            coal_hist = np.bincount(np.clip(cover.ravel(), 0, 8), minlength=9).astype(int)
            conf_med = float(np.median(conf_vals)) if conf_vals else 0.0
            conf_max = float(np.max(conf_vals)) if conf_vals else 0.0
            # store ephemeral stats on the detector state for debugging/telemetry
            ds.last_stats = dict(
                px_core=post_px,
                px_pre_gate=pre_px,
                cover_mean=float(cover.mean()),
                cover_ge2=int((cover >= 2).sum()),
                coal_hist=coal_hist[:9].tolist(),
                weight_thr=float(thr) if np.isfinite(thr) else None,
                agree_tau_eff=float(agree_gate_tau) if agree_gate_tau is not None else None,
                min_kids=int(min_kids),
                min_area=int(min_area),
                conf_med=conf_med,
                conf_max=conf_max,
            )
    except Exception:
        pass

    # -------------- log line --------------------------------------------------
    try:
        tau_repr = f"{thr:.3f}" if cfg_weight_tau is not None else f"auto->{thr:.3f}"
        extra = ""
        if conf_vals:
            extra = f" conf_med={float(np.median(conf_vals)):.3f} conf_max={float(np.max(conf_vals)):.3f}"
        print(f"[DETECT] regs={len(regions)} px_keep={pre_px} px_post={post_px} "
              f"min_area={int(min_area)} weight_tau={tau_repr} "
              f"cover_sum={int(cover.sum())} mean_w={float(np.nan_to_num(mean_weight).mean()):.3f}{extra}")
    except Exception:
        pass

    if not return_cand_map:
        return regions

    # candidate score map = max over regions of (mask * confidence)
    cand_map = np.zeros((H, W), np.float32)
    for R in regions.values():
        conf = float(getattr(R, "confidence", 1.0))
        if conf <= 0.0:
            continue
        cand_map = np.maximum(cand_map, np.asarray(R.mask, np.float32) * conf)

    return regions, cand_map





# --- NEW: single, minimal detector -------------------------------------------

def _make_region(mask: np.ndarray, rid: int):
    """
    Construct a Region with whatever signature your project uses.
    Tries several common ctor shapes; falls back to a simple object.
    """
    area  = int((mask > 0).sum())
    state = getattr(config, "region_active_state", "active")  # or 1 / True, if your code uses enums

    # Try common constructor shapes
    attempts = [
        lambda: Region(id=rid, state=state, mask=mask, area=area),
        lambda: Region(id=rid, state=state, mask=mask),
        lambda: Region(rid, state, mask),           # positional
        lambda: Region(mask=mask),                  # legacy minimal
        lambda: Region(mask),                       # very old minimal
    ]
    for ctor in attempts:
        try:
            return ctor()
        except TypeError:
            pass

    # Fallback: lightweight object with the fields we need downstream
    class _R: pass
    R = _R()
    R.id = rid
    R.state = state
    R.mask = mask
    R.area = area
    return R




def _parse_kl_tau(x):
    if x is None:
        return None
    if isinstance(x, str) and x.strip().lower() in ("none", "off", "disabled", ""):
        return None
    try:
        return float(x)
    except Exception:
        return None

