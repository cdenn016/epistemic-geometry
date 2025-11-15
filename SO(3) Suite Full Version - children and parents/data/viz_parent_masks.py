# viz_parent_masks.py
# viz_parent_masks.py
import os
import numpy as np

# Always force a headless backend *before* importing pyplot
import matplotlib as mpl
mpl.use("Agg", force=True)
import matplotlib.pyplot as plt

from core.numerical_utils import resize_nn
import core.config as config  # ensure available as `config` everywhere




def _collect_all_parents(ctx):
    """
    Return a flat list of parent objects from any of the known containers:
      - ctx.parents_by_level : {level -> {pid -> parent}}
      - ctx.parents_by_region / ctx.parents : {pid -> parent}
      - ctx.parents_latest : list or {pid -> parent}
    Preserves last-writer-wins semantics for duplicate IDs.
    """
    if ctx is None:
        return []

    candidates = []

    # 1) parents_by_level (newer code path)
    pbL = getattr(ctx, "parents_by_level", None)
    if isinstance(pbL, dict) and pbL:
        for bucket in pbL.values():
            if isinstance(bucket, dict):
                candidates.extend(list(bucket.values()))
            else:
                candidates.extend(list(bucket))

    # 2) legacy / alternate storages
    for name in ("parents_by_region", "parents", "parents_latest"):
        store = getattr(ctx, name, None)
        if store is None:
            continue
        if isinstance(store, dict):
            candidates.extend(list(store.values()))
        else:
            candidates.extend(list(store))

    # De-dup by id
    uniq = {}
    for p in candidates:
        pid = int(getattr(p, "id", id(p)))
        uniq[pid] = p
    # stable order by id
    return [uniq[k] for k in sorted(uniq.keys())]


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# --- fix: blend helpers (bad slicing) ------------------------------------------
def _blend_parents_max(masks_dict, cmap_name="viridis", floor=1e-4):
    """
    Combine parent masks into an RGB image without normalization artifacts.
    - Uses per-pixel max over raw mask values (after floor),
      not sum/mean/auto-gain.
    - No per-parent normalization.
    """
    
    import matplotlib.cm as cm

    if not masks_dict:
        return np.zeros((1, 1, 3), dtype=np.float32)

    arrs = [np.asarray(m, np.float32) for m in masks_dict.values()]
    H, W = arrs[0].shape

    # floor tiny dust to zero before combine
    arrs = [np.where(a > float(floor), a, 0.0) for a in arrs]

    # per-pixel max
    I = np.maximum.reduce(arrs)
    I = np.clip(I, 0.0, 1.0)

    # map to RGB via colormap (no extra normalization)
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(I)                     # (H,W,4)
    rgb = rgba[..., :3].astype(np.float32)
    return rgb


def _blend_colors_max(rgb_list, weights=None):
    """Max-blend multiple RGB images with optional weights in [0,1]."""
    
    if not rgb_list:
        return np.zeros((1, 1, 3), dtype=np.float32)
    out = np.zeros_like(rgb_list[0], dtype=np.float32)
    if weights is None:
        for c in rgb_list:
            out = np.maximum(out, np.asarray(c, np.float32))
        return out
    # weighted max
    for c, w in zip(rgb_list, weights):
        cc = np.asarray(c, np.float32)
        mm = float(np.clip(w, 0.0, 1.0))
        out = np.maximum(out, mm[..., None] * cc)
    return out


# --- fix: use local roll_center + Agg backend; drop bare `config` import -------
def plot_parent_masks_grid(runtime_ctx, step, outdir,
                           *, center_on_com=True, shared_scale=True,
                           core_from="parent_area_threshold"):
    import numpy as np, matplotlib as mpl, os
    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    Ps = list((getattr(runtime_ctx, "parents_by_region", {}) or {}).values())
    if not Ps:
        return
    H, W = Ps[0].mask.shape[:2]
    per_x = bool(getattr(config, "periodic_x", False))
    per_y = bool(getattr(config, "periodic_y", False))

    core_thr = float(getattr(config, core_from, getattr(config, "core_abs_tau", 0.20)))

    masks, vmaxs, titles = [], [], []
    for P in Ps:
        m = np.asarray(P.mask, np.float32)
        if center_on_com:
            # use local helper defined in this file
            m = roll_center(m, per_x=per_x, per_y=per_y)
        masks.append(m)
        vmaxs.append(float(np.nanmax(m)) if m.size else 1.0)
        k = len(getattr(P, "child_ids", ()))
        nz = int((m > core_thr).sum())
        titles.append(f"P{int(getattr(P,'id',-1))}  kids={k}  nz>{core_thr:g}={nz}")

    vmin = 0.0
    vmax = float(np.nanpercentile(vmaxs, 99.0)) if shared_scale else None

    n = len(Ps)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    os.makedirs(outdir, exist_ok=True)
    fig = plt.figure(figsize=(4*cols, 4*rows), constrained_layout=True)

    axes = []
    last_im = None
    for i, m in enumerate(masks):
        ax = fig.add_subplot(rows, cols, i+1)
        axes.append(ax)
        last_im = ax.imshow(m, origin="upper",
                            vmin=vmin, vmax=(vmax if vmax is not None else None),
                            cmap="magma")
        if (m > core_thr).any():
            ax.contour((m > core_thr).astype(float), levels=[0.5],
                       colors="white", linewidths=0.6)
        ax.set_title(titles[i], fontsize=10)
        ax.axis("off")

    if last_im is not None:
        fig.colorbar(last_im, ax=axes, location="right", shrink=0.9, pad=0.02)

    fig.suptitle(f"Parent Masks (step {step})", fontsize=14)
    fig.savefig(os.path.join(outdir, f"parent_masks_step_{int(step):04d}.png"), dpi=140)
    plt.close(fig)



def _stack_child_support(children, shape, *, eps=1e-3):
    """Return (bool stack (N,H,W), ids). Support is (mask > eps), resized as needed."""
    
    
    H, W = shape
    stacks, ids = [], []
    for ch in (children or []):
        m = getattr(ch, "mask", None)
        if m is None:
            continue
        M = np.asarray(m, np.float32)
        if M.shape != (H, W):
            M = resize_nn(M, (H, W)).astype(np.float32)
        stacks.append(M > float(eps))
        ids.append(int(getattr(ch, "id", len(ids))))
    if not stacks:
        return None, ()
    return np.stack(stacks, axis=0).astype(bool), tuple(ids)


def _masked_mean_over_kids(stack, support_stack):
    """
    Pixelwise mean over the *active* kids only.
    Returns (mean, count). If no active kids at a pixel, count=0 and mean is 0 (unused).
    """
    
    if stack is None or support_stack is None:
        return None, None
    S = support_stack.astype(np.float32)
    num = np.nansum(np.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0) * S, axis=0)
    den = np.maximum(S.sum(axis=0), 1.0)
    mean = num / den
    count = support_stack.sum(axis=0).astype(np.int32)
    return mean.astype(np.float32), count


def _normalize01(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0 or not np.isfinite(x).any():
        return np.zeros_like(x, dtype=float)
    x = x - np.nanmin(x)
    mx = np.nanmax(x)
    return x / (mx + 1e-12) if mx > 0 else np.zeros_like(x, dtype=float)

def _autogain_support(x, *, vis_floor=1e-2, lo_p=0.0, hi_p=98.0, gamma=1.1):
    """
    Autogain only over the mask support (x >= vis_floor). Returns:
      norm_img in [0,1], support boolean, (lo, hi) original-value range used.
    """
    x = np.asarray(x, dtype=float)
    supp = x >= vis_floor
    if not supp.any():
        return np.zeros_like(x), supp, (0.0, 1.0)

    vals = x[supp]
    lo = np.percentile(vals, lo_p)
    hi = np.percentile(vals, hi_p)
    if hi <= lo:
        y = np.zeros_like(x)
        y[supp] = 1.0
        return y, supp, (lo, hi)

    y = np.zeros_like(x)
    z = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    if gamma > 0 and gamma != 1.0:
        z = np.power(z, 1.0 / gamma)
    y[supp] = z[supp]
    return y, supp, (float(lo), float(hi))



# --- add near the top (PBC helpers) -------------------------------------------
def _com(mask):
    
    m = np.asarray(mask, np.float32)
    s = float(m.sum())
    if s <= 0: return 0.0, 0.0
    H, W = m.shape[:2]
    Y, X = np.indices((H, W), dtype=np.float32)
    return float((Y*m).sum()/s), float((X*m).sum()/s)

def _wrap_delta(d, L):
    # minimal signed displacement on a ring of length L ([-L/2, L/2))
    return (d + 0.5*L) % L - 0.5*L

def roll_center(mask, *, per_x=True, per_y=True):
    """Roll mask so its COM lands at the image center (PBC-correct view)."""
   
    m = np.asarray(mask, np.float32)
    H, W = m.shape[:2]
    yc, xc = _com(m)
    dy = int(round(_wrap_delta(yc - (H/2.0-0.5), H))) if per_y else int(round(yc - (H/2.0-0.5)))
    dx = int(round(_wrap_delta(xc - (W/2.0-0.5), W))) if per_x else int(round(xc - (W/2.0-0.5)))
    return np.roll(np.roll(m, -dy, axis=0), -dx, axis=1)





def _stack_child_field(children, attr, shape):
   
    H, W = shape
    fields, ids = [], []
    for ch in (children or []):
        fld = getattr(ch, attr, None)
        if fld is None: 
            continue
        f = np.asarray(fld, np.float32)
        if f.shape != (H, W):
            f = resize_nn(f, (H, W)).astype(np.float32)
        fields.append(f); ids.append(int(getattr(ch, "id", len(ids))))
    if not fields:
        return None, ()
    return np.stack(fields, axis=0), tuple(ids)


def save_alignment_vs_parents(
    runtime_ctx, step, outdir, *,
    mode="agreement", cmap="viridis",
    show=False,           # ignored; plots are never shown interactively
    plot_delta=True
):
    # --- force non-interactive backend BEFORE importing pyplot ---
    import os, numpy as np, matplotlib as mpl
    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import core.config as CFG

    _ensure_dir(outdir)

    children = getattr(runtime_ctx, "children_latest", None)
    parents  = getattr(runtime_ctx, "parents_by_region", {}) or {}
    parents  = list(parents.values()) if isinstance(parents, dict) else list(parents)
    if not children:
        return
    H, W = np.asarray(children[0].mask).shape[:2]

    tau_q    = float(getattr(CFG, "tau_align_q", 0.45))
    tau_p    = float(getattr(CFG, "tau_align_p", 0.45))
    kl_cap   = float(getattr(CFG, "signal_kl_cap", 10.0))
    child_eps= float(getattr(CFG, "support_tau", 0.08))

    # Aggregates (preferred)
    Aq_agg      = getattr(runtime_ctx, "Aq_agg", None)
    Ap_agg      = getattr(runtime_ctx, "Ap_agg", None)
    Aq_support  = getattr(runtime_ctx, "Aq_support", None)     # bool mask (pairwise support)
    KLq_mean_ag = getattr(runtime_ctx, "KLq_mean_agg", None)

    # Fallback: union of kids (≥1 active)
    def _union_of_kids():
        try:
            M_stack, _ = _stack_child_support(children, (H, W), eps=child_eps)
            if M_stack is None:
                return None
            return np.any(M_stack.astype(bool), axis=0)
        except Exception:
            return None

    # Visualization mask preference
    if isinstance(Aq_support, np.ndarray) and Aq_support.any():
        viz_mask = Aq_support.astype(bool)
    else:
        U = _union_of_kids()
        viz_mask = U if (isinstance(U, np.ndarray) and U.any()) else None

    KLq_mean = None
    KLp_mean = None
    Aq = None
    Ap = None

    # Prefer aggregates; reconstruct KL from A if needed
    if (Aq_agg is not None) or (KLq_mean_ag is not None):
        if Aq_agg is not None:
            Aq = np.ones((H, W), np.float32)
            if isinstance(Aq_support, np.ndarray) and Aq_support.any():
                Aq[Aq_support] = np.clip(np.asarray(Aq_agg, np.float32)[Aq_support], 0.0, 1.0)
            else:
                Aq = np.clip(np.asarray(Aq_agg, np.float32), 0.0, 1.0)

        if KLq_mean_ag is not None:
            KLq_mean = np.asarray(KLq_mean_ag, np.float32)
        elif Aq is not None:
            KLq_mean = -tau_q * np.log(np.clip(np.asarray(Aq, np.float32), 1e-6, 1.0))
    else:
        # Legacy per-child fallback
        KLq_stack, _ = _stack_child_field(children, "cached_alignment_kl", (H, W))
        KLp_stack, _ = _stack_child_field(children, "cached_model_alignment_kl", (H, W))
        M_stack,  C  = _stack_child_support(children, (H, W), eps=child_eps)
        KLq_mean, _  = _masked_mean_over_kids(KLq_stack, M_stack)
        KLp_mean, _  = _masked_mean_over_kids(KLp_stack, M_stack)

        def _to_A_from_KL(KL_mean, tau):
            if KL_mean is None:
                return None
            K = np.clip(np.asarray(KL_mean, np.float32), 0.0, kl_cap)
            return np.exp(-K / max(1e-6, float(tau))).astype(np.float32)

        Aq = _to_A_from_KL(KLq_mean, tau_q)
        Ap = _to_A_from_KL(KLp_mean, tau_p)
        if isinstance(C, np.ndarray):
            viz_mask = C >= 2 if (viz_mask is None) else viz_mask

    # If model-side not present, reconstruct from Ap_agg if we have it
    if (Ap is None) and (Ap_agg is not None):
        Ap = np.clip(np.asarray(Ap_agg, np.float32), 0.0, 1.0)
    if (KLp_mean is None) and (Ap is not None):
        KLp_mean = -tau_p * np.log(np.clip(Ap, 1e-6, 1.0))

    # Parent & child outlines
    thr = float(getattr(CFG, "parent_area_threshold",
                        getattr(CFG, "core_abs_tau", 0.20)))
    parent_outlines = [(np.asarray(P.mask, float) > thr) for P in parents]
    child_outlines = []
    try:
        M_stack_plot, _ = _stack_child_support(children, (H, W), eps=child_eps)
        if M_stack_plot is not None:
            for k in range(M_stack_plot.shape[0]):
                child_outlines.append(M_stack_plot[k])
    except Exception:
        pass

    def _draw(field, title, subdir, vmin=None, vmax=None, *, diverging=False):
        _ensure_dir(os.path.join(outdir, subdir))
        plt.figure(figsize=(4, 4))
        ax = plt.gca()

        if field is None:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=12)
            ax.set_title(f"{title} (step {step})")
            ax.axis("off")
            path = os.path.join(outdir, subdir, f"step_{step:04d}.png")
            plt.tight_layout(); plt.savefig(path, dpi=140)
            print("Saved (placeholder):", path)
            plt.close()
            return

        F = np.asarray(field, np.float32)

        # Apply viz mask by filling outside region with vmin (not NaN)
        if isinstance(viz_mask, np.ndarray):
            mask = viz_mask.astype(bool)
            if vmin is None: vmin = np.nanmin(F)
            if vmax is None: vmax = np.nanmax(F)
            F = np.where(mask, F, vmin)

        # Autoscale
        if diverging:
            amax = float(np.nanmax(np.abs(F))) if np.isfinite(F).any() else 1.0
            vmin, vmax = -amax, amax
            cm = "coolwarm"
        else:
            if vmin is None or vmax is None:
                vmin, vmax = np.nanmin(F), np.nanmax(F)
                if vmin == vmax: vmax = vmin + 1e-6
            cm = cmap

        im = ax.imshow(F, origin="upper", cmap=cm, vmin=vmin, vmax=vmax)

        for ol in child_outlines:
            if np.asarray(ol).any():
                ax.contour(ol.astype(float), levels=[0.5], colors="gray", linewidths=0.4, alpha=0.6)
        if isinstance(viz_mask, np.ndarray) and viz_mask.any():
            ax.contour(viz_mask.astype(float), levels=[0.5], colors="red", linewidths=0.8)
        for ol in parent_outlines:
            if np.asarray(ol).any():
                ax.contour(ol.astype(float), levels=[0.5], colors="white", linewidths=0.5)

        ax.set_title(f"{title} (step {step})")
        ax.axis("off")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        path = os.path.join(outdir, subdir, f"step_{step:04d}.png")
        plt.savefig(path, dpi=140)
        print("Saved:", path)
        plt.close()

    # ---- base plots ----
    if mode == "agreement":
        _draw(Aq, "Agreement (belief) + child/overlap + parent contours",
              "alignment_belief_vs_parents", vmin=0.0, vmax=1.0)
        _draw(Ap, "Agreement (model) + child/overlap + parent contours",
              "alignment_model_vs_parents", vmin=0.0, vmax=1.0)
    else:  # "kl"
        if KLq_mean is not None:
            _draw(np.clip(KLq_mean, 0.0, kl_cap), "KL mean (belief) on overlaps + contours",
                  "alignment_belief_vs_parents", vmin=0.0, vmax=kl_cap)
        if KLp_mean is not None:
            _draw(np.clip(KLp_mean, 0.0, kl_cap), "KL mean (model) on overlaps + contours",
                  "alignment_model_vs_parents", vmin=0.0, vmax=kl_cap)

    # ---- delta plots (current - previous) ----
    if plot_delta:
        prev_KLq = getattr(runtime_ctx, "_last_KLq_mean", None)
        prev_KLp = getattr(runtime_ctx, "_last_KLp_mean", None)
        prev_Aq  = getattr(runtime_ctx, "_last_Aq", None)
        prev_Ap  = getattr(runtime_ctx, "_last_Ap", None)

        if mode == "kl":
            if KLq_mean is not None and isinstance(prev_KLq, np.ndarray):
                _draw(np.nan_to_num(KLq_mean - prev_KLq, nan=0.0),
                      "Δ KL (belief) current - previous", "alignment_belief_dKL",
                      diverging=True)
            if KLp_mean is not None and isinstance(prev_KLp, np.ndarray):
                _draw(np.nan_to_num(KLp_mean - prev_KLp, nan=0.0),
                      "Δ KL (model) current - previous", "alignment_model_dKL",
                      diverging=True)
        else:  # agreement
            if Aq is not None and isinstance(prev_Aq, np.ndarray):
                _draw(np.nan_to_num(Aq - prev_Aq, nan=0.0),
                      "Δ Agreement (belief)", "alignment_belief_dA",
                      diverging=True)
            if Ap is not None and isinstance(prev_Ap, np.ndarray):
                _draw(np.nan_to_num(Ap - prev_Ap, nan=0.0),
                      "Δ Agreement (model)", "alignment_model_dA",
                      diverging=True)

        # cache current for next step
        if KLq_mean is not None: runtime_ctx._last_KLq_mean = np.asarray(KLq_mean, np.float32).copy()
        if KLp_mean is not None: runtime_ctx._last_KLp_mean = np.asarray(KLp_mean, np.float32).copy()
        if Aq is not None:       runtime_ctx._last_Aq       = np.asarray(Aq, np.float32).copy()
        if Ap is not None:       runtime_ctx._last_Ap       = np.asarray(Ap, np.float32).copy()

    # ---- NPZ snapshot ----
    npz_dir = os.path.join(outdir, "alignment_npz"); _ensure_dir(npz_dir)
    parent_ids = np.array([int(getattr(P, "id", i)) for i, P in enumerate(parents)], np.int32)
    parent_masks = np.stack([np.asarray(P.mask, np.float32) for P in parents], axis=0) if parents else np.zeros((0, H, W), np.float32)

    _Aq_npz  = (Aq if Aq is not None else (np.asarray(Aq_agg, np.float32) if Aq_agg is not None else np.ones((H, W), np.float32)))
    _KLq_npz = (KLq_mean if KLq_mean is not None else np.full((H, W), np.nan, np.float32))
    _mask_npz= (viz_mask.astype(np.bool_) if isinstance(viz_mask, np.ndarray) else np.zeros((H, W), np.bool_))
    _child_ids = np.array([getattr(c, "id", i) for i, c in enumerate(children)], np.int32)

    np.savez_compressed(
        os.path.join(npz_dir, f"step_{step:04d}.npz"),
        child_ids=_child_ids,
        KLq_mean=_KLq_npz,
        Aq=_Aq_npz.astype(np.float32),
        viz_mask=_mask_npz,
        parent_ids=parent_ids,
        parent_masks=parent_masks,
    )



def save_alignment_metrics_table(runtime_ctx, step, outdir):
    import csv
    os.makedirs(outdir, exist_ok=True)
    parents = getattr(runtime_ctx, "parents_by_region", {}) or {}
    parents = list(parents.values()) if isinstance(parents, dict) else list(parents)
    if not parents:
        return
    H, W = parents[0].mask.shape[:2]
    Aq = getattr(runtime_ctx, "Aq_agg", None)
    Ap = getattr(runtime_ctx, "Ap_agg", None)
    if Aq is None and Ap is None:
        return
    try:
        
        thrs = list(getattr(config, "align_eval_thresholds", [0.30, 0.45, 0.60]))
    except Exception:
        thrs = [0.30, 0.45, 0.60]

    def _metrics(A, Pmask, thr):
        if A is None: return (np.nan, np.nan, np.nan)
        M = (A >= thr)
        P = Pmask.astype(bool)
        inter = np.count_nonzero(M & P)
        union = np.count_nonzero(M | P)
        iou  = (inter / union) if union else 0.0
        prec = (inter / max(1, np.count_nonzero(M)))
        rec  = (inter / max(1, np.count_nonzero(P)))
        return (iou, prec, rec)

    rows = []
    for P in parents:
        pmask = (np.asarray(P.mask, np.float32) > 0.3)
        for side, A in (("belief", Aq), ("model", Ap)):
            for t in thrs:
                iou, pr, rc = _metrics(A, pmask, t)
                rows.append({
                    "step": step, "parent_id": int(getattr(P, "id", -1)),
                    "side": side, "tau": t,
                    "IoU": iou, "Precision": pr, "Recall": rc,
                })

    csv_path = os.path.join(outdir, f"alignment_metrics_step_{step:04d}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def save_parent_mask_frame(runtime_ctx, step, outdir, draw_per_parent=True, *, child_eps=1e-3):
    
    
    # ---------- unified viz params (no new knobs needed) ----------
    tau_on  = float(getattr(config, "support_tau", 1e-4))             # draw at the same floor as the pipeline
    tau_off = float(getattr(config, "viz_tau_off_frac", 0.85)) * tau_on  # hysteresis off-threshold
    ema_a   = float(getattr(config, "viz_mask_ema", 0.30))            # temporal smoothing (0..1); 0.3 is mild
    iou_thr = float(getattr(config, "viz_iou_match_thr", 0.30))       # match current↔prev if IDs churn
    min_px  = int(getattr(config, "viz_min_px", 8))                   # don’t draw tiny specks

    # persistent viz state on ctx
    if not hasattr(runtime_ctx, "_viz_prev_soft"): runtime_ctx._viz_prev_soft = {}   # pid -> soft mask (smoothed)
    if not hasattr(runtime_ctx, "_viz_prev_ids"):  runtime_ctx._viz_prev_ids  = set()
    prev_soft: dict[int, np.ndarray] = runtime_ctx._viz_prev_soft

    parents = _collect_all_parents(runtime_ctx)
    if not parents:
        if getattr(config, "debug_parent_masks", False):
            print(f"[viz] step {step}: no parents found (all containers empty)")
        return


    H, W = parents[0].mask.shape

    # ---------------- child union (context) ----------------
    children = getattr(runtime_ctx, "children_latest", None) or getattr(runtime_ctx, "last_children", None)
    child_union = np.zeros((H, W), bool)
    if children:
        for ch in children:
            child_union |= (np.asarray(ch.mask, float) > float(child_eps))

    # ---------------- helpers ----------------
    def _ensure_dir(d): os.makedirs(d, exist_ok=True)

    def _blend_colors_max(mask_dict, floor=1e-3):
        import colorsys
        rgb = np.zeros((H, W, 3), np.float32)
        for pid, m in mask_dict.items():
            rnd = (int(pid) * 2654435761) & 0xFFFFFFFF
            h = ((rnd % 360) / 360.0); s = 0.65; v = 1.0
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            c = np.array([r, g, b], np.float32)
            mm = np.clip(np.asarray(m, np.float32), 0.0, 1.0)
            rgb = np.maximum(rgb, mm[..., None] * c)
        if floor > 0: rgb = np.maximum(rgb, floor)
        return np.clip(rgb, 0.0, 1.0)

    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        inter = np.count_nonzero(a & b)
        if inter == 0: return 0.0
        u = a.sum() + b.sum() - inter
        return float(inter) / float(u) if u > 0 else 0.0

    def _prev_for_pid_or_match(pid: int, M_now_on: np.ndarray):
        """Return a previous soft mask to smooth with; if pid not found, try IoU-match."""
        if pid in prev_soft:
            return prev_soft[pid], pid
        # try to match any previous by IoU if IDs changed
        best_pid, best_iou = None, 0.0
        for old_pid, S_prev in prev_soft.items():
            prev_on = (S_prev > tau_on)  # previous on-region
            iou = _iou(prev_on, M_now_on)
            if iou > best_iou:
                best_pid, best_iou = old_pid, iou
        if best_iou >= iou_thr:
            return prev_soft[best_pid], best_pid
        return None, None

    def _smooth_and_hysteresis(pid: int, M_soft: np.ndarray) -> np.ndarray:
        """EMA smooth + hysteresis → binary mask."""
        on_now = (M_soft > tau_on)
        S_prev, match_pid = _prev_for_pid_or_match(pid, on_now)
        if S_prev is None:
            S_smooth = M_soft
            stable_pid = pid
        else:
            # if matched to a different previous ID, re-use that ID’s state (stabilizes churn)
            stable_pid = match_pid
            S_smooth = (1.0 - ema_a) * S_prev + ema_a * M_soft

        # hysteresis on smoothed signal
        on_high = (S_smooth > tau_on)
        on_low  = (S_smooth > tau_off)
        prev_on = (prev_soft.get(stable_pid, S_smooth) > tau_on)
        on_out  = np.logical_or(on_high, np.logical_and(prev_on, on_low))

        # store back under the *current* pid for next step
        prev_soft[pid] = S_smooth
        return on_out

    # ---------------- make soft masks dict ----------------
    masks_soft = {int(getattr(P, "id", i)): np.asarray(P.mask, float) for i, P in enumerate(parents)}

    if getattr(config, "debug_parent_masks", False):
        keys_show = list(masks_soft.keys())[:8]
        mx = max(float(m.max()) for m in masks_soft.values())
        print(f"[viz] step {step}: plotting {len(parents)} parents (keys={keys_show}{'...' if len(masks_soft)>8 else ''})")
        print(f"[viz] step {step}: max_mask={mx:.3f} | τ_on={tau_on:g} τ_off={tau_off:g} ema={ema_a}\n")

    # ================= 1) Combined overlay =================
    out1 = os.path.join(outdir, "parent_masks_combined"); _ensure_dir(out1)
    rgb = _blend_colors_max(masks_soft, floor=float(getattr(config, "viz_union_floor", 1e-3)))

    fig = plt.figure(figsize=(4, 4), constrained_layout=True); ax = fig.add_subplot(1, 1, 1)
    ax.imshow(rgb, origin="upper")

    for P in parents:
        pid = int(getattr(P, "id", -1))
        pm_bool = _smooth_and_hysteresis(pid, masks_soft[pid])
        if pm_bool.sum() >= min_px:
            ax.contour(pm_bool.astype(float), levels=[0.5], colors="white", linewidths=0.6)

    if child_union.any():
        ax.contour(child_union.astype(float), levels=[0.5], colors=[(0.85,0.85,0.85)], linewidths=0.8)

    ax.set_title(f"Parents (step {step}) | n={len(parents)}"); ax.axis("off")
    fig.savefig(os.path.join(out1, f"step_{step:04d}.png"), dpi=140); plt.close(fig)

    if not draw_per_parent: return

    # ================= 2) Per-parent panels =================
    out2 = os.path.join(outdir, "parent_masks_panels"); _ensure_dir(out2)
    vmin_fixed = float(getattr(config, "viz_mask_vmin", 0.0))
    vmax_fixed = float(getattr(config, "viz_mask_vmax", 1.0))
    if vmax_fixed <= vmin_fixed: vmax_fixed = vmin_fixed + 1e-6

    n = len(parents)
    cols = min(4, max(1, int(np.ceil(np.sqrt(n)))))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.2 * rows), constrained_layout=True)
    axes = np.atleast_2d(axes).reshape(rows, cols)

    last_im = None
    for idx, P in enumerate(parents):
        r, c = divmod(idx, cols); ax = axes[r, c]
        pid = int(getattr(P, "id", -1))
        M = masks_soft[pid]
        last_im = ax.imshow(M, cmap="magma", origin="upper", vmin=vmin_fixed, vmax=vmax_fixed, alpha=0.95)

        pm_bool = _smooth_and_hysteresis(pid, M)
        if pm_bool.sum() >= min_px:
            ax.contour(pm_bool.astype(float), levels=[0.5], colors="white", linewidths=0.8)

        if child_union.any():
            ax.contour(child_union.astype(float), levels=[0.5], colors=[(0.85,0.85,0.85)], linewidths=0.7)

        wq = float(getattr(P, "lambda_q_weight", 0.0))
        kids = len(getattr(P, "child_ids", []))
        nz = int(pm_bool.sum())
        ax.set_title(f"P{getattr(P,'id','?')}  wq={wq:.2f}  kids={kids}  nz>viz={nz}", fontsize=9)
        ax.axis("off")

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols); axes[r, c].axis("off")
    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(), location="right", shrink=0.9, pad=0.02)
    fig.suptitle(f"Parent Masks (step {step})")
    fig.savefig(os.path.join(out2, f"step_{step:04d}.png"), dpi=140); plt.close(fig)


# --- NEW: parent mask deltas ---------------------------------------------------
def save_parent_mask_delta(runtime_ctx, step, outdir, *, thr_from="parent_area_threshold"):
    """
    Visualize Δ mask per parent: ΔM_t = M_t - M_{t-1}.
    - Stores the previous mask per parent on runtime_ctx._viz_prev_parent_masks
    - Produces:
        1) parent_mask_delta/combined_maxabs/step_XXXX.png  (max |Δ| across parents)
        2) parent_mask_delta/per_parent/step_XXXX.png       (grid of Δ per parent)
    """


    # ensure root dir
    if not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)

    parents = _collect_all_parents(runtime_ctx)
    if not parents:
        return


    H, W = parents[0].mask.shape[:2]
    # stable threshold for captions
    core_thr = float(getattr(config, thr_from,
                    getattr(config, "core_abs_tau", 0.20)))

    # cache of last masks
    if not hasattr(runtime_ctx, "_viz_prev_parent_masks"):
        runtime_ctx._viz_prev_parent_masks = {}
    prev = runtime_ctx._viz_prev_parent_masks

    deltas = {}
    for P in parents:
        pid = int(getattr(P, "id", -1))
        M_now = np.asarray(P.mask, np.float32)
        M_prev = np.asarray(prev.get(pid, np.zeros_like(M_now)), np.float32)
        deltas[pid] = (M_now - M_prev)

    # --- (1) combined: max |Δ| across parents --------------------------------
    out1 = os.path.join(outdir, "parent_mask_delta", "combined_maxabs")
    os.makedirs(out1, exist_ok=True)
    if deltas:
        Dmax = np.maximum.reduce([np.abs(d) for d in deltas.values()])
        # robust vmax (avoid all-zeros warnings)
        Dmax_pos = Dmax[Dmax > 0]
        vmax = float(np.percentile(Dmax_pos, 99.0)) if Dmax_pos.size else 1e-6
        vmax = max(vmax, 1e-6)

        fig = plt.figure(figsize=(4, 4), constrained_layout=True)
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(Dmax, origin="upper", cmap="magma", vmin=0.0, vmax=vmax)
        # parent outlines at core threshold for context
        for P in parents:
            pm = np.asarray(P.mask, float)
            if (pm > core_thr).any():
                ax.contour((pm > core_thr).astype(float), levels=[0.5], colors="white", linewidths=0.6)
        ax.set_title(f"Max |Δmask| across parents (step {step})")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.savefig(os.path.join(out1, f"step_{step:04d}.png"), dpi=140)
        plt.close(fig)

    # --- (2) per-parent symmetric grid ---------------------------------------
    out2 = os.path.join(outdir, "parent_mask_delta", "per_parent")
    os.makedirs(out2, exist_ok=True)

    # symmetric shared scale around 0 using 99th percentile of |Δ| over all parents
    if deltas:
        all_abs = np.concatenate([np.abs(d).ravel() for d in deltas.values()])
        all_abs_pos = all_abs[all_abs > 0]
        vmax = float(np.percentile(all_abs_pos, 99.0)) if all_abs_pos.size else 1e-6
    else:
        vmax = 1e-6
    vlim = max(vmax, 1e-6)

    n = len(parents)
    cols = min(4, max(1, int(np.ceil(np.sqrt(n)))))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.2 * rows), constrained_layout=True)
    axes = np.atleast_2d(axes).reshape(rows, cols)

    # stable child union for light outline
    children = getattr(runtime_ctx, "children_latest", None)
    child_union = np.zeros((H, W), bool)
    if children:
        ceps = float(getattr(config, "support_tau", 1e-3))
        for ch in children:
            child_union |= (np.asarray(ch.mask, float) > ceps)

    ims = []
    for idx, P in enumerate(parents):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        pid = int(getattr(P, "id", -1))
        D = deltas[pid]
        im = ax.imshow(D, origin="upper", cmap="coolwarm", vmin=-vlim, vmax=+vlim)
        ims.append(im)
        # overlays
        pm_bool = (np.asarray(P.mask, float) > core_thr)
        if pm_bool.any():
            ax.contour(pm_bool.astype(float), levels=[0.5], colors="k", linewidths=0.6)
        if child_union.any():
            ax.contour(child_union.astype(float), levels=[0.5], colors=[(0.7, 0.7, 0.7)], linewidths=0.6)
        nz = int(np.count_nonzero(np.abs(D) > 0))
        ax.set_title(f"P{pid}  |Δ|>0 px={nz}", fontsize=9)
        ax.axis("off")

    # hide empties
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    # one shared colorbar that cooperates with constrained_layout
    if ims:
        fig.colorbar(ims[0], ax=axes.ravel().tolist(), location="right", shrink=0.9, pad=0.02)

    fig.suptitle(f"Parent Δ Masks (step {step})")
    fig.savefig(os.path.join(out2, f"step_{step:04d}.png"), dpi=140)
    plt.close(fig)

    # update cache after plotting
    for P in parents:
        prev[int(getattr(P, "id", -1))] = np.asarray(P.mask, np.float32).copy()
