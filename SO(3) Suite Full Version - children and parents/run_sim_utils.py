from __future__ import annotations
"""
Created on Tue Aug 12 20:36:43 2025

@author: chris and christine
"""
# -*- coding: utf-8 -*-
"""run_sim_utils — streamlined"""

import os, pickle
from typing import Dict, Tuple
import numpy as np
import core.config as config

from core.config_utils import set_config_from_params
from diagnostics import compute_global_metrics, print_energy_breakdown
from core.get_generators import get_generators
from core.agent_initialization import initialize_agents, initialize_agent_gradients
from runtime_context import ensure_runtime_defaults
from data.viz_parent_masks import save_parent_mask_frame  # only if keeping wrap_epilogue



# ------------------------------- Logging defaults ----------------------------
LOGGING_DEFAULTS = dict(
    enable_scalar=True,
    enable_fields=False,
    parent_snapshot_every=0,
    parent_snapshot_dir="parents",
    child_snapshot_every=0,
    child_snapshot_dir="children",
)



def _sync_ctx_config_from_global(ctx):
    try:
        import core.config as g
    except Exception:
        import config as g

    ctx.config = {
        "alpha": float(getattr(g, "alpha", 1.0)),
        "beta": float(getattr(g, "beta", 0.0)),                 # alignment (q)
        "beta_model": float(getattr(g, "beta_model", 0.0)),     # alignment (p)
        "feedback_weight": float(getattr(g, "feedback_weight", 0.0)),  # keep 0 if unused
        "belief_mass": float(getattr(g, "belief_mass", 0.0)),
        "model_mass": float(getattr(g, "model_mass", 0.0)),
        "curvature_weight": float(getattr(g, "curvature_weight", 0.0)),
        "model_curvature_weight": float(getattr(g, "model_curvature_weight", 0.0)),
        "energy_eps": float(getattr(g, "energy_eps", 1e-6)),
        "support_cutoff_eps": float(getattr(g, "support_cutoff_eps", 1e-3)),
        "grid_dx": float(getattr(g, "grid_dx", 1.0)),
    }


def ensure_runtime(params: Dict):
    """Create/normalize the runtime context once and attach to params."""
    ctx = ensure_runtime_defaults(params.get("runtime_ctx", None))  # or your existing builder
    _sync_ctx_config_from_global(ctx)
    return ctx


def _prewarm_for_diagnostics(ctx, agents):
    try:
        from transport.transport_cache import Phi
    except Exception:
        return
    for a in agents:
        # Warm both directions Φ just once per agent
        try:
            Phi(ctx, a, kind="q_to_p")
            Phi(ctx, a, kind="p_to_q")
        except Exception:
            pass


import hashlib
import numpy as np

def array_fingerprint(arr: np.ndarray, tol: float = 1e-6) -> str:
    """
    Stable, cheap-ish fingerprint for numeric arrays.
    - Quantizes by `tol` (reduce churn from tiny float jitters)
    - Includes shape in the digest
    """
    a = np.asarray(arr, np.float32)
    if tol > 0.0:
        q = np.round(a / tol) * tol
    else:
        q = a
    h = hashlib.blake2b(digest_size=8)
    h.update(q.tobytes())
    h.update(str(q.shape).encode("utf-8"))
    return h.hexdigest()

def cfg_digest(d: dict, keys: tuple[str, ...], tol: float = 1e-9) -> tuple:
    """
    Stable digest for the subset of config entries that actually affect compute.
    Floats are quantized to `tol`.
    """
    out = []
    for k in keys:
        v = d.get(k, None)
        if isinstance(v, float):
            v = float(np.round(v / tol) * tol)
        out.append((k, v))
    return tuple(out)



def log_and_viz_after_step(runtime_ctx, agents, step, outdir, params, logcfg,
                           parent_metrics, metric_log, num_cores):
    """
    Minimal, objective-faithful logging hook.
    - Uses ctx+cache diagnostics for energies (self, feedback, mass, curvature).
    - Leaves your existing field logging/viz toggles intact:
        logcfg["enable_scalar"], logcfg["enable_fields"], etc. (optional)
    - No global config access; everything via runtime_ctx.config if needed.
    """
    ctx = runtime_ctx
    _sync_ctx_config_from_global(ctx)  # <— reflect any runtime overrides
    _prewarm_for_diagnostics(ctx, agents)
    # ---- 1) Energies from the SAME primitives as the solver ----
    metrics_global = compute_global_metrics(ctx, agents)
    print_energy_breakdown(ctx, metrics_global)

    # ---- 2) Optional: keep your existing field snapshots/viz here ----
    # If your older implementation wrote field dumps or gifs, you can
    # keep calling those helpers behind your existing flags.
    try:
        enable_fields = bool(logcfg.get("enable_fields", False))
    except Exception:
        enable_fields = False

    if enable_fields:
        # If you have a lightweight snapshot helper already, call it here.
        # e.g., dump_snapshot_npz(agents, step, outdir, params=params)
        pass

    # ---- 3) Optional: push per-step scalars into your trackers ----
    if bool(logcfg.get("enable_scalar", True)):
        metric_log.append({
            "step": int(getattr(ctx, "global_step", step)),
            "E_total": float(metrics_global.get("E_total", 0.0)),
            "mean_total": float(metrics_global.get("mean_total", 0.0)),
            "self_sum": float(metrics_global.get("self_sum", 0.0)),
            "feedback_sum": float(metrics_global.get("feedback_sum", 0.0)),
            "mass_q_sum": float(metrics_global.get("mass_q_sum", 0.0)),
            "mass_p_sum": float(metrics_global.get("mass_p_sum", 0.0)),
            "curv_q_sum": float(metrics_global.get("curv_q_sum", 0.0)),
            "curv_p_sum": float(metrics_global.get("curv_p_sum", 0.0)),
        })












def load_checkpoint(outdir: str):
    # prefer compressed if present
    for name in ("checkpoint.pkl.xz", "checkpoint.pkl"):
        p = os.path.join(outdir, name)
        if os.path.exists(p):
            if name.endswith(".xz"):
                import lzma
                with lzma.open(p, "rb") as f:
                    return pickle.load(f), name
            with open(p, "rb") as f:
                return pickle.load(f), name
    # fall back to last step_XXXX
    step_files = sorted([f for f in os.listdir(outdir) if f.startswith("step_") and (f.endswith(".pkl") or f.endswith(".pkl.xz"))])
    if not step_files: return (None, None)
    last = step_files[-1]
    if last.endswith(".xz"):
        import lzma
        with lzma.open(os.path.join(outdir, last), "rb") as f:
            return pickle.load(f), last
    with open(os.path.join(outdir, last), "rb") as f:
        return pickle.load(f), last


def find_resume_step(outdir: str) -> int:
    step_files = sorted([f for f in os.listdir(outdir) if f.startswith("step_") and (f.endswith(".pkl") or f.endswith(".pkl.xz"))])
    if not step_files: return 0
    last = step_files[-1]
    num = last.replace("step_", "").replace(".pkl.xz","").replace(".pkl","")
    try: return int(num) + 1
    except: return 0




def merge_logging_cfg(params_or_cfg) -> Dict:
    cfg = {}
    if hasattr(params_or_cfg, "get"):       # dict-like
        cfg = params_or_cfg.get("logging", {}) or {}
    elif hasattr(params_or_cfg, "__dict__"): # module-like (config)
        cfg = getattr(params_or_cfg, "logging", {}) or {}
    out = dict(LOGGING_DEFAULTS)
    out.update(cfg)
    return out


# ------------------------------ Generator prep ------------------------------

def prepare_generators(params: Dict) -> Tuple[np.ndarray, np.ndarray]:
    Gq = params.get("generators_q")
    Gp = params.get("generators_p")
    if Gq is None:
        Gq, _ = get_generators(getattr(config, "group_name", "so3"), getattr(config, "K_q", getattr(config, "K", 3)), return_meta=True)
        params["generators_q"] = Gq
    if Gp is None:
        Gp, _ = get_generators(getattr(config, "group_name", "so3"), getattr(config, "K_p", getattr(config, "K", 3)), return_meta=True)
        params["generators_p"] = Gp
    return Gq, Gp


# -------------------------- Init / resume helpers ---------------------------

def initialize_or_resume(outdir: str, resume: bool, params: Dict, clear_agent_logs: bool=True):
    """Return (agents, step_start)."""
    if resume:
        agents, _ = load_checkpoint(outdir)
        step_start = find_resume_step(outdir)
    else:
        agents = initialize_agents(
            seed=getattr(config, "seed", None),
            domain_size=config.domain_size,
            N=config.N,
            K_q=config.K_q,
            K_p=config.K_p,
            lie_algebra_dim=3,
            visualize=False,
            fixed_location=getattr(config, "fixed_location", False), 
        )
        step_start = 0
        for A in agents:
            initialize_agent_gradients(A, config)
        if clear_agent_logs:
            for A in agents:
                if hasattr(A, "metrics_log"):
                    A.metrics_log.clear()
    return agents, step_start




# -------- compact NPZ snapshots (masks only; tiny) --------

def snapshot_parents_npz_simple(parents, save_path):
    """
    Minimal parent snapshot:
      - ids: (P,)
      - masks: (P,H,W) if same-shape; else masks_obj: array(object)
    """
    ids, masks = [], []
    for idx, P in enumerate(parents or ()):
        ids.append(int(getattr(P, "id", idx)))
        masks.append(np.asarray(getattr(P, "mask", 0.0), np.float32))
    out = {"ids": np.asarray(ids, dtype=np.int32)}
    shapes = {m.shape for m in masks if m is not None}
    if len(shapes) == 1 and len(masks) > 0:
        out["masks"] = np.stack(masks, axis=0)
    else:
        out["masks_obj"] = np.array(masks, dtype=object)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, **out)

def snapshot_children_npz_simple(children, save_path):
    """Minimal child snapshot: ids + masks."""
    ids, masks = [], []
    for idx, C in enumerate(children or ()):
        ids.append(int(getattr(C, "id", idx)))
        masks.append(np.asarray(getattr(C, "mask", 0.0), np.float32))
    out = {"ids": np.asarray(ids, dtype=np.int32)}
    shapes = {m.shape for m in masks if m is not None}
    if len(shapes) == 1 and len(masks) > 0:
        out["masks"] = np.stack(masks, axis=0)
    else:
        out["masks_obj"] = np.array(masks, dtype=object)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, **out)

def write_mask_snapshots(runtime_ctx, agents, step, outdir, logcfg):
    """Handles parent/child mask-only snapshots, throttled by logcfg."""
    snap_every  = int(logcfg.get("parent_snapshot_every", getattr(config, "parent_snapshot_every", 0)) or 0)
    snap_dir    = logcfg.get("parent_snapshot_dir", getattr(config, "parent_snapshot_dir", "parents"))
    child_every = int(logcfg.get("child_snapshot_every", getattr(config, "child_snapshot_every", 0)) or 0)
    child_dir   = logcfg.get("child_snapshot_dir", getattr(config, "child_snapshot_dir", "children"))

    

    if child_every > 0 and (step % child_every) == 0 and agents:
        cdir = os.path.join(outdir, child_dir); os.makedirs(cdir, exist_ok=True)
        try:
            snapshot_children_npz_simple(agents, os.path.join(cdir, f"children_step_{step:04d}.npz"))
        except Exception as e:
            print(f"[SNAP][CHILDREN][ERROR] step {step}: {e}")










# run_sim_utils.py
def save_step(agents, outdir, step, *, save_every=1, compress=True):
    import pickle, os
    os.makedirs(outdir, exist_ok=True)
    ckpt_path = os.path.join(outdir, "checkpoint.pkl")
    if compress:
        import lzma
        with lzma.open(ckpt_path + ".xz", "wb", preset=3) as f:
            pickle.dump(agents, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(ckpt_path, "wb") as f:
            pickle.dump(agents, f, protocol=pickle.HIGHEST_PROTOCOL)

    if (step is not None) and (step % int(save_every) == 0):
        step_base = os.path.join(outdir, f"step_{step:04d}.pkl")
        if compress:
            import lzma
            with lzma.open(step_base + ".xz", "wb", preset=3) as f:
                pickle.dump(agents, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(step_base, "wb") as f:
                pickle.dump(agents, f, protocol=pickle.HIGHEST_PROTOCOL)







    

# --- fix: checkpoint save_step wrong kw ---------------------------------------
def checkpoint_step(agents, outdir):
    """Optional pre-checkpoint compaction; then save a non-step checkpoint."""
    for A in agents:
        if hasattr(A, "clear_omega_cache"):   A.clear_omega_cache()
        if hasattr(A, "clear_fisher_caches"): A.clear_fisher_caches()
    save_step(agents, outdir, step=None)  # filename handled by save_step



def wrap_epilogue(runtime_ctx, agents, outdir, parent_metrics, metric_log):
    import os, pickle
    import numpy as np
    try:
        import pandas as pd
    except Exception:
        pd = None


    # --- 1) Save the metrics log (pickle) + optional per-agent CSV ---
    if metric_log:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "metric_log.pkl"), "wb") as f:
            pickle.dump(metric_log, f)

        # flatten per-agent step logs if present
        rows = [{"agent": getattr(A, "id", -1), **e}
                for A in agents
                for e in (getattr(A, "metrics_log", None) or [])]
        if rows and pd is not None:
            pd.DataFrame(rows).to_csv(os.path.join(outdir, "per_agent_metrics.csv"), index=False)

    
    # --- 4) Optional parent metrics CSV ---
    if parent_metrics and pd is not None:
        try:
            pd.DataFrame(parent_metrics).to_csv(os.path.join(outdir, "parent_metrics.csv"), index=False)
        except Exception:
            pass
