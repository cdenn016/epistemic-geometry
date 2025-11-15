# generalized_simulation.py
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import sys, time, pickle
import core.config as config
import shutil, os
from run_sim_utils import (
    prepare_generators, merge_logging_cfg, initialize_or_resume,
    ensure_runtime, log_and_viz_after_step,
    wrap_epilogue, set_config_from_params, save_step
)

from updates.update_rules import synchronous_step

def run_simulation(
    outdir="checkpoints", 
    resume=False,
    log_metrics=True,
    log_fields=True,
    num_cores=None,
    params=None,
    fresh_outdir=False,
    clear_agent_logs=True,
):
    # outdir rotation + readme
    if not resume and fresh_outdir and os.path.isdir(outdir):
        ts = time.strftime("%Y%m%d_%H%M%S")
        shutil.move(outdir, f"{outdir}_old_{ts}")
    os.makedirs(outdir, exist_ok=True)

    # params + cores
    try:
        import multiprocessing as _mp
        _avail = max(1, (_mp.cpu_count() or 1) - 1)
    except Exception:
        _avail = 8

    num_cores = _avail if num_cores is None else int(num_cores)
    num_cores = max(1, num_cores)

    params = {} if params is None else dict(params)

    config.n_jobs = num_cores

    # generators
    G_q, G_p = prepare_generators(params)

    # logging cfg
    logcfg = merge_logging_cfg(params if isinstance(params, dict) else vars(config))
    if not log_metrics: logcfg["enable_scalar"] = False
    if not log_fields:  logcfg["enable_fields"] = False
    if logcfg.get("enable_fields", False):
        os.makedirs(os.path.join(outdir, logcfg["full_field_outdir"]), exist_ok=True)

    # init or resume
    agents, step_start = initialize_or_resume(outdir, resume, params, clear_agent_logs=clear_agent_logs)

    # runtime context (mounted lvl-1 view)
    runtime_ctx = ensure_runtime(params)
    runtime_ctx.global_step = step_start  # initialize step index
    params["runtime_ctx"] = runtime_ctx   # expose ctx via params for legacy helpers

    # Early-exit if resuming past the configured steps
    if step_start >= int(getattr(config, "steps", step_start)):
        wrap_epilogue(runtime_ctx, agents, outdir, [], [])
        print("[DONE] Nothing to do: resume step >= total steps.")
        return agents, []

    # trackers
    parent_metrics, metric_log = [], []
    print(f"[RUN] Starting simulation at step {step_start} with {num_cores if (num_cores is not None) else 'auto'} cores")

    for step in range(step_start, config.steps):
        # keep global_step in sync BEFORE core update so gates use it
        runtime_ctx.global_step = step
        runtime_ctx.dirty.reset_step(step)
                
        print(f"\n[STEP {step}] -----------------------------\n")
        t0 = time.perf_counter()
        step_params = {**params, "step": step, "sanitize_strict_pre": False}

        # core synchronous update (ctx is threaded inside)
        agents = synchronous_step(
            agents, G_q, G_p,
            params=step_params,
            n_jobs=num_cores,
            runtime_ctx=runtime_ctx,
        )

        print(f"[TIMER] synchronous update: {time.perf_counter() - t0:.3f}s")

        # --- cache stats probe every 5 steps ---
        try:
            if step % 1 == 0:
                tc = getattr(runtime_ctx, "cache", None)
                if tc is not None and hasattr(tc, "full_stats"):
                    print("[cache]", tc.full_stats())
        except Exception as e:
            print(f"[cache] stats probe failed: {e}")

        # viz & metrics
        runtime_ctx.children_latest = agents
        
        log_and_viz_after_step(runtime_ctx, agents, step, outdir, params, logcfg,
                               parent_metrics, metric_log, num_cores)
        assert_energy_canaries(metric_log[-1], step=step)
        save_step(agents, outdir, step)
        print(f"[SAVE] Checkpoint --> {outdir}/step_{step:04d}.pkl")

    # epilogue
    wrap_epilogue(runtime_ctx, agents, outdir, parent_metrics, metric_log)
    print("[DONE] Simulation completed.")
    return agents, metric_log

# --- BEGIN: tiny calibration helpers -----------------------------------------
# diagnostics_sanity.py (you already have the file)
def assert_energy_canaries(stats: dict, *, step: int, max_growth=100.0):
    import numpy as np
    keys = ["self_sum","feedback_sum","align_q_sum","align_p_sum",
            "mass_q_sum","mass_p_sum","curv_q_sum","curv_p_sum","total_energy"]
    for k in keys:
        v = float(stats.get(k, 0.0))
        if not np.isfinite(v): raise FloatingPointError(f"[{step}] {k} not finite: {v}")
        if v < -1e-8:          raise AssertionError(f"[{step}] {k} < 0: {v}")


def run_tiny_calibration(steps: int = 5, seed: int = 123, H: int = 8, W: int = None):
    import os, sys, pathlib, importlib, importlib.util, random
    import numpy as np

    # 1) Deterministic seeding across common RNGs (cheap & safe)
    os.environ.setdefault("PYTHONHASHSEED", "0")
    random.seed(seed)
    np.random.seed(seed)

    # 2) Load a FRESH config module every call (avoid state carry-over)
    def _load_fresh_config():
        # Try import or reload known names
        for name in ("config", "core.config"):
            if name in sys.modules:
                return importlib.reload(sys.modules[name])
            try:
                return importlib.import_module(name)
            except ModuleNotFoundError:
                pass
        # Fallback: load config.py from same folder as this file
        here = pathlib.Path(__file__).resolve().parent
        cfg_path = here / "config.py"
        if not cfg_path.exists():
            raise ModuleNotFoundError("config not found (looked for 'config', 'core.config', and config.py next to generalized_simulation.py)")
        spec = importlib.util.spec_from_file_location("config", str(cfg_path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["config"] = mod
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod

    config = _load_fresh_config()

    # 3) Tiny, deterministic overrides
    W = H if W is None else W
    config.steps = int(steps)
    config.N = 1
    config.L = int(H)
    config.D = 2
    config.K_q = 3
    config.K_p = 5
    config.fixed_location = True
    config.agent_radius_range = (max(1, H // 3), max(1, H // 3))
    config.checkpoint_interval = 0
    config.num_cores = 1
    config.seed = int(seed)   # <<< crucial: initialize_agents will use this

    # 4) Run
    outdir = "checkpoints_tiny"
    agents, metric_log = run_simulation(
        outdir=outdir,
        resume=False,
        log_metrics=True,
        log_fields=False,
        num_cores=1,
        params={},
        fresh_outdir=True,
        clear_agent_logs=True,
    )
    return metric_log

if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1], "rb") as f:
            param_override = pickle.load(f)
        set_config_from_params(param_override)
    else:
        param_override = {}

    # Prepare runtime generators for convenience CLI usage
    from core.get_generators import get_generators
    
    G_q, meta_q = get_generators(config.group_name, config.K_q, return_meta=True)
    G_p, meta_p = get_generators(config.group_name, config.K_p, return_meta=True)
    runtime_params = {"generators_q": G_q, "generators_p": G_p}

    run_simulation(
        outdir="checkpoints",
        resume=False,
        log_metrics=True,
        log_fields=True,
        params=runtime_params,
    )



