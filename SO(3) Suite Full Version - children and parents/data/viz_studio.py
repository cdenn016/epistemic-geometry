

from __future__ import annotations

# --- Headless & thread caps ---------------------------------------------------
import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# --- Imports ------------------------------------------------------------------

import math
import pickle
import warnings
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


try:
    import core.config as config
    IMSHOW_ORIGIN  = getattr(config, "imshow_origin",  "lower")
    PLOT_TRANSPOSE = getattr(config, "plot_transpose", True)   # <— default True fixes your case
    PLOT_FLIPX     = getattr(config, "plot_flipx",     False)
    PLOT_FLIPY     = getattr(config, "plot_flipy",     False)
except Exception:
    IMSHOW_ORIGIN, PLOT_TRANSPOSE, PLOT_FLIPX, PLOT_FLIPY = "lower", True, False, False

def _orient(A: np.ndarray) -> np.ndarray:
    """Consistently orient (H,W) scalars for imshow."""
    B = np.asarray(A)
    if PLOT_TRANSPOSE: B = B.T           # swap X/Y if stored as (W,H)
    if PLOT_FLIPX:     B = B[:, ::-1]
    if PLOT_FLIPY:     B = B[::-1, :]
    return B


# imageio: accept v2 or v3; prefer v2 writer API
_HAS_IMAGEIO = False
_HAS_IMAGEIO_WRITER = False
try:
    import imageio.v2 as iio  # v2-compatible
    _HAS_IMAGEIO = True
    _HAS_IMAGEIO_WRITER = hasattr(iio, "get_writer")
except Exception:
    try:
        import imageio as iio  # may be v3
        _HAS_IMAGEIO = True
        _HAS_IMAGEIO_WRITER = hasattr(iio, "get_writer")
    except Exception:
        iio = None
        _HAS_IMAGEIO = False
        _HAS_IMAGEIO_WRITER = False

# optional: RAM-aware worker cap
try:
    import psutil
except Exception:
    psutil = None

import multiprocessing as mp
from joblib import Parallel, delayed, parallel_backend

# --- Data containers -----------------------------------------------------------
@dataclass
class Snapshot:
    path: str
    step: int
    data: Dict[str, np.ndarray]

# --- Panels (default grid order) ----------------------------------------------
_DEFAULT_PANELS = [
    "phi_norm", "phi_model_norm",
    "belief_align", "model_align",
    "Fq_norm", "Fp_norm",
    "mu_q_norm", "mu_p_norm",
    "sigma_q_logdet", "sigma_p_logdet",
    "phi_morph_norm", "phi_tilde_morph_norm",
    "lambda_q_norm", "lambda_p_norm",
    "grad_phi_norm", "grad_phi_model_norm",
    
]

# --- Snapshot utilities --------------------------------------------------------
import re
_step_re = re.compile(r"step[_\-]?(\d+)", re.IGNORECASE)

def _parse_step_from_name(path: str) -> Optional[int]:
    stem = Path(path).stem
    m = _step_re.search(stem)
    return int(m.group(1)) if m else None

def _iter_snapshot_paths(snapshot_dir: str, limit: Optional[int] = None) -> Iterator[Tuple[int, Path]]:
    """Yield (step, path) sorted by numeric step without loading arrays."""
    files = sorted(Path(snapshot_dir).glob("*.npz"),
                   key=lambda p: int(_step_re.search(p.stem).group(1)) if _step_re.search(p.stem) else -1)
    if limit is not None:
        files = files[:limit]
    for f in files:
        yield _parse_step_from_name(str(f)) or -1, f


def _load_npz_as_dict(path: Path) -> Dict[str, np.ndarray]:
    """Safely load an .npz into a plain dict of arrays (no open handles)."""
    with np.load(path, allow_pickle=False) as npz:
        return {k: npz[k] for k in npz.files}


def _iter_snapshots_lazy(snapshot_dir: str, limit: Optional[int] = None):
    """Yield (step, dict-of-arrays) lazily; no leaked NpzFile objects."""
    for step, f in _iter_snapshot_paths(snapshot_dir, limit):
        yield step, _load_npz_as_dict(f)


def load_snapshots_dir(directory: str, pattern: str = "step*.npz") -> List[Snapshot]:
    snaps: List[Snapshot] = []
    for f in sorted(Path(directory).glob(pattern)):
        step = _parse_step_from_name(str(f))
        if step is None:
            continue
        snaps.append(Snapshot(str(f), step, dict(np.load(f, allow_pickle=False))))
    return snaps

# --- Scalar map extraction -----------------------------------------------------
def _robust_norm(arr: np.ndarray, pmin: float = 1.0, pmax: float = 99.0) -> Tuple[float, float]:
    a = np.asarray(arr)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0, 1.0
    lo = np.percentile(a, pmin)
    hi = np.percentile(a, pmax)
    if not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(a))
        hi = float(np.nanmax(a))
        if not np.isfinite(hi) or hi <= lo:
            hi = lo + 1.0
    return float(lo), float(hi)

def _logdet2d(M: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """M: (H,W,K,K) SPD -> (H,W) slogdet (float32-safe)."""
    H, W, K, _ = M.shape
    S = M.astype(np.float32, copy=False).reshape(-1, K, K)
    I = np.eye(K, dtype=np.float32)
    S = 0.5 * (S + np.swapaxes(S, -1, -2)) + eps * I[None, ...]
    _, ld = np.linalg.slogdet(S)
    out = np.zeros((H, W), dtype=np.float32)
    out.reshape(-1)[:] = ld
    return out

def compute_scalar_maps(npz: Dict[str, np.ndarray], agent_index: int = 0) -> Dict[str, np.ndarray]:
    """Derive scalar 2D fields for plotting/animation from snapshot dict."""
    import numpy as np

    maps: Dict[str, np.ndarray] = {}
    g = npz.get

    # ---------------- mask ----------------
    m = g("mask")
    if m is not None:
        maps["mask"] = m[agent_index] if m.ndim == 3 else m

    # ---------------- φ norms ----------------
    phi = g("phi")
    if phi is not None:
        maps["phi_norm"] = (np.linalg.norm(phi[agent_index], axis=-1) if phi.ndim == 4
                            else np.linalg.norm(phi, axis=-1))
    phi_m = g("phi_model")
    if phi_m is not None:
        maps["phi_model_norm"] = (np.linalg.norm(phi_m[agent_index], axis=-1) if phi_m.ndim == 4
                                  else np.linalg.norm(phi_m, axis=-1))

    # ---------------- alignment (mask-gated) ----------------
    ba = g("belief_align");  ma = g("model_align")
    if ba is not None:
        maps["belief_align"] = ba[agent_index] if ba.ndim == 3 else ba
    if ma is not None:
        maps["model_align"]  = ma[agent_index] if ma.ndim == 3 else ma

    # enforce mask gating for alignment (robust to older snapshots)
    if "mask" in maps:
        try:
            _mm = (maps["mask"] > 0).astype(np.float32)
            if "belief_align" in maps:
                maps["belief_align"] = np.asarray(maps["belief_align"], np.float32) * _mm
            if "model_align" in maps:
                maps["model_align"]  = np.asarray(maps["model_align"],  np.float32) * _mm
        except Exception:
            pass

    # ---------------- curvature norms ----------------
    Fq = g("Fq_norm");  Fp = g("Fp_norm")
    if Fq is not None: maps["Fq_norm"] = Fq[agent_index] if Fq.ndim == 3 else Fq
    if Fp is not None: maps["Fp_norm"] = Fp[agent_index] if Fp.ndim == 3 else Fp

    # ---------------- μ norms ----------------
    muq = g("mu_q");  mup = g("mu_p")
    if muq is not None:
        maps["mu_q_norm"] = (np.linalg.norm(muq[agent_index], axis=-1) if muq.ndim == 4
                             else np.linalg.norm(muq, axis=-1))
    if mup is not None:
        maps["mu_p_norm"] = (np.linalg.norm(mup[agent_index], axis=-1) if mup.ndim == 4
                             else np.linalg.norm(mup, axis=-1))

    # ---------------- Σ stats (logdet) ----------------
    Sq = g("sigma_q");  Sp = g("sigma_p")
    if Sq is not None:
        maps["sigma_q_logdet"] = _logdet2d(Sq[agent_index]) if Sq.ndim == 5 else _logdet2d(Sq)
    if Sp is not None:
        maps["sigma_p_logdet"] = _logdet2d(Sp[agent_index]) if Sp.ndim == 5 else _logdet2d(Sp)

    # ---------------- morphism/gauge extras (if present) ----------------
    for k in ("phi_morph_norm", "phi_tilde_morph_norm", "lambda_q_norm", "lambda_p_norm"):
        arr = g(k)
        if arr is not None:
            maps[k] = arr[agent_index] if arr.ndim == 3 else arr

    # ---------------- φ gradients ----------------
    G = g("grad_phi") or g("grad_phi_norm")
    if G is not None:
        # grad_phi: (N,H,W,3) -> norm over last axis; OR already a norm map
        if G.ndim == 4:
            maps["grad_phi_norm"] = np.linalg.norm(G[agent_index], axis=-1)
        elif G.ndim == 3:
            # already (N,H,W) scalar maps
            maps["grad_phi_norm"] = G[agent_index]
        else:
            # (H,W,3) or (H,W) single-agent
            maps["grad_phi_norm"] = (np.linalg.norm(G, axis=-1) if G.ndim == 3 else G)

    Gm = g("grad_phi_model") or g("grad_phi_model_norm")
    if Gm is not None:
        if Gm.ndim == 4:
            maps["grad_phi_model_norm"] = np.linalg.norm(Gm[agent_index], axis=-1)
        elif Gm.ndim == 3:
            maps["grad_phi_model_norm"] = Gm[agent_index]
        else:
            maps["grad_phi_model_norm"] = (np.linalg.norm(Gm, axis=-1) if Gm.ndim == 3 else Gm)

    # ---------------- μ/Σ gradients (NEW) ----------------
    # Expect raw grads as:
    #   grad_mu_q, grad_mu_p:  (N,H,W,K)  -> L2 over last axis
    #   grad_sigma_q, grad_sigma_p: (N,H,W,K,K) -> Frobenius over last two axes
    # Or pre-normed:
    #   grad_mu_q_norm, grad_mu_p_norm: (N,H,W) or (H,W)
    #   grad_sigma_q_norm, grad_sigma_p_norm: (N,H,W) or (H,W)
    def _maybe_vec_norm(key_raw, key_norm, out_key):
        A = g(key_raw) or g(key_norm)
        if A is None: return
        if A.ndim == 4:       # (N,H,W,K) or (N,H,W) already normed
            if A.shape[-1] > 1:
                maps[out_key] = np.linalg.norm(A[agent_index], axis=-1)
            else:
                maps[out_key] = A[agent_index, ..., 0]
        elif A.ndim == 3:     # (N,H,W) or (H,W,K)
            # Try interpret as (N,H,W) scalar first
            if A.shape[0] == npz["agent_ids"].shape[0] if "agent_ids" in npz else True:
                maps[out_key] = A[agent_index]
            else:
                # (H,W,K) single-agent vector -> norm
                maps[out_key] = np.linalg.norm(A, axis=-1)
        elif A.ndim == 2:     # (H,W) single-agent scalar map
            maps[out_key] = A
        else:
            # degenerate shapes ignored
            pass

    def _maybe_mat_fro(key_raw, key_norm, out_key):
        A = g(key_raw) or g(key_norm)
        if A is None: return
        if A.ndim == 5:       # (N,H,W,K,K)
            maps[out_key] = np.linalg.norm(A[agent_index], axis=(-2, -1))
        elif A.ndim == 3:     # (N,H,W) already a Frobenius norm map
            maps[out_key] = A[agent_index]
        elif A.ndim == 4:     # (H,W,K,K) single-agent
            maps[out_key] = np.linalg.norm(A, axis=(-2, -1))
        elif A.ndim == 2:     # (H,W) single-agent norm map
            maps[out_key] = A
        else:
            pass

    # μ grads
    _maybe_vec_norm("grad_mu_q", "grad_mu_q_norm", "grad_mu_q_norm")
    _maybe_vec_norm("grad_mu_p", "grad_mu_p_norm", "grad_mu_p_norm")

    # Σ grads
    _maybe_mat_fro("grad_sigma_q", "grad_sigma_q_norm", "grad_sigma_q_norm")
    _maybe_mat_fro("grad_sigma_p", "grad_sigma_p_norm", "grad_sigma_p_norm")

    return maps


# --- Static grid plotting ------------------------------------------------------
def _render_panel(ax, arr: np.ndarray, title: str, cmap: str = "viridis", robust: bool = True):
    if arr is None:
        ax.axis("off"); ax.set_title(f"{title} (missing)"); return
    vmin, vmax = _robust_norm(arr) if robust else (float(np.min(arr)), float(np.max(arr)))
    im = ax.imshow(_orient(arr), cmap=cmap, vmin=vmin, vmax=vmax,
               interpolation="nearest", origin=IMSHOW_ORIGIN, aspect="equal")
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def plot_grid_from_npz(npz: Dict[str, np.ndarray],
                       panels: List[str] = None,
                       suptitle: str = "",
                       figsize: Tuple[int,int] = (14, 10),
                       cmap: str = "viridis",
                       robust: bool = True,
                       agent_index: int = 0):
    if panels is None:
        panels = _DEFAULT_PANELS
    maps = compute_scalar_maps(npz, agent_index=agent_index)
    keys = [k for k in panels if k in maps]
    n = max(len(keys), 1); cols = min(3, n); rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    for k, ax in zip(keys, axes.flat): _render_panel(ax, maps[k], k, cmap=cmap, robust=robust)
    for ax in axes.flat[len(keys):]: ax.axis("off")
    if suptitle: fig.suptitle(suptitle)
    fig.tight_layout()
    return fig

# --- Animation primitives ------------------------------------------------------
def _gif_writer(out_path: str, fps: int):
    if not _HAS_IMAGEIO or not _HAS_IMAGEIO_WRITER:
        raise RuntimeError("imageio (v2 API) not available; install 'imageio' for GIFs")
    return iio.get_writer(out_path, mode="I", fps=fps)

def animate_separate_metrics(snapshot_dir: str,
                             out_dir: str,
                             metrics: Optional[List[str]] = None,
                             fps: int = 8,
                             dpi: int = 120,
                             cmap: str = "viridis",
                             agent_index: int = 0,
                             exclude: Tuple[str, ...] = ("mask",),
                             per_metric_percentiles: Optional[Dict[str, Tuple[float, float]]] = None,
                             limit: Optional[int] = None) -> List[str]:
    """
    Write one GIF per metric (mask excluded by default).
    vmin/vmax are estimated from a sampled subset of frames for stability.
    """
    # discover metrics from first frame
    try:
        step0, data0 = next(_iter_snapshots_lazy(snapshot_dir, limit=1))
    except StopIteration:
        print(f"[ANIM] no snapshots under {snapshot_dir}"); return []
    maps0 = compute_scalar_maps(data0, agent_index=agent_index)
    if metrics is None:
        metrics = [k for k in _DEFAULT_PANELS if k in maps0 and k not in exclude]
    metrics = [m for m in metrics if m not in exclude]
    if not metrics: return []

    # build robust vlims from a sample
    sample_paths = list(_iter_snapshot_paths(snapshot_dir, limit))
    if len(sample_paths) <= 256:
        sample_idxs = range(len(sample_paths))
    else:
        sample_idxs = np.linspace(0, len(sample_paths) - 1, 256, dtype=int)

    vlims: Dict[str, Tuple[float, float]] = {}
    for m in metrics:
        lows: List[float] = []; highs: List[float] = []
        pmin_default, pmax_default = ((0.1, 99.9) if m in ("belief_align","model_align") else (1.0, 99.0))
        pmin, pmax = (per_metric_percentiles.get(m) if per_metric_percentiles and m in per_metric_percentiles
                      else (pmin_default, pmax_default))
        for idx in sample_idxs:
            _, f = sample_paths[idx]
            with np.load(f, allow_pickle=False) as npz:
                v = compute_scalar_maps({k: npz[k] for k in npz.files}, agent_index=agent_index).get(m)
            if v is None: continue
            finite = v[np.isfinite(v)]
            if finite.size:
                lows.append(float(np.percentile(finite, pmin)))
                highs.append(float(np.percentile(finite, pmax)))
        if lows and highs:
            lo, hi = float(np.median(lows)), float(np.median(highs))
            if not np.isfinite(hi) or hi <= lo: hi = lo + 1e-6
            vlims[m] = (lo, hi)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    def _fig_to_rgb(fig) -> np.ndarray:
        fig.canvas.draw(); w, h = fig.canvas.get_width_height()
        return np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)

    for m in metrics:
        out_path = str(Path(out_dir) / f"{m}.gif")
        if not _HAS_IMAGEIO: 
            print(f"[ANIM] imageio missing; cannot write {out_path}"); 
            continue
        with _gif_writer(out_path, fps=fps) as writer:
            for step, data in _iter_snapshots_lazy(snapshot_dir, limit):
                plt.close("all")
                maps = compute_scalar_maps(data, agent_index=agent_index)
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                if m not in maps:
                    ax.axis("off"); ax.set_title(f"{m} (missing)")
                else:
                    vmin, vmax = vlims.get(m, (None, None))
                    im = ax.imshow(_orient(maps[m]), cmap=cmap, vmin=vmin, vmax=vmax,
                                   interpolation="nearest", origin=IMSHOW_ORIGIN, aspect="equal")
                    ax.set_title(f"{m} â€” step {step}")
                    ax.set_xticks([]); ax.set_yticks([])
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                fig.set_dpi(dpi)
                writer.append_data(_fig_to_rgb(fig))
                plt.close(fig)
        written.append(out_path)

    if written:
        print(f"[ANIM] wrote {len(written)} GIFs to {out_dir}")
    return written

def animate_from_dir(snapshot_dir: str,
                     out_path: str,
                     panels: List[str] = None,
                     fps: int = 8,
                     dpi: int = 120,
                     cmap: str = "viridis",
                     robust: bool = True,
                     limit: Optional[int] = None,
                     agent_index: int = 0) -> str:
    """
    Streamed animation of a full panel grid to GIF (or MP4 via ffmpeg fallback).
    """
    want_gif = out_path.lower().endswith(".gif")

    def _fig_to_rgb(fig) -> np.ndarray:
        fig.canvas.draw(); w, h = fig.canvas.get_width_height()
        return np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)

    if want_gif:
        if not _HAS_IMAGEIO: raise RuntimeError("imageio not available; install 'imageio'")
        any_frame = False
        with _gif_writer(out_path, fps=fps) as writer:
            for step, data in _iter_snapshots_lazy(snapshot_dir, limit):
                any_frame = True; plt.close("all")
                
                fig = plot_grid_from_npz(data, panels=panels, suptitle=f"step {step}",
                                         cmap=cmap, robust=robust, agent_index=agent_index)
                fig.set_dpi(dpi); writer.append_data(_fig_to_rgb(fig)); plt.close(fig)
        if not any_frame: raise FileNotFoundError(f"No snapshots under {snapshot_dir}")
    else:
        tmpdir = Path(snapshot_dir) / "_frames_tmp"; tmpdir.mkdir(exist_ok=True)
        pngs: List[str] = []; any_frame = False
        try:
            for t, (step, data) in enumerate(_iter_snapshots_lazy(snapshot_dir, limit)):
                any_frame = True; plt.close("all")
                
                fig = plot_grid_from_npz(data, panels=panels, suptitle=f"step {step}",
                                         cmap=cmap, robust=robust, agent_index=agent_index)
                p = tmpdir / f"frame_{t:05d}.png"; fig.savefig(p, dpi=dpi, bbox_inches="tight")
                pngs.append(str(p)); plt.close(fig)
            if not any_frame: raise FileNotFoundError(f"No snapshots under {snapshot_dir}")
            cmd = f'ffmpeg -y -r {fps} -i "{tmpdir.as_posix()}/frame_%05d.png" -vcodec libx264 -pix_fmt yuv420p "{out_path}"'
            subprocess.run(shlex.split(cmd), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        except Exception as e:
            if _HAS_IMAGEIO:
                warnings.warn(f"ffmpeg failed ({e}); writing GIF instead", RuntimeWarning)
                gif_path = str(Path(out_path).with_suffix(".gif"))
                with _gif_writer(gif_path, fps=fps) as writer:
                    for p in pngs: writer.append_data(iio.imread(p))
                out_path = gif_path
            else:
                raise
        finally:
            for p in pngs:
                try: os.remove(p)
                except Exception: pass
            try: tmpdir.rmdir()
            except Exception: pass
    print(f"[ANIM] wrote {out_path}")
    return out_path

# --- Metric log loading & plots ----------------------------------------------
def _find_metric_log(start: str | Path) -> Optional[Path]:
    """
    Resolve metric log robustly (accepts metric_log.pkl or metrics_log.pkl).
    Search: start, parent, */checkpoints siblings, CWD, env(METRIC_LOG_FILE/DIR).
    """
    names = ("metric_log.pkl", "metrics_log.pkl")
    s = Path(start).resolve()
    bases: List[Path] = [s if s.is_dir() else s.parent, (s if s.is_dir() else s.parent).parent,
                         (s if s.is_dir() else s.parent) / "checkpoints",
                         ((s if s.is_dir() else s.parent).parent) / "checkpoints", Path.cwd()]
    env_file = os.getenv("METRIC_LOG_FILE")
    if env_file and Path(env_file).exists(): return Path(env_file)
    env_dir = os.getenv("METRIC_LOG_DIR")
    if env_dir: bases.insert(0, Path(env_dir))
    for b in bases:
        try:
            for n in names:
                p = b / n
                if p.exists(): return p
        except Exception:
            pass
    return None

def load_metric_log(outdir: str | Path) -> List[dict]:
    pkl = _find_metric_log(outdir)
    if pkl and pkl.exists():
        with open(pkl, "rb") as f: return pickle.load(f)
    csv = (pkl.parent if pkl else Path(outdir)) / "per_agent_metrics.csv"
    if csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(csv)
            numeric = [c for c in df.columns if df[c].dtype != "O" and c not in ("agent",)]
            g = df.groupby("step")[numeric].mean().reset_index()
            return g.to_dict(orient="records")
        except Exception as e:
            warnings.warn(f"Failed to aggregate per_agent_metrics.csv: {e}")
    return []

def plot_energy_timeseries(outdir: str,
                           keys: List[str] = None,
                           savepath: Optional[str] = None,
                           figsize: Tuple[int,int] = (10,6)):
    recs = load_metric_log(outdir)
    if not recs: raise FileNotFoundError(f"No metric logs found under {outdir}")
    if keys is None:
        keys = ["e_self","e_feedback","e_align","e_mod_align","e_curv","e_curv_mod","e_mass","e_mass_mod","total_energy"]
    steps = [r.get("step", i) for i, r in enumerate(recs)]
    plt.figure(figsize=figsize)
    for k in keys:
        vals = [r.get(k, np.nan) for r in recs]
        if np.all(np.isnan(vals)): continue
        plt.plot(steps, vals, label=k, lw=1.8)
    plt.xlabel("step"); plt.ylabel("energy"); plt.title("Energies over time")
    plt.grid(alpha=0.3); plt.legend(ncol=2, fontsize=9)
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=140, bbox_inches="tight")
        print(f"[PLOT] wrote {savepath}")
    return plt.gcf()

def plot_energy_timeseries_separate(outdir: str,
                                    keys: List[str] = None,
                                    save_dir: Optional[str] = None,
                                    figsize: Tuple[int,int] = (8,5),
                                    dpi: int = 140) -> List[str]:
    recs = load_metric_log(outdir)
    if not recs: raise FileNotFoundError(f"No metric logs found under {outdir}")
    if keys is None:
        keys = ["e_self","e_feedback","e_align","e_mod_align","e_curv","e_curv_mod","e_mass","e_mass_mod","total_energy"]
    steps = np.array([r.get("step", i) for i, r in enumerate(recs)])
    written: List[str] = []
    if save_dir: Path(save_dir).mkdir(parents=True, exist_ok=True)
    for k in keys:
        vals = np.array([r.get(k, np.nan) for r in recs], dtype=float)
        if np.all(np.isnan(vals)): continue
        plt.figure(figsize=figsize); plt.plot(steps, vals, lw=2.0)
        plt.xlabel("step"); plt.ylabel(k); plt.title(k.replace("_"," ")); plt.grid(alpha=0.3)
        if save_dir:
            fpath = str(Path(save_dir) / f"{k}.png")
            plt.savefig(fpath, dpi=dpi, bbox_inches="tight")
            written.append(fpath); plt.close()
    if save_dir and written: print(f"[PLOT] wrote {len(written)} figures to {save_dir}")
    return written

# --- Parallel GIFs per metric -------------------------------------------------
def _list_metrics_from_first_snapshot(snapshot_dir: str, exclude=()):
    snaps = load_snapshots_dir(snapshot_dir)
    if not snaps: return []
    maps0 = compute_scalar_maps(snaps[0].data, agent_index=0)
    return [k for k in maps0.keys() if k not in exclude]

def animate_metrics_parallel(snapshot_dir: str,
                             out_dir: str,
                             *,
                             metrics=None,
                             exclude=("mask",),
                             fps=15,
                             dpi=130,
                             agent_index=0,
                             per_metric_percentiles=None,
                             limit=None,
                             n_jobs=-1,
                             backend="threading",
                             verbose=10):
    """
    Render each metric into its own subfolder as a GIF, in parallel.
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    if metrics is None:
        metrics = _list_metrics_from_first_snapshot(snapshot_dir, exclude=exclude)
    if not metrics:
        print("[parallel] no metrics found to render"); return

    if n_jobs in (-1, None):
        n_jobs = max(1, min(6, (mp.cpu_count() or 1) // 3 or 1))
    if psutil is not None:
        try:
            avail = psutil.virtual_memory().available
            n_jobs = min(n_jobs, max(1, int(avail // (400 * 1024 * 1024))))  # ~400MB per worker
        except Exception:
            pass

    import shutil
    def _render_one_metric(metric: str):
        metric_dir = out / metric; metric_dir.mkdir(parents=True, exist_ok=True)
        animate_separate_metrics(snapshot_dir, str(metric_dir),
                                 metrics=[metric], fps=fps, dpi=dpi, agent_index=agent_index,
                                 exclude=(), per_metric_percentiles=per_metric_percentiles, limit=limit)
        # flat copy-back for convenience
        src = metric_dir / f"{metric}.gif"; dst = out / f"{metric}.gif"
        try:
            if src.exists(): shutil.copy2(src, dst)
        except Exception as e:
            print(f"[parallel] copy-back failed for {metric}: {e}")
        return metric

    print(f"[parallel] rendering {len(metrics)} metrics with backend={backend}, n_jobs={n_jobs}")
    with parallel_backend(backend, n_jobs=n_jobs):
        Parallel(verbose=verbose)(delayed(_render_one_metric)(m) for m in metrics)

# --- Lightweight helpers for the sim loop ------------------------------------
def render_latest_frame_from_latest_snapshot(snapshot_dir: str,
                                             out_path: str,
                                             panels=None,
                                             agent_index: int = 0,
                                             dpi: int = 110,
                                             robust: bool = True):
    snaps = load_snapshots_dir(snapshot_dir)
    if not snaps:
        print(f"[FRAME] no snapshots found under {snapshot_dir}"); return None
    snap = snaps[-1]
    fig = plot_grid_from_npz(snap.data, panels=panels, suptitle=f"step {snap.step}",
                             robust=robust, agent_index=agent_index)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    print(f"[FRAME] wrote {out_path}")
    return out_path

def update_studio_animations(snapshot_dir: str,
                             out_dir: str,
                             *,
                             agent_index: int = 0,
                             fps: int = 12,
                             dpi: int = 110,
                             limit: int | None = None,
                             backend: str = "threading",
                             n_jobs: int = -1,
                             per_metric_percentiles=None,
                             exclude=("mask",)):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    animate_metrics_parallel(snapshot_dir, out_dir, metrics=None, exclude=exclude,
                             fps=fps, dpi=dpi, agent_index=agent_index,
                             per_metric_percentiles=per_metric_percentiles,
                             limit=limit, n_jobs=n_jobs, backend=backend, verbose=5)



# --- Metrics summary + dashboard --------------------------------------------
def _metrics_dataframe(outdir: str | Path) -> pd.DataFrame:
    recs = load_metric_log(outdir)
    if not recs:
        raise FileNotFoundError(f"No metric logs found under {outdir}")
    df = pd.DataFrame(recs).copy()
    if "step" not in df.columns:
        df["step"] = np.arange(len(df))
    # expand per-level support if present
    def _lvl0_support(x):
        try:
            return x.get(0, {}).get("support_total", np.nan)
        except Exception:
            return np.nan
    if "levels" in df.columns:
        df["support_total"] = df["levels"].map(_lvl0_support)
    # derived
    tol = np.maximum(df["total_energy"].abs(), 1e-9)
    for k in ("e_self","e_align","e_mod_align","e_curv","e_curv_mod","e_mass","e_mass_mod"):
        if k in df.columns:
            df[f"{k}_pct"] = 100.0 * df[k] / tol
    if "e_align" in df.columns and "e_mod_align" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["align_ratio_q_over_p"] = df["e_align"] / np.where(df["e_mod_align"]==0, np.nan, df["e_mod_align"])
    # deltas (signed) and rolling medians
    for k in ("total_energy","e_self","e_align","e_mod_align"):
        if k in df.columns:
            df[f"d_{k}"] = df[k].diff()
            df[f"{k}_rollmed20"] = df[k].rolling(20, min_periods=3).median()
    return df

def print_metrics_summary(outdir: str | Path, last_n: int = 1) -> None:
    """
    Compact console summary for the last N records:
      - total E, per-term %, grad norms, support area, compute_ms
    """
    df = _metrics_dataframe(outdir)
    tail = df.tail(last_n)
    for _, r in tail.iterrows():
        step = int(r.get("step", -1))
        parts = []
        parts.append(f"step {step}")
        parts.append(f"E_total={r.get('total_energy', np.nan):.4e}")
        # per-term %
        for k in ("e_self_pct","e_align_pct","e_mod_align_pct","e_curv_pct","e_curv_mod_pct","e_mass_pct","e_mass_mod_pct"):
            if k in df.columns and pd.notnull(r.get(k)):
                parts.append(f"{k[:-4]}={r[k]:.2f}%")
        # grads
        if pd.notnull(r.get("grad_phi_norm", np.nan)):
            parts.append(f"||âˆ‡Ï†||={r['grad_phi_norm']:.3e}")
        if pd.notnull(r.get("grad_phi_model_norm", np.nan)):
            parts.append(f"||âˆ‡Ï†Ìƒ||={r['grad_phi_model_norm']:.3e}")
        # support + time
        if pd.notnull(r.get("support_total", np.nan)):
            parts.append(f"support={int(r['support_total'])}")
        if pd.notnull(r.get("compute_ms", np.nan)):
            parts.append(f"compute={int(r['compute_ms'])}ms")
        print("[METRICS] " + " | ".join(parts))

def plot_metrics_dashboard(outdir: str | Path, savepath: Optional[str] = None,
                           dpi: int = 140) -> str | None:
    """
    2x3 dashboard:
      (1) total energy (+ rolling median)
      (2) per-term % of total (self/align/curv/mass)
      (3) grad norms (Ï†, Ï†Ìƒ)
      (4) support area (lvl-0)
      (5) alignment terms (q vs p)
      (6) compute time (ms)
    """
    df = _metrics_dataframe(outdir)
    steps = df["step"].to_numpy()
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))

    # (1) total energy
    a = ax[0,0]
    a.plot(steps, df["total_energy"], label="total", lw=1.6)
    if "total_energy_rollmed20" in df.columns:
        a.plot(steps, df["total_energy_rollmed20"], label="rollmed20", lw=2.2, alpha=0.8)
    a.set_title("Total energy"); a.set_xlabel("step"); a.set_ylabel("E"); a.grid(alpha=0.3); a.legend()

    # (2) per-term %
    a = ax[0,1]
    for k,label in [("e_self_pct","self"),("e_align_pct","align_q"),("e_mod_align_pct","align_p"),
                    ("e_curv_pct","curv_q"),("e_curv_mod_pct","curv_p"),("e_mass_pct","mass_q"),("e_mass_mod_pct","mass_p")]:
        if k in df.columns:
            a.plot(steps, df[k], label=label, lw=1.2)
    a.set_title("Per-term % of total"); a.set_xlabel("step"); a.set_ylabel("%"); a.grid(alpha=0.3); a.legend(ncol=2, fontsize=8)

    # (3) gradient norms
    a = ax[0,2]
    if "grad_phi_norm" in df.columns: a.plot(steps, df["grad_phi_norm"], label="||âˆ‡Ï†||", lw=1.6)
    if "grad_phi_model_norm" in df.columns: a.plot(steps, df["grad_phi_model_norm"], label="||âˆ‡Ï†Ìƒ||", lw=1.6)
    a.set_yscale("log"); a.set_title("Gradient norms"); a.set_xlabel("step"); a.set_ylabel("norm"); a.grid(alpha=0.3, which="both"); a.legend()

    # (4) support area
    a = ax[1,0]
    if "support_total" in df.columns:
        a.plot(steps, df["support_total"], lw=1.6)
    a.set_title("Level-0 support area"); a.set_xlabel("step"); a.set_ylabel("pixels"); a.grid(alpha=0.3)

    # (5) alignment terms & ratio
    a = ax[1,1]
    if "e_align" in df.columns: a.plot(steps, df["e_align"], label="align_q", lw=1.4)
    if "e_mod_align" in df.columns: a.plot(steps, df["e_mod_align"], label="align_p", lw=1.4)
    if "align_ratio_q_over_p" in df.columns:
        yy = df["align_ratio_q_over_p"].replace([np.inf,-np.inf], np.nan)
        a2 = a.twinx(); a2.plot(steps, yy, label="q/p ratio", lw=1.0, alpha=0.7); a2.set_ylabel("q/p")
    a.set_title("Alignment"); a.set_xlabel("step"); a.grid(alpha=0.3); a.legend(loc="upper left")

    # (6) compute time
    a = ax[1,2]
    if "compute_ms" in df.columns:
        a.plot(steps, df["compute_ms"], lw=1.6)
    a.set_title("Compute time"); a.set_xlabel("step"); a.set_ylabel("ms"); a.grid(alpha=0.3)

    fig.tight_layout()
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
        print(f"[PLOT] wrote {savepath}")
        plt.close(fig)
        return savepath
    return None





# --- CLI / script entry -------------------------------------------------------
if __name__ == "__main__":
    PROJ_ROOT = Path(__file__).resolve().parents[1]               # â€¦/SO(3) suite
    SNAPSHOT_DIR = os.getenv("SNAPSHOT_DIR", str(PROJ_ROOT / "checkpoints" / "fields"))
    OUT_DIR      = os.getenv("STUDIO_OUT_DIR", str(PROJ_ROOT / "plots_and_gifs" / "studio"))

    # toggles
    DO_FRAME        = False
    DO_ANIM         = True
    fp16_snapshot   = False
    DO_TS_SEPARATE  = True
    DO_DASHBOARD    = True
    DO_SUMMARY      = True

    # options
    PANELS          = ["mask", *_DEFAULT_PANELS]
    PAR_BACKEND     = "loky"     # "loky" | "threading"
    NUM_WORKERS     = -1
    LIMIT_FRAMES    = None       # e.g., 2000
    FRAME_FILENAME  = str(Path(OUT_DIR) / "latest_frame.png")
    ANIM_DIR        = str(Path(OUT_DIR) / "gifs")
    FPS, DPI        = 50, 100
    ROBUST          = True
    AGENT_INDEX     = 3
    PER_METRIC_PERCENTILES = {"belief_align": (0.1, 99.9), "model_align": (0.1, 99.9)}

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    if DO_ANIM:
        animate_metrics_parallel(SNAPSHOT_DIR, ANIM_DIR, metrics=None, exclude=("mask",),
                                 fps=FPS, dpi=DPI, agent_index=AGENT_INDEX,
                                 per_metric_percentiles=PER_METRIC_PERCENTILES, limit=LIMIT_FRAMES,
                                 n_jobs=NUM_WORKERS, backend=PAR_BACKEND, verbose=10)

    if DO_FRAME:
        render_latest_frame_from_latest_snapshot(SNAPSHOT_DIR, FRAME_FILENAME,
                                                panels=PANELS, agent_index=AGENT_INDEX, 
                                                dpi=DPI, robust=ROBUST)

    if DO_DASHBOARD:
        dash_path = str(Path(OUT_DIR) / "metrics_dashboard.png")
        ts_dir_env = os.getenv("METRIC_LOG_DIR")
        ts_dir = Path(ts_dir_env) if ts_dir_env else Path(SNAPSHOT_DIR).resolve().parent
        try:
            plot_metrics_dashboard(ts_dir, savepath=dash_path, dpi=140)
        except FileNotFoundError:
            print(f"[DASH] skipped: no metric logs found under {ts_dir}")

    if DO_SUMMARY:
        ts_dir_env = os.getenv("METRIC_LOG_DIR")
        ts_dir = Path(ts_dir_env) if ts_dir_env else Path(SNAPSHOT_DIR).resolve().parent
        try:
            print_metrics_summary(ts_dir, last_n=1)
        except FileNotFoundError:
            print(f"[METRICS] skipped: no metric logs found under {ts_dir}")

    if DO_TS_SEPARATE:
        sep_dir = str(Path(OUT_DIR) / "energies")
        ts_dir_env = os.getenv("METRIC_LOG_DIR")
        ts_dir = Path(ts_dir_env) if ts_dir_env else Path(SNAPSHOT_DIR).resolve().parent
        try:
            plot_energy_timeseries_separate(ts_dir, save_dir=sep_dir)
        except FileNotFoundError:
            print(f"[TS] skipped (separate): no metric logs found under {ts_dir}")
