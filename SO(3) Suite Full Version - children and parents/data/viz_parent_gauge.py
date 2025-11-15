# tools/viz_agents_from_snapshot.py
import os
import numpy as np

import matplotlib as mpl
mpl.use("Agg", force=True)
import matplotlib.pyplot as plt

# ---------- small helpers ----------

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def _mask_outline(mask: np.ndarray) -> np.ndarray:
    core = np.asarray(mask, bool)
    if core.ndim != 2 or core.size == 0:
        return np.zeros_like(core, dtype=bool)
    up    = np.pad(core[1: , : ], ((0,1),(0,0)), constant_values=False)
    down  = np.pad(core[:-1, : ], ((1,0),(0,0)), constant_values=False)
    left  = np.pad(core[: , 1: ], ((0,0),(0,1)), constant_values=False)
    right = np.pad(core[: , :-1], ((0,0),(1,0)), constant_values=False)
    interior = core & up & down & left & right
    return core & (~interior)

def _save_field(img, outline, title, outpath):
    _ensure_dir(os.path.dirname(outpath))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.asarray(img, np.float32), origin="lower", cmap="viridis")
    if outline is not None and np.any(outline):
        ax.contour(outline.astype(np.float32), levels=[0.5], colors="white", linewidths=0.5)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)

def _trace(mat):   # (...,K,K) -> (...)
    m = np.asarray(mat, np.float32)
    return np.trace(m, axis1=-2, axis2=-1)

def _frob(mat):    # (...,K,K) -> (...)
    m = np.asarray(mat, np.float32)
    return np.sqrt(np.sum(m*m, axis=(-2, -1)))

def _safe_logdet(mat, eps=1e-6):
    m = np.asarray(mat, np.float32)
    H, W, K, _ = m.shape
    flat = m.reshape(-1, K, K)
    out = np.empty((H*W,), dtype=np.float32)
    for i, M in enumerate(flat):
        Ms = 0.5*(M + M.T)
        try:
            sgn, ld = np.linalg.slogdet(Ms)
            if sgn > 0:
                out[i] = float(ld); continue
        except Exception:
            pass
        w = np.linalg.eigvalsh(Ms)
        w = np.clip(w, eps, None)
        out[i] = float(np.sum(np.log(w)))
    return out.reshape(H, W)

# ---------- main ----------

def plot_agents_from_npz(
    npz_path: str,
    outdir: str = "plots",
    *,
    level: int | None = None,   # e.g., 0 for children, 1 for parents
    ids: list[int] | None = None,
    mask_thresh: float = 1e-3
) -> None:
    """
    Visualize everything imagable from a step snapshot:
      - φ, φ̃ norms
      - μ_q, μ_p norms
      - Σ_q, Σ_p trace & logdet
      - Φ, Φ̃ Frobenius (if present in the file)
    Selection by `level` or explicit `ids`. If both provided, uses intersection.
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(npz_path)

    data = np.load(npz_path, allow_pickle=True)

    # required metadata (added by snapshot patch)
    agent_ids    = data.get("agent_ids", None)
    agent_levels = data.get("agent_levels", None)

    if agent_ids is None or agent_levels is None:
        raise ValueError("snapshot is missing 'agent_ids' / 'agent_levels'; update dump_snapshot_npz first.")

    agent_ids    = np.asarray(agent_ids).astype(np.int32)
    agent_levels = np.asarray(agent_levels).astype(np.int16)
    N = int(agent_ids.shape[0])

    # tensors (optional)
    mask      = data.get("mask", None)           # (N,H,W)
    phi       = data.get("phi", None)            # (N,H,W,3)
    phi_model = data.get("phi_model", None)      # (N,H,W,3)
    mu_q      = data.get("mu_q", None)           # (N,H,W,Kq)
    mu_p      = data.get("mu_p", None)           # (N,H,W,Kp)
    sigma_q   = data.get("sigma_q", None)        # (N,H,W,Kq,Kq)
    sigma_p   = data.get("sigma_p", None)        # (N,H,W,Kp,Kp)
    Phi       = data.get("bundle_morphism_q_to_p", None)   # optional
    Phi_t     = data.get("bundle_morphism_p_to_q", None)   # optional

    # choose indices to render
    idxs = np.arange(N, dtype=np.int32)
    if level is not None:
        idxs = idxs[agent_levels[idxs] == int(level)]
    if ids is not None:
        ids = set(int(x) for x in ids)
        idxs = np.array([i for i in idxs if int(agent_ids[i]) in ids], dtype=np.int32)

    outdir = _ensure_dir(outdir)

    print(f"[viz] {os.path.basename(npz_path)}: rendering {len(idxs)} agents "
          f"(levels={sorted(set(int(agent_levels[i]) for i in idxs))})")

    for i in idxs:
        pid = int(agent_ids[i])
        base = os.path.join(outdir, f"agent_{pid:03d}_L{int(agent_levels[i])}")
        m_i = (np.asarray(mask[i]) > float(mask_thresh)) if (mask is not None) else None
        outline = _mask_outline(m_i) if isinstance(m_i, np.ndarray) else None

        # --- φ, φ̃ norms ---
        if phi is not None:
            _save_field(np.linalg.norm(phi[i], axis=-1), outline, f"agent {pid} ‖φ‖", f"{base}_phi_norm.png")
        if phi_model is not None:
            _save_field(np.linalg.norm(phi_model[i], axis=-1), outline, f"agent {pid} ‖φ̃‖", f"{base}_phi_tilde_norm.png")

        # --- μ norms ---
        if mu_q is not None:
            _save_field(np.linalg.norm(mu_q[i], axis=-1), outline, f"agent {pid} ‖μ_q‖", f"{base}_mu_q_norm.png")
        if mu_p is not None:
            _save_field(np.linalg.norm(mu_p[i], axis=-1), outline, f"agent {pid} ‖μ_p‖", f"{base}_mu_p_norm.png")

        # --- Σ trace + logdet ---
        if sigma_q is not None:
            _save_field(_trace(sigma_q[i]),   outline, f"agent {pid} tr(Σ_q)",    f"{base}_sigma_q_trace.png")
            _save_field(_safe_logdet(sigma_q[i]), outline, f"agent {pid} logdet(Σ_q)", f"{base}_sigma_q_logdet.png")
        if sigma_p is not None:
            _save_field(_trace(sigma_p[i]),   outline, f"agent {pid} tr(Σ_p)",    f"{base}_sigma_p_trace.png")
            _save_field(_safe_logdet(sigma_p[i]), outline, f"agent {pid} logdet(Σ_p)", f"{base}_sigma_p_logdet.png")

        # --- Φ maps (if present in the snapshot) ---
        if Phi is not None:
            _save_field(_frob(Phi[i]), outline, f"agent {pid} ‖Φ‖_F", f"{base}_Phi_frob.png")
        if Phi_t is not None:
            _save_field(_frob(Phi_t[i]), outline, f"agent {pid} ‖Φ̃‖_F", f"{base}_Phi_tilde_frob.png")

if __name__ == "__main__":
    # Example usages:
    
    plot_agents_from_npz("checkpoints/fields/step_0059.npz", outdir="viz_step24_parents", level=1)
   
    plot_agents_from_npz("checkpoints/fields/step_0059.npz", outdir="viz_step24_children", level=0)
   # plot_agents_from_npz("checkpoints/fields/step00024.npz", outdir="viz_step24_sel", ids=[3, 7, 1001])

    pass
