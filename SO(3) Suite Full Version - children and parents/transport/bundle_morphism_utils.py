# -*- coding: utf-8 -*-

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional

from typing import Optional, Tuple
from core.omega import exp_lie_algebra_irrep
import numpy as np
from scipy.ndimage import gaussian_filter

import core.config as config
from transport.transport_cache import E_grid
from core.numerical_utils import safe_omega_inv



# =============================================================================
#                         Direct (gauge-covariant) morphisms
# =============================================================================




def compute_direct_morphisms(
    *,
    phi: np.ndarray,
    phi_model: np.ndarray,
    generators_q: np.ndarray,
    generators_p: np.ndarray,
    Phi_0: np.ndarray,
    Phi_tilde_0: np.ndarray,
    ctx: Optional[object] = None,
    group_name: Optional[str] = None,
):
    group = (group_name or getattr(config, "group_name", "so3")).lower()
    so3_orth = bool(getattr(config, "so3_irreps_are_orthogonal", True)) and group == "so3"

    phi      = np.asarray(phi,       np.float32)
    phi_model= np.asarray(phi_model, np.float32)
    Gq = np.asarray(generators_q, np.float32)
    Gp = np.asarray(generators_p, np.float32)

    if ctx is not None and getattr(ctx, "cache", None) is not None:
        class _A:
            __slots__=("phi","phi_model","generators_q","generators_p","id")
            def __init__(self, phi, phi_m, Gq, Gp): self.phi=phi; self.phi_model=phi_m; self.generators_q=Gq; self.generators_p=Gp; self.id=id(self)
        Aview = _A(phi, phi_model, Gq, Gp)
        Eq = E_grid(ctx, Aview, which="q").astype(np.float32, copy=False)
        Ep = E_grid(ctx, Aview, which="p").astype(np.float32, copy=False)
    else:
        Eq = exp_lie_algebra_irrep(phi,      Gq).astype(np.float32, copy=False)
        Ep = exp_lie_algebra_irrep(phi_model,Gp).astype(np.float32, copy=False)

    Eq_invT = np.swapaxes(Eq, -1, -2) if so3_orth else safe_omega_inv(Eq).astype(np.float32, copy=False)
    Ep_invT = np.swapaxes(Ep, -1, -2) if so3_orth else safe_omega_inv(Ep).astype(np.float32, copy=False)

    Kq, Kp = int(Eq.shape[-1]), int(Ep.shape[-1])
    Phi_0       = np.asarray(Phi_0,       np.float32)
    Phi_tilde_0 = np.asarray(Phi_tilde_0, np.float32)
    if Phi_0.shape[-2:] != (Kp, Kq):        Phi_0       = np.eye(Kp, Kq, dtype=np.float32)
    if Phi_tilde_0.shape[-2:] != (Kq, Kp):  Phi_tilde_0 = np.eye(Kq, Kp, dtype=np.float32)
    if Phi_0.ndim == 2:        Phi_0       = np.broadcast_to(Phi_0,       Ep.shape[:-2] + (Kp, Kq))
    if Phi_tilde_0.ndim == 2:  Phi_tilde_0 = np.broadcast_to(Phi_tilde_0, Eq.shape[:-2] + (Kq, Kp))

    Phi       = np.einsum("...ik,...kj,...jl->...il", Ep, Phi_0,       Eq_invT, optimize=True).astype(np.float32, copy=False)
    Phi_tilde = np.einsum("...ik,...kj,...jl->...il", Eq, Phi_tilde_0, Ep_invT, optimize=True).astype(np.float32, copy=False)
    return Phi, Phi_tilde






def gauge_covariant_transform_phi(
    phi_p: np.ndarray,
    phi_q: np.ndarray,
    Phi0: np.ndarray,
    G_q: np.ndarray,
    G_p: np.ndarray,
    *,
    ctx: Optional[object] = None,
    group_name: str = "so3",
) -> np.ndarray:
    # cached path
    if ctx is not None and getattr(ctx, "cache", None) is not None:
        class _Ap:  # minimal agent views for cache
            __slots__ = ("phi_model","generators_p","id","phi","generators_q")
            def __init__(self, phi_p, Gp): self.phi_model=np.asarray(phi_p,np.float32); self.generators_p=np.asarray(Gp,np.float32); self.id=id(self); self.phi=None; self.generators_q=None
        class _Aq:
            __slots__ = ("phi","generators_q","id","phi_model","generators_p")
            def __init__(self, phi_q, Gq): self.phi=np.asarray(phi_q,np.float32); self.generators_q=np.asarray(Gq,np.float32); self.id=id(self); self.phi_model=None; self.generators_p=None

        A_p = _Ap(phi_p, G_p)
        A_q = _Aq(phi_q, G_q)
        E_p = E_grid(ctx, A_p, which="p")  # (..., Kp, Kp)
        E_q = E_grid(ctx, A_q, which="q")  # (..., Kq, Kq)
    else:
        # fallback (no cache yet during initialization)
        E_p = exp_lie_algebra_irrep(np.asarray(phi_p, np.float32), np.asarray(G_p, np.float32))
        E_q = exp_lie_algebra_irrep(np.asarray(phi_q, np.float32), np.asarray(G_q, np.float32))

    Kp, Kq = int(E_p.shape[-1]), int(E_q.shape[-1])
    Phi0 = np.asarray(Phi0, np.float32)
    if Phi0.shape[-2:] != (Kp, Kq):
        Phi0 = np.eye(Kp, Kq, dtype=E_p.dtype)
    if Phi0.ndim == 2:
        lead = np.broadcast_shapes(E_p.shape[:-2], E_q.shape[:-2])
        Phi0 = np.broadcast_to(Phi0, (*lead, Kp, Kq))

    so3_orth = bool(getattr(config, "so3_irreps_are_orthogonal", True)) and str(group_name).lower() == "so3"
    E_q_inv = np.swapaxes(E_q, -1, -2) if so3_orth else safe_omega_inv(E_q).astype(np.float32, copy=False)

    Phi = np.einsum("...ik,...kj,...jl->...il", E_p, Phi0, E_q_inv, optimize=True)
    return Phi.astype(np.float32, copy=False)


# =============================================================================
#                             Init / Recompute on agent
# =============================================================================

def init_parent_morphisms(P, generators_q, generators_p, *, ctx: Optional[object] = None):
    """
    Ensure a newly spawned parent has BASE morphisms and valid φ-fields.
    - Sets identity Φ₀ (Kp×Kq) and Φ̃₀ (Kq×Kp) if missing/mismatched.
    - Stashes generators on the parent if not already present.
    - Does NOT store per-pixel bundle morphisms on the agent.
      (Φ and Φ̃ are computed lazily via transport_cache.Phi on demand.)
    - If ctx is provided, pre-warms Φ/Φ̃ in the central cache.
    """
    import numpy as np

    # infer dims from parent's fields
    try:
        Kq = int(np.asarray(P.mu_q_field).shape[-1])
        Kp = int(np.asarray(P.mu_p_field).shape[-1])
    except Exception:
        return
    if Kq <= 0 or Kp <= 0:
        return

    H, W = np.asarray(P.mu_q_field).shape[:2]
    dtype = getattr(P.mu_q_field, "dtype", np.float32)

    # algebra dims (fallbacks if generators not set yet)
    d_q = int(np.asarray(generators_q).shape[0]) if generators_q is not None else int(np.asarray(getattr(P, "phi", np.zeros(3))).shape[-1])
    d_p = int(np.asarray(generators_p).shape[0]) if generators_p is not None else int(np.asarray(getattr(P, "phi_model", np.zeros(3))).shape[-1])

    # base morphisms (identities with correct shapes)
    Phi0 = getattr(P, "Phi_0", None)
    if Phi0 is None or np.asarray(Phi0).shape[-2:] != (Kp, Kq):
        Phi0 = np.eye(Kp, Kq, dtype=dtype)
        P.Phi_0 = Phi0

    Phit0 = getattr(P, "Phi_tilde_0", None)
    if Phit0 is None or np.asarray(Phit0).shape[-2:] != (Kq, Kp):
        Phit0 = np.eye(Kq, Kp, dtype=dtype)
        P.Phi_tilde_0 = Phit0

    # stash generators (don’t overwrite if already present)
    if getattr(P, "generators_q", None) is None:
        P.generators_q = generators_q
    if getattr(P, "generators_p", None) is None:
        P.generators_p = generators_p

    # ensure phi fields exist (H,W,d)
    if getattr(P, "phi", None) is None or np.asarray(P.phi).ndim < 3:
        P.phi = np.zeros((H, W, d_q), dtype=dtype)
    if getattr(P, "phi_model", None) is None or np.asarray(P.phi_model).ndim < 3:
        P.phi_model = np.zeros((H, W, d_p), dtype=dtype)

    # optional: persist bases into CacheHub (if that API exists)
    if ctx is not None:
        try:
            from transport.transport_cache import set_morphism_bases
            set_morphism_bases(ctx, P, Phi0, Phit0)
        except Exception:
            pass

    # pre-warm Φ/Φ̃ in central cache (no agent writes)
    if ctx is not None:
        try:
            from transport.transport_cache import Phi as tc_Phi
            _ = tc_Phi(ctx, P, kind="q_to_p")
            _ = tc_Phi(ctx, P, kind="p_to_q")
        except Exception:
            pass

    # mark clean flag if present (we didn't write agent fields, only cache)
    if hasattr(P, "morphisms_dirty"):
        P.morphisms_dirty = False

    


def recompute_agent_morphisms(
    agent,
    *,
    generators_q: Optional[np.ndarray] = None,
    generators_p: Optional[np.ndarray] = None,
    allow_make_bases: bool = True,
    ctx: Optional[object] = None,
) -> None:
    """
    
    Uses CacheHub if ctx is provided.
    """
    # 0) Ensure generators
    if hasattr(agent, "ensure_generators"):
        agent.ensure_generators(generators_q, generators_p)
    Gq = getattr(agent, "generators_q", generators_q)
    Gp = getattr(agent, "generators_p", generators_p)
    if Gq is None or Gp is None:
        raise ValueError("recompute_agent_morphisms: generators_q/p required or must be present on agent")

    # 1) Fiber sizes
    try:
        Kq = int(agent.mu_q_field.shape[-1])
        Kp = int(agent.mu_p_field.shape[-1])
    except Exception as e:
        raise ValueError("recompute_agent_morphisms: agent.mu_q_field / mu_p_field missing or malformed") from e

    # 2) Base morphisms
    if getattr(agent, "Phi_0", None) is None:
        if not allow_make_bases:
            raise ValueError("recompute_agent_morphisms: Phi_0 missing and allow_make_bases=False")
        agent.Phi_0 = np.eye(Kp, Kq, dtype=agent.mu_q_field.dtype)
    if getattr(agent, "Phi_tilde_0", None) is None:
        if not allow_make_bases:
            raise ValueError("recompute_agent_morphisms: Phi_tilde_0 missing and allow_make_bases=False")
        agent.Phi_tilde_0 = np.eye(Kq, Kp, dtype=agent.mu_q_field.dtype)

    # 3) Seed bases into CacheHub if available (optional)
    if ctx is not None:
        try:
            from transport.transport_cache import set_morphism_bases  # optional API
            set_morphism_bases(ctx, agent, agent.Phi_0, agent.Phi_tilde_0)
        except Exception:
            print("\n FAIL SET MORPH BASES\n")
            pass

    # 4) Pre-warm Φ/Φ̃ in the central cache (no agent writes)
    if ctx is not None:
        try:
            from transport.transport_cache import Phi as tc_Phi
            _ = tc_Phi(ctx, agent, kind="q_to_p")
            _ = tc_Phi(ctx, agent, kind="p_to_q")
        except Exception:
            print("\n FAIL SET Phi\n")
            pass
    else:
        # No ctx: optionally build once to validate shapes (discard results).
        try:
            _Phi, _Phit = compute_direct_morphisms(
                phi=agent.phi,
                phi_model=agent.phi_model,
                generators_q=Gq,
                generators_p=Gp,
                Phi_0=agent.Phi_0,
                Phi_tilde_0=agent.Phi_tilde_0,
                ctx=None,
            )
            _ = _Phi, _Phit  # explicitly unused
        except Exception:
            print("\n FAILcomp-direct-morph\n")
            pass

    # 5) Mark clean flag if present (we didn’t write agent fields)
    if hasattr(agent, "morphisms_dirty"):
        agent.morphisms_dirty = False

def mark_morphism_dirty(agent, *, ctx=None) -> None:
    """
    Mark Φ/Φ̃ as stale for this agent. We no longer mutate or clear
    per-agent bundle_morphism_* fields; morphisms are derived and cached
    centrally via transport_cache.
    """
    # Delegate if the agent provides its own hook
    if hasattr(agent, "mark_morphism_dirty") and callable(agent.mark_morphism_dirty):
        agent.mark_morphism_dirty()
        return

    # Local flag only; no per-agent tensor mutation
    setattr(agent, "morphisms_dirty", True)

    # Optionally notify runtime so ensure_dirty will prewarm caches soon
    if ctx is not None:
        try:
            # Centralized invalidation path (don’t warm here).
            from update_refresh_utils import mark_dirty  # preferred
        except Exception:
            try:
                from runtime_context import mark_dirty  # fallback if exposed there
            except Exception:
                mark_dirty = None
        if mark_dirty is not None:
            # KL refresh not strictly required just for morphisms,
            # but setting kl=True is safe if your pipeline expects it.
            mark_dirty(ctx, agent, kl=False)


def have_parent_morphisms(P) -> bool:
    """
    True iff base morphisms exist with correct rectangular shapes, and generators present:

        Phi_0       : (Kp, Kq)
        Phi_tilde_0 : (Kq, Kp)
    """
    if not hasattr(P, "mu_q_field") or not hasattr(P, "mu_p_field"):
        return False
    Kq = int(getattr(P.mu_q_field, "shape", (0,))[-1])
    Kp = int(getattr(P.mu_p_field, "shape", (0,))[-1])
    if Kq <= 0 or Kp <= 0:
        return False

    ok_gen = hasattr(P, "generators_q") and hasattr(P, "generators_p")

    A = getattr(P, "Phi_0", None)
    B = getattr(P, "Phi_tilde_0", None)
    ok_A = isinstance(A, np.ndarray) and A.ndim == 2 and A.shape == (Kp, Kq)
    ok_B = isinstance(B, np.ndarray) and B.ndim == 2 and B.shape == (Kq, Kp)

    return bool(ok_gen and ok_A and ok_B)


# =============================================================================
#                            Morphism field generators
# =============================================================================

# ===================== Intertwiner Bases Builder (SO(3) reps) =====================
# Drop this into bundle_morphism_utils.py (or a new module) and import where needed.

# --- Keep Φ̃0 coherent when Φ0 is Frobenius-rescaled ---
def _rescale_pair_fro(Phi0: np.ndarray, Phit0: np.ndarray, fro_target: float | None) -> tuple[np.ndarray, np.ndarray, float]:
    """
    If fro_target is given, scale Φ0 -> c·Φ0 and Φ̃0 -> (1/c)·Φ̃0 so that ||Φ0||_F == fro_target.
    Returns (Phi0_scaled, Phit0_scaled, c). No-op if fro_target is None or ||Φ0||_F == 0.
    """
    if fro_target is None:
        return Phi0, Phit0, 1.0
    fn = float(np.linalg.norm(Phi0, ord="fro"))
    if fn <= 0.0:
        return Phi0, Phit0, 1.0
    c = float(fro_target) / fn
    return (c * Phi0, (1.0 / max(c, 1e-30)) * Phit0, c)

# --- Safer choice of Φ̃0 from Φ0 (transpose when square & well-conditioned; else pinv) ---
def _choose_tilde_from_base_safe(Phi0: np.ndarray) -> np.ndarray:
    Kp, Kq = Phi0.shape
    if Kp == Kq:
        # condition check via singular values
        s = np.linalg.svd(Phi0, compute_uv=False)
        if s.size and s.min() > 1e-8 and s.max() < 1e8:
            return Phi0.T
    return np.linalg.pinv(Phi0)

# --------------------------- Public Entry Points ---------------------------------

def build_intertwiner_bases(
    G_q: np.ndarray,            # (3, Kq, Kq) real generators for q-rep
    G_p: np.ndarray,            # (3, Kp, Kp) real generators for p-rep
    *,
    method: str = "casimir",    # "casimir" | "nullspace" | "auto"
    data: Optional[Dict[str, np.ndarray]] = None,   # optional {'mu_q': [...], 'mu_p': [...]}, arrays (T, H, W, K)
    l_tol: float = 1e-6,        # eigenvalue clustering tol for ℓ(ℓ+1)
    nullspace_tol: float = 1e-9,# SVD tolerance for commutator-nullspace
    max_rank: Optional[int] = None, # optional cap on Frobenius rank of Φ0
    return_meta: bool = False
) -> Tuple[np.ndarray, np.ndarray, Optional[dict]]:
    """
    Returns (Phi0, Phit0[, meta]) as rectangular intertwiners between the two real SO(3) reps.
    - Casimir method (default): block-diagonal (by ℓ), partial isometry in multiplicities ⊗ I_{2ℓ+1}.
    - Data-aware (optional): uses μ-projections to pick a canonical orthogonal mixer per ℓ (Procrustes).
    - Nullspace fallback: solves Gp Φ0 = Φ0 Gq exactly; used if no ℓ-overlap or method="nullspace".

    data:
      dict with optional 'mu_q', 'mu_p' arrays (T, H, W, Kq/Kp). We use a few samples to guide multiplicity mixers.
    """
    method = method.lower()
    if method not in ("casimir", "nullspace", "auto"):
        method = "casimir"

    # 1) Try Casimir blocks (sound + fast)
    if method in ("casimir", "auto"):
        Phi0, meta = _build_intertwiner_casimir(G_q, G_p, data=data, l_tol=l_tol, max_rank=max_rank)
        if Phi0 is not None:
            Phit0 = _choose_tilde_from_base(Phi0, mode="pinv")  # pinv is robust for rectangular partial isometries
            return (Phi0.astype(np.float32), Phit0.astype(np.float32), meta if return_meta else None)
        if method == "casimir":
            # explicit request but no compatible ℓ content
            # fall through to nullspace to produce the closest legal intertwiner
            pass

    # 2) Nullspace method (exact intertwiner constraint)
    Phi0, meta = _build_intertwiner_nullspace(G_q, G_p, data=data, tol=nullspace_tol, max_rank=max_rank)
    Phit0 = _choose_tilde_from_base_safe(Phi0)
    return (Phi0.astype(np.float32), Phit0.astype(np.float32), meta if return_meta else None)


def initialize_intertwiners_for_agent(
    ctx,
    agent,
    G_q: np.ndarray, G_p: np.ndarray,
    *,
    method: str = "casimir",
    data: Optional[Dict[str, np.ndarray]] = None,
    allow_overwrite: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (Φ0, Φ̃0) once and freeze them into your persistent cache for this agent.
    Assumes you added the 'allow_overwrite' kwarg to set_morphism_bases(...).
    """
    Phi0, Phit0, _ = build_intertwiner_bases(G_q, G_p, method=method, data=data, return_meta=True)
    # Persist exactly once (unless you explicitly allow overwrite)
    from transport.transport_cache import set_morphism_bases  # adjust import if needed
    set_morphism_bases(ctx, agent, Phi0, Phit0, allow_overwrite=bool(allow_overwrite))
    return Phi0, Phit0


# --------------------------- Casimir (ℓ-block) path -------------------------------

def _build_intertwiner_casimir(G_q, G_p, *, data=None, l_tol=1e-6, max_rank=None):
    Cq = _casimir_from_generators(G_q); λq, Wq = np.linalg.eigh(Cq)
    Cp = _casimir_from_generators(G_p); λp, Wp = np.linalg.eigh(Cp)

    # NEW: use rel-tol degeneracy clustering (scale-invariant)
    blocks_q = _cluster_into_l_blocks(λq, Wq, rel_tol=1e-4)
    blocks_p = _cluster_into_l_blocks(λp, Wp, rel_tol=1e-4)

    # Match blocks by d (2l+1); don’t rely on absolute λ
    by_dim_q = {}
    for b in blocks_q.values():
        by_dim_q.setdefault(b["d"], []).append(b["W"])
    by_dim_p = {}
    for b in blocks_p.values():
        by_dim_p.setdefault(b["d"], []).append(b["W"])

    common_ds = sorted(set(by_dim_q.keys()).intersection(by_dim_p.keys()))
    if not common_ds:
        return (None, {"reason": "no_l_overlap"})  # will be handled by caller

    Kq = G_q.shape[-1]; Kp = G_p.shape[-1]
    Phi0 = np.zeros((Kp, Kq), dtype=np.float64)
    meta = {"used_dims": [], "rank_by_dim": {}, "path": "casimir"}

    # Optional data-aware mixing on multiplicities can be kept as before.
    for d in common_ds:
        Q_list = by_dim_q[d]   # each (Kq, m_q*d) with orthonormal cols
        P_list = by_dim_p[d]   # each (Kp, m_p*d)
        # concatenate copies -> treat multiplicity collectively
        Wq_d = np.concatenate(Q_list, axis=1)
        Wp_d = np.concatenate(P_list, axis=1)
        # multiplicities
        m_q = Wq_d.shape[1] // d
        m_p = Wp_d.shape[1] // d
        m   = min(m_q, m_p)
        if m == 0:
            continue
        # split into per-copy blocks
        Q_blocks = [Wq_d[:, i*d:(i+1)*d] for i in range(m_q)]
        P_blocks = [Wp_d[:, i*d:(i+1)*d] for i in range(m_p)]
        # identity mixer on multiplicities (orthogonal per copy)
        for i in range(m):
            Phi0 += P_blocks[i] @ Q_blocks[i].T
        meta["used_dims"].append(d)
        meta["rank_by_dim"][d] = m * d

    if max_rank is not None:
        Phi0 = _best_rank_approx(Phi0, rank=max_rank)

    return (Phi0, meta)


# ------------------------------ Nullspace path -----------------------------------

def _build_intertwiner_nullspace(
    G_q: np.ndarray, G_p: np.ndarray,
    *,
    tol: float = 1e-9,
    data: Optional[Dict[str, np.ndarray]] = None,
    max_rank: Optional[int] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Solve (I ⊗ Gp_a - Gq_a^T ⊗ I) vec(Phi0) = 0 for a in {x,y,z}.
    Then select a canonical combination in the nullspace (optionally data-aware).
    """
    Kq, Kp = G_q.shape[-1], G_p.shape[-1]
    Iq = np.eye(Kq); Ip = np.eye(Kp)
    rows = []
    for a in range(3):
        A = np.kron(Iq, G_p[a]) - np.kron(G_q[a].T, Ip)
        rows.append(A)
    M = np.concatenate(rows, axis=0)

    # Nullspace via SVD
    U, S, Vh = np.linalg.svd(M, full_matrices=False)
    mask = (S <= tol)
    if not np.any(mask):
        # numerically tiny but nonzero; take the smallest singular vector
        basis = Vh[-1:, :]
    else:
        basis = Vh[mask, :]        # (m, Kp*Kq)
    # Orthonormalize nullspace basis vectors (Frobenius) via QR
    Q, _ = np.linalg.qr(basis.T)   # (Kp*Kq, m) -> Q has orthonormal columns
    B = Q                          # (Kp*Kq, m_eff)

    # Default: pick the first basis vector -> simplest nonzero intertwiner
    coeff = np.zeros((B.shape[1],), dtype=np.float64)
    coeff[0] = 1.0

    # Data-aware LS in span(B): minimize Σ ||Φ0 μ_q - μ_p||_2^2
    if data is not None:
        mu_q = np.asarray(data.get("mu_q", None))
        mu_p = np.asarray(data.get("mu_p", None))
        if mu_q is not None and mu_q.size and mu_p is not None and mu_p.size:
            Qs = _thin_samples(mu_q.reshape(-1, mu_q.shape[-1]), max_samples=4096)  # (N, Kq)
            Ps = _thin_samples(mu_p.reshape(-1, mu_p.shape[-1]), max_samples=4096)  # (N, Kp)
            # Build design matrix A * coeff ≈ b, with A = [ (Bi mat) @ Qs^T ] stacked
            # We solve per-sample in Kp, so do least squares per output dim jointly:
            # vec(Φ0) = B @ coeff  =>  Φ0 = reshape(B@coeff, (Kp, Kq))
            # We choose coeff that minimizes ||Φ0 Qs^T - Ps^T||_F^2
            # This is linear in coeff:
            # Let Zi = reshape(B[:,i], Kp,Kq) @ Qs^T  -> (Kp,N)
            # Stack Zi along i to get (Kp*N, m), target vec(Ps^T) = vec(Ps.T)
            Z_blocks = []
            for i in range(B.shape[1]):
                Zi = (B[:, i].reshape(Kp, Kq) @ Qs.T)  # (Kp, N)
                Z_blocks.append(Zi.reshape(-1, 1))     # (Kp*N, 1)
            A_ls = np.concatenate(Z_blocks, axis=1)    # (Kp*N, m)
            b_ls = Ps.T.reshape(-1)                    # (Kp*N,)
            # regularized least squares to avoid degeneracy
            reg = 1e-8
            coeff, *_ = np.linalg.lstsq(A_ls.T @ A_ls + reg*np.eye(A_ls.shape[1]), A_ls.T @ b_ls, rcond=None)

    vecPhi = (B @ coeff).reshape(Kp, Kq)
    Phi0 = vecPhi
    if max_rank is not None:
        Phi0 = _best_rank_approx(Phi0, rank=max_rank)

    meta = {"nullspace_dim": B.shape[1]}
    return (Phi0, meta)


# ------------------------------ Internal Helpers ---------------------------------

def _casimir_from_generators(G: np.ndarray) -> np.ndarray:
    """Casimir C = Gx^2 + Gy^2 + Gz^2 for a real rep."""
    C = np.zeros((G.shape[-1], G.shape[-1]), dtype=np.float64)
    for a in range(3):
        C += G[a] @ G[a]
    return 0.5 * (C + C.T)  # enforce symmetry numerically


def _cluster_into_l_blocks(λ: np.ndarray, W: np.ndarray, rel_tol: float = 1e-4):
    """
    Scale-invariant clustering of Casimir eigenpairs by degeneracy.
    Groups consecutive eigenvalues whose relative gap ≤ rel_tol.
    For each cluster of size g, pick d = largest odd divisor of g;
    set multiplicity m = g // d and split the cluster into m blocks of size d.
    Returns dict: l -> {"W": (K, m*d), "d": d, "m": m}
    """
    K = W.shape[0]
    # sort by eigenvalue
    idx = np.argsort(λ)
    λs  = λ[idx]
    Ws  = W[:, idx]

    # cluster by relative tolerance
    clusters = []
    start = 0
    for i in range(1, len(λs) + 1):
        if i == len(λs):
            clusters.append((start, i))
            break
        # relative gap
        denom = max(abs(λs[i-1]), abs(λs[i]), 1.0)
        if abs(λs[i] - λs[i-1]) > rel_tol * denom:
            clusters.append((start, i))
            start = i

    blocks = {}
    # helper: largest odd divisor
    def largest_odd_divisor(n: int) -> int:
        while n % 2 == 0 and n > 0:
            n //= 2
        return max(1, n)

    l_counter = 0
    for (lo, hi) in clusters:
        g = hi - lo                     # cluster size
        if g <= 0:
            continue
        d = largest_odd_divisor(g)      # irrep dim = 2l+1 (odd)
        m = g // d
        if d <= 0 or m <= 0:
            continue
        # take this cluster’s eigenvectors
        Wc = Ws[:, lo:hi]               # (K, g)
        # split into m blocks of size d
        keep = Wc[:, : m*d]             # truncate if any tiny overhang
        l_val = (d - 1) // 2
        blocks[l_counter] = {"W": keep, "d": d, "m": m, "ell": l_val}
        l_counter += 1

    return blocks


def _thin_samples(X: np.ndarray, max_samples: int = 4096) -> np.ndarray:
    """Uniformly sub-sample rows to at most max_samples."""
    N = X.shape[0]
    if N <= max_samples:
        return X
    idx = np.linspace(0, N-1, num=max_samples, dtype=int)
    return X[idx]


def _procrustes_multiplicity(
    Q_blocks: List[np.ndarray],  # list of (Kq, d)
    P_blocks: List[np.ndarray],  # list of (Kp, d)
    Qs: np.ndarray,              # (N, Kq)
    Ps: np.ndarray,              # (N, Kp)
    *,
    d: int,
    m: int
) -> np.ndarray:
    """
    Choose U ∈ O(m) to align multiplicity spaces using empirical μ projections.
    We compute per-copy coefficients by projecting samples onto each block, then solve
    a Procrustes problem on the m-dimensional coefficient clouds (shared d collapsed by norm).
    """
    # For each block, build a projection operator Pi = Bi (d×K) @, but Bi are (K, d) with orthonormal columns.
    # Coeffs for sample s on block i: c_i(s) = ||B_i^T x||_2 (collapse d to scalar magnitude)
    def coeffs(B_list: List[np.ndarray], X: np.ndarray) -> np.ndarray:
        # returns (N, len(B_list)) of magnitudes
        N = X.shape[0]
        out = np.zeros((N, len(B_list)), dtype=np.float64)
        for i, Bi in enumerate(B_list):
            # (K, d)^T @ (N, K)^T = (d, N) -> take column norms
            Zi = Bi.T @ X.T    # (d, N)
            out[:, i] = np.linalg.norm(Zi, axis=0)  # (N,)
        return out

    Cq = coeffs(Q_blocks, Qs)[:, :m]  # (N, m)
    Cp = coeffs(P_blocks, Ps)[:, :m]  # (N, m)

    # Center
    Cq -= Cq.mean(axis=0, keepdims=True)
    Cp -= Cp.mean(axis=0, keepdims=True)

    # Cross-covariance
    H = Cp.T @ Cq  # (m, m)
    Uh, _, Vh = np.linalg.svd(H, full_matrices=False)
    U = Uh @ Vh    # optimal orthogonal (Procrustes)
    return U


def _choose_tilde_from_base(Phi0: np.ndarray, mode: str = "pinv") -> np.ndarray:
    """Pick Φ̃0 given Φ0. For partial isometries, pinv and transpose are both reasonable; pinv is robust."""
    if mode == "transpose":
        return Phi0.T
    # default: pseudoinverse (handles rectangular / rank-deficient cleanly)
    return np.linalg.pinv(Phi0)


def _best_rank_approx(M: np.ndarray, rank: int) -> np.ndarray:
    """Best Frobenius-norm rank-k approximation via SVD."""
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    k = min(rank, len(S))
    return (U[:, :k] * S[:k]) @ Vt[:k, :]


# =========================== End Intertwiner Builder ==============================













def generate_smooth_morphism_fieldID(  # kept for backward compat
    H: int,
    W: int,
    K_src: int,
    K_tgt: int,
    sigma: float = getattr(config, "gaussian_blur_range_morphism", 1.5),
    low_rank=None,
    seed=None,
    normalize=False,
    mask=None,
) -> np.ndarray:
    """
    Identity-like morphism field on shared rank R = min(K_src, K_tgt).
    """
    R = min(K_src, K_tgt)
    field = np.zeros((H, W, K_tgt, K_src), dtype=np.float32)
    ones = np.ones((H, W), dtype=np.float32)
    coeff = gaussian_filter(ones, sigma=1.5, mode="wrap")
    for i in range(R):
        field[..., i, i] = coeff
    return field


def generate_smooth_morphism_field(
    H: int,
    W: int,
    K_src: int,
    K_tgt: int,
    sigma: float = getattr(config, "gaussian_blur_range_morphism", 1.5),
    low_rank: Optional[int] = None,
    seed: Optional[int] = None,
    normalize: bool = False,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Generate a smooth spatial field of K_tgt × K_src morphisms,
    optionally low-rank and softly blended to identity outside support.
    """
    rng = np.random.default_rng(seed)
    rank = low_rank if low_rank is not None else min(K_src, K_tgt)

    field = np.zeros((H, W, K_tgt, K_src), dtype=np.float32)
    for r in range(rank):
        basis = rng.standard_normal((K_tgt, K_src)).astype(np.float32)
        coeff = rng.standard_normal((H, W)).astype(np.float32)
        coeff = gaussian_filter(coeff, sigma=sigma, mode="wrap")
        field += coeff[..., None, None] * basis[None, None, :, :]

    if normalize:
        eps = 1e-8
        norm = np.linalg.norm(field, axis=(-2, -1), keepdims=True)
        field = field / np.where(norm > eps, norm, eps)

    if mask is not None:
        soft = gaussian_filter(mask.astype(np.float32), sigma=1.0)
        soft = np.clip(soft, 0.0, 1.0)[..., None, None]
        I = np.eye(K_tgt, K_src, dtype=np.float32)[None, None, :, :]
        field = soft * field + (1.0 - soft) * I

    return field


def clip_operator_norm(field: np.ndarray, max_norm: float = 1.0, eps: float = 1e-8) -> np.ndarray:
    """
    Enforce operator norm ≤ max_norm for each Φ(x) in (H,W,Kp,Kq).
    """
    H, W, Kp, Kq = field.shape
    out = np.empty_like(field)
    for i in range(H):
        for j in range(W):
            Phi = field[i, j]
            u, s, vh = np.linalg.svd(Phi, full_matrices=False)
            s_clipped = np.clip(s, 0, max_norm)
            out[i, j] = (u @ np.diag(s_clipped) @ vh).astype(field.dtype)
    return out


def initialize_morphism_pair(
    domain_size: Tuple[int, int],
    K_q: int,
    K_p: int,
    rng: np.random.Generator,
    morphism_type: str = "identity",
    mask: Optional[np.ndarray] = None,
    config_tag: str = "",
    phi_q_field: Optional[np.ndarray] = None,
    phi_p_field: Optional[np.ndarray] = None,
    G_q: Optional[np.ndarray] = None,
    G_p: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize bundle morphisms Φ (q→p) and Φ̃ (p→q).

    If morphism_type == "gauge_covariant", applies:
        Φ(x) = exp(φ_p(x)) · Φ₀(x) · exp(−φ_q(x))
    """
    H, W = domain_size

    if morphism_type == "identity":
        if K_q != K_p:
            raise ValueError(f"Cannot use identity morphism with mismatched K_q={K_q} and K_p={K_p}")
        eye = np.eye(K_q, dtype=np.float32)
        phi_q_to_p = np.broadcast_to(eye, (H, W, K_q, K_q)).copy()
        phi_p_to_q = np.broadcast_to(eye, (H, W, K_q, K_q)).copy()
        if mask is not None:
            soft = gaussian_filter(mask.astype(np.float32), sigma=1.0)
            soft = np.clip(soft, 0.0, 1.0)[..., None, None]
            phi_q_to_p = soft * phi_q_to_p + (1.0 - soft) * phi_q_to_p
            phi_p_to_q = soft * phi_p_to_q + (1.0 - soft) * phi_p_to_q
        return phi_q_to_p, phi_p_to_q

    if morphism_type == "gauge_covariant":
        assert phi_q_field is not None and phi_p_field is not None, "φ_q and φ_p fields required"
        assert G_q is not None and G_p is not None, "Generators G_q and G_p required"

        sigma = getattr(config, f"morphism_sigma{config_tag}", 1.5)
        rank  = getattr(config, f"morphism_rank{config_tag}", min(K_q, K_p))
        seed  = rng.integers(0, 1e9)

        Phi_0_q_to_p = generate_smooth_morphism_field(
            H, W, K_src=K_q, K_tgt=K_p,
            sigma=sigma, low_rank=rank,
            normalize=False, mask=mask, seed=seed
        )
        Phi_0_p_to_q = generate_smooth_morphism_field(
            H, W, K_src=K_p, K_tgt=K_q,
            sigma=sigma, low_rank=rank,
            normalize=False, mask=mask, seed=seed + 1
        )

        phi_q_to_p = gauge_covariant_transform_phi(
            phi_p_field, phi_q_field, Phi_0_q_to_p, G_q=G_q, G_p=G_p
        )
        phi_p_to_q = gauge_covariant_transform_phi(
            phi_q_field, phi_p_field, Phi_0_p_to_q, G_q=G_p, G_p=G_q
        )

        if mask is not None:
            soft = gaussian_filter(mask.astype(np.float32), sigma=1.0)
            soft = np.clip(soft, 0.0, 1.0)[..., None, None]
            I_qp = np.eye(K_p, K_q, dtype=np.float32)[None, None, :, :]
            I_pq = np.eye(K_q, K_p, dtype=np.float32)[None, None, :, :]
            phi_q_to_p = soft * phi_q_to_p + (1.0 - soft) * I_qp
            phi_p_to_q = soft * phi_p_to_q + (1.0 - soft) * I_pq

        return phi_q_to_p, phi_p_to_q

    # Generic smooth/low-rank morphism
    sigma     = getattr(config, f"morphism_sigma{config_tag}", 1.5)
    rank      = getattr(config, f"morphism_rank{config_tag}", min(K_q, K_p))
    normalize = getattr(config, f"morphism_normalize{config_tag}", True)
    seed      = rng.integers(0, 1e9)

    phi_q_to_p = generate_smooth_morphism_field(
        H, W, K_src=K_q, K_tgt=K_p,
        sigma=sigma, low_rank=rank,
        normalize=False, mask=mask, seed=seed
    )
    phi_p_to_q = np.swapaxes(phi_q_to_p, -1, -2)

    if normalize:
        phi_q_to_p = clip_operator_norm(phi_q_to_p, max_norm=1.0)
        phi_p_to_q = clip_operator_norm(phi_p_to_q, max_norm=1.0)

    return phi_q_to_p, phi_p_to_q


def generate_morphism_pair(
    H: int,
    W: int,
    K_q: int,
    K_p: int,
    rank: Optional[int] = None,
    sigma: float = 1.5,
    seed: Optional[int] = None,
    mask: Optional[np.ndarray] = None,
    s_floor: float = 0.05,
    s_scale: float = 0.25,
    dtype=np.float32,
):
    """
    Build a paired rectangular morphism field (Φ0, Φ̃0) with shared low-rank frames:

        Φ0(x)   = U S(x) Vᵀ   ∈ ℝ^{K_p × K_q}
        Φ̃0(x)  = V S(x) Uᵀ   ∈ ℝ^{K_q × K_p}

    U ∈ ℝ^{K_p×r}, V ∈ ℝ^{K_q×r} orthonormal; S(x) diagonal, positive, smooth.
    """
    rng = np.random.default_rng(seed)
    r = rank if rank is not None else min(K_q, K_p)

    U0, _ = np.linalg.qr(rng.standard_normal((K_p, r)))
    V0, _ = np.linalg.qr(rng.standard_normal((K_q, r)))

    S_fields = []
    for _ in range(r):
        f = rng.standard_normal((H, W)).astype(dtype)
        f = gaussian_filter(f, sigma=sigma, mode="wrap")
        f = np.abs(f) * s_scale + s_floor  # positive, bounded away from 0
        if mask is not None:
            f = f * mask
        S_fields.append(f)
    S_stack = np.stack(S_fields, axis=-1)  # (H, W, r)

    S_diag = np.zeros((H, W, r, r), dtype=dtype)
    idx = np.arange(r)
    S_diag[..., idx, idx] = S_stack

    Phi_0 = np.einsum("pr,hwrs,sq->hwpq", U0.astype(dtype), S_diag, V0.T.astype(dtype))
    Phi_tilde_0 = np.einsum("qr,hwrs,sp->hwqp", V0.astype(dtype), S_diag, U0.T.astype(dtype))
    return Phi_0, Phi_tilde_0
