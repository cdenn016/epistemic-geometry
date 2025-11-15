
from __future__ import annotations
import numpy as np
from typing import Iterable, List, Tuple, Optional, Dict

import numpy as np
from scipy.linalg import expm

#=======================================================
#
#           Generators
#
#=========================================================


_generator_cache = {}  # Already declared

def get_generators(group_name, K, return_meta=False):
    group_name = group_name.lower()
    if group_name not in ["sl2r", "so3"]:
        raise ValueError(f"Unsupported group: {group_name}")

    if isinstance(K, int):
        key = (group_name, K)
    elif isinstance(K, list):
        key = (group_name, tuple(K))
    else:
        raise TypeError(f"[get_generators] Invalid type for K: {type(K)}")

    if key in _generator_cache:
        G, meta = _generator_cache[key]
    else:
        if isinstance(K, int):
            if group_name == "sl2r":
                G_dict = sl2r_irrep(K)
            else:  # SO(3)
                G_dict = so3_irrep(K)

            for name in ["E", "F", "H"]:
                Gmat = G_dict[name]
                assert Gmat.ndim == 2 and Gmat.shape[0] == Gmat.shape[1]

            G = np.stack([G_dict["E"], G_dict["F"], G_dict["H"]], axis=0)
            meta = {"blocks": [K]}
        else:
            G, meta = _get_reducible_sl2r(K)  # You can also later define `_get_reducible_so3`

        _generator_cache[key] = (G, meta)

    return (G, meta) if return_meta else G



def so3_irrep(K: int) -> dict:
    if K % 2 == 0:
        raise ValueError("K must be odd for SO(3) irreps (K = 2ℓ + 1).")
    ℓ = (K - 1) // 2

    # 1) Complex |m> basis, m = -ℓ..ℓ
    Jp = np.zeros((K, K), dtype=np.complex64)
    Jm = np.zeros((K, K), dtype=np.complex64)
    Jz = np.zeros((K, K), dtype=np.complex64)
    for m in range(-ℓ, ℓ + 1):
        i = m + ℓ
        Jz[i, i] = m
        if m < ℓ:
            a = np.sqrt((ℓ - m) * (ℓ + m + 1))
            Jp[i, i + 1] = a
            Jm[i + 1, i] = a
    Jx = (Jp + Jm) / 2.0
    Jy = (Jp - Jm) / (2.0j)

    # 2) Complex->real spherical-harmonic basis transform S
    S = np.zeros((K, K), dtype=np.complex64)
    S[0, ℓ] = 1.0  # m=0 stays real
    r = 1
    for m in range(1, ℓ + 1):
        phase = (-1) ** m
        # cosine row (real)
        S[r,   ℓ + m] = 1 / np.sqrt(2)
        S[r,   ℓ - m] = phase / np.sqrt(2)
        # sine row (real)
        S[r+1, ℓ + m] = -1j / np.sqrt(2)
        S[r+1, ℓ - m] =  1j * phase / np.sqrt(2)
        r += 2
    Sinv = S.conj().T  # unitary

    # 3) Real skew-symmetric algebra generators: G = S (iJ) S^{-1}
    def real_skew(iJ):
        G = (S @ iJ @ Sinv).real
        return 0.5 * (G - G.T)  # enforce exact skew-symmetry

    Gx = real_skew(1j * Jx)
    Gy = real_skew(1j * Jy)
    Gz = real_skew(1j * Jz)

    # Return in a fixed order; pick any consistent (E,F,H) mapping
    return {"E": Gx, "F": Gy, "H": Gz}




def sl2r_irrep(K: int) -> dict:
    """
    Return the real irreducible SL(2,R) generators E, F, H for dimension K = 2ℓ + 1.
    """
    if K % 2 == 0:
        raise ValueError("K must be odd for SL(2,R) irreps (K = 2ℓ + 1).")
    ℓ = (K - 1) // 2
    E = np.zeros((K, K), dtype=float)
    F = np.zeros((K, K), dtype=float)
    H = np.zeros((K, K), dtype=float)

    for m in range(-ℓ, ℓ + 1):
        i = m + ℓ
        H[i, i] = m
        if i < K - 1:
            a = np.sqrt((ℓ - m) * (ℓ + m + 1))
            E[i, i + 1] = a
            E[i + 1, i] = a
            F[i, i + 1] = -a
            F[i + 1, i] = a
    return {"E": E, "F": F, "H": H}



#==========================================
#
#    Build Reducible Representations
#
#==========================================

# get_generators.py
# Build real SO(3) generators for arbitrary reducible representations:
#   G = direct sum over ℓ of multiplicity m_ℓ copies of the (2ℓ+1)-dim real irrep.
# Optional: apply a global orthogonal similarity transform to mix the blocks.


Array = np.ndarray

# ----------------------- Public API -----------------------

def make_reducible_generators(
    spec: Iterable[Tuple[int, int]],
    *,
    mix: bool = False,
    rng: np.random.Generator | None = None,
    dtype=np.float32,
) -> Array:
    """
    Build generators J = (Jx, Jy, Jz) for a real reducible SO(3) rep.

    Args
    ----
    spec : iterable of (ell, multiplicity) tuples, ell >= 0, multiplicity >= 1
           Example: [(1,2), (2,1)] -> two copies of ℓ=1 (dim 3 each) ⊕ one copy of ℓ=2 (dim 5)
    mix  : if True, apply a random global orthogonal Q so G_a := Q G_a Q^T (hides block structure)
    rng  : optional numpy.random.Generator for deterministic mixing
    dtype: output dtype

    Returns
    -------
    G : np.ndarray with shape (3, K, K) in 'x,y,z' order.
    """
    blocks: List[Array] = []
    total_dim = 0
    for ell, mult in spec:
        if ell < 0 or mult <= 0:
            raise ValueError(f"Invalid (ell,mult)=({ell},{mult}).")
        d = 2 * ell + 1
        Gi = _real_irrep_generators(ell)  # (3, d, d), float64
        for _ in range(mult):
            blocks.append(Gi)
            total_dim += d

    if total_dim == 0:
        raise ValueError("Empty spec: total dimension is zero.")

    # Block-diagonal stack
    G = _block_diag_generators(blocks)    # (3, K, K) float64

    # Optional global orthogonal mixing
    if mix:
        if rng is None:
            rng = np.random.default_rng()
        Q = _random_orthogonal(total_dim, rng=rng)     # (K,K) float64
        for a in range(3):
            G[a] = Q @ G[a] @ Q.T

    # Cast
    G = G.astype(dtype, copy=False)

    # Validate once (tolerant)
    ok, resid = check_so3_relations(G, rtol=5e-6, atol=5e-7, return_resid=True)
    if not ok:
        raise RuntimeError(f"SO(3) commutator check failed, residual {resid:.3e}")

    return G


def check_so3_relations(
    G: Array, *, rtol: float = 1e-7, atol: float = 1e-8, return_resid: bool = False
) -> bool | Tuple[bool, float]:
    """
    Verify [Jx, Jy] = Jz, [Jy, Jz] = Jx, [Jz, Jx] = Jy in Frobenius norm.

    Returns True if all residuals <= atol + rtol*||target||; otherwise False.
    If return_resid=True, also returns combined residual sqrt(sum ||err||_F^2).
    """
    Gx, Gy, Gz = G[0], G[1], G[2]
    errs = []
    errs.append(_fro_norm(Gx @ Gy - Gy @ Gx - Gz))
    errs.append(_fro_norm(Gy @ Gz - Gz @ Gy - Gx))
    errs.append(_fro_norm(Gz @ Gx - Gx @ Gz - Gy))
    # Targets have norms comparable to ||G||
    target_norm = max(_fro_norm(Gx), _fro_norm(Gy), _fro_norm(Gz), 1.0)
    thresh = atol + rtol * target_norm
    ok = all(e <= thresh for e in errs)
    if return_resid:
        res = float(np.sqrt(sum(e*e for e in errs)))
        return ok, res
    return ok


# ----------------------- Irrep construction (real basis) -----------------------

def _real_irrep_generators(ell: int) -> Array:
    """
    Construct the real (tesseral) basis generators (Jx, Jy, Jz) for integer ℓ.
    Start from the complex spherical basis |ℓ,m>, m=-ℓ..ℓ:
      J_+|ℓ,m> = √(ℓ(ℓ+1)-m(m+1)) |ℓ,m+1>
      J_-|ℓ,m> = √(ℓ(ℓ+1)-m(m-1)) |ℓ,m-1>
      Jz|ℓ,m> = m |ℓ,m>
      Jx = (J_+ + J_-)/2, Jy = (J_+ - J_-)/(2i)
    Then convert to the real (tesseral) basis:
      m=0 stays; for m>0:
        c_m = (|m> + (-1)^m |-m>) / √2
        s_m = (|m> - (-1)^m |-m>) / (√2 i)
    Finally reorder basis as: [m=0, c_1, s_1, c_2, s_2, ..., c_ℓ, s_ℓ].
    """
    d = 2*ell + 1
    # Complex spherical basis operators
    Jp = np.zeros((d, d), dtype=np.complex128)
    Jm = np.zeros((d, d), dtype=np.complex128)
    Jz = np.zeros((d, d), dtype=np.complex128)

    # m index mapping: m=-ell..ell -> idx = m + ell
    def idx(m): return m + ell

    for m in range(-ell, ell+1):
        Jz[idx(m), idx(m)] = m
        if m < ell:
            c = np.sqrt(ell*(ell+1) - m*(m+1))
            Jp[idx(m+1), idx(m)] = c
        if m > -ell:
            c = np.sqrt(ell*(ell+1) - m*(m-1))
            Jm[idx(m-1), idx(m)] = c

    Jx = 0.5 * (Jp + Jm)
    Jy = (Jp - Jm) / (2.0j)

    # Complex -> real (tesseral) basis transform U (unitary)
    U = _complex_to_tesseral_unitary(ell)  # (d,d), unitary
    Jx_r = (U.conj().T @ Jx @ U).real
    Jy_r = (U.conj().T @ Jy @ U).real
    Jz_r = (U.conj().T @ Jz @ U).real

    return np.stack([_sym(Jx_r), _sym(Jy_r), _sym(Jz_r)], axis=0)  # (3,d,d), float64


def _complex_to_tesseral_unitary(ell: int) -> Array:
    """
    Build unitary U that maps spherical |ℓ,m> to real tesseral basis:
      [ m=0, c_1, s_1, c_2, s_2, ..., c_ℓ, s_ℓ ]
    with
      c_m = (|m> + (-1)^m |-m>) / √2
      s_m = (|m> - (-1)^m |-m>) / (√2 i)
    """
    d = 2*ell + 1
    U = np.zeros((d, d), dtype=np.complex128)
    # Column order for tesseral basis:
    cols: List[Tuple[str,int]] = [("z", 0)]
    for m in range(1, ell+1):
        cols += [("c", m), ("s", m)]

    def idx(m): return m + ell
    col = 0
    # m=0 stays the same
    U[idx(0), col] = 1.0 + 0j
    col += 1
    for m in range(1, ell+1):
        # c_m
        phase = (+1.0 if (m % 2 == 0) else -1.0)  # (-1)^m
        U[idx(m),   col] = 1.0/np.sqrt(2.0)
        U[idx(-m),  col] = phase/np.sqrt(2.0)
        col += 1
        # s_m
        U[idx(m),   col] =  1.0/(np.sqrt(2.0)*1j)
        U[idx(-m),  col] = -phase/(np.sqrt(2.0)*1j)
        col += 1
    # Verify unitary numerically
    # (optional) could re-orthonormalize with QR if desired
    return U


# ----------------------- Reducible assembly helpers -----------------------

def _block_diag_generators(blocks: List[Array]) -> Array:
    """
    Given a list of (3, d_i, d_i) irrep generator triplets, place them on the block diagonal.
    Returns (3, sum d_i, sum d_i).
    """
    if not blocks:
        raise ValueError("No blocks to assemble.")
    sizes = [b.shape[1] for b in blocks]
    K = int(sum(sizes))
    G = np.zeros((3, K, K), dtype=np.float64)
    r0 = 0
    for b in blocks:
        d = b.shape[1]
        r1 = r0 + d
        for a in range(3):
            G[a, r0:r1, r0:r1] = b[a]
        r0 = r1
    return G


def _random_orthogonal(n: int, rng: np.random.Generator) -> Array:
    """Random Haar orthogonal via QR of Gaussian matrix (sign-fix)."""
    A = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(A)
    # Fix signs to ensure uniform Haar
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    Q = Q * s
    return Q


# ----------------------- Small utilities -----------------------

def _fro_norm(A: Array) -> float:
    return float(np.linalg.norm(A, ord="fro"))

def _sym(A: Array) -> Array:
    """Force numerically real-symmetric where expected (generators are real but not symmetric in general).
    We keep it as-is; this helper is here if you want a symmetric check in the pipeline."""
    return (A + A) * 0.5  # no-op; kept for clarity


# ----------------------- Convenience: auto spec from total K -----------------------
# (Optional) If you want to propose a decomposition given K, uncomment and adapt:
#
# def propose_spec_for_dimension(K: int) -> List[Tuple[int,int]]:
#     """
#     Heuristic splitter: fill K greedily with largest 2ℓ+1 not exceeding remaining.
#     Always returns some decomposition; not unique nor canonical.
#     """
#     if K <= 0:
#         raise ValueError("K must be positive.")
#     spec = []
#     rem = K
#     ell = int((K-1)//2)  # start near largest odd <= K
#     while rem > 0:
#         # find largest d=2ℓ+1 <= rem
#         while (2*ell + 1) > rem:
#             ell -= 1
#             if ell < 0:
#                 # fallback: fill with ℓ=0
#                 ell = 0
#                 break
#         d = 2*ell + 1
#         spec.append((ell, 1))
#         rem -= d
#     return spec



def block_diag_generators(blocks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """
    Combine a list of irreducible generator sets {E,F,H} into a block-diagonal reducible set.

    Args:
        blocks: list of dicts, each with {"E", "F", "H"} generators of shape (Ki, Ki)

    Returns:
        dict with:
            "E" : (K_total, K_total) ndarray
            "F" : (K_total, K_total) ndarray
            "H" : (K_total, K_total) ndarray
    """
    def block_diag(gen_list):
        sizes = [g.shape[0] for g in gen_list]
        total = sum(sizes)
        out = np.zeros((total, total), dtype=np.float64)
        offset = 0
        for g in gen_list:
            k = g.shape[0]
            out[offset:offset+k, offset:offset+k] = g
            offset += k
        return out

    return {
        "E": block_diag([b["E"] for b in blocks]),
        "F": block_diag([b["F"] for b in blocks]),
        "H": block_diag([b["H"] for b in blocks])
    }



def _get_reducible_sl2r(K_list):
    """
    Construct reducible SL(2,R) representation from list of irreducible blocks.

    Args:
        K_list : list[int] — sizes of each irreducible block

    Returns:
        G      : ndarray (3, K_total, K_total) — stacked [E, F, H]
        meta   : dict {"blocks": K_list}
    """
    irreps = [sl2r_irrep(K) for K in K_list]  # each returns {"E", "F", "H"}
    G_dict = block_diag_generators(irreps)    # builds block-diagonal E/F/H
    G = np.stack([G_dict["E"], G_dict["F"], G_dict["H"]], axis=0)
    return G, {"blocks": K_list}
