# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 17:54:26 2025

@author: chris and christine
"""
from scipy.ndimage import gaussian_filter
import core.config as config
import numpy as np


from core.agent_schema import Agent
from core.get_generators import get_generators
from core.numerical_utils import soft_clip_phi_norm
from numpy.random import default_rng
from transport.bundle_morphism_utils import generate_morphism_pair, gauge_covariant_transform_phi

from core.numerical_utils import sanitize_sigma  


def initialize_agents(
    seed: int,
    domain_size: tuple[int, ...],
    N: int,
    K_q: int,
    K_p: int,
    lie_algebra_dim: int,
    visualize: bool = False,
    fixed_location: bool = True,
    params=None,
) -> list:
    

    rng = default_rng(seed)
    dtype = getattr(config, "dtype", np.float32)
    cfg = {k: getattr(config, k) for k in dir(config) if not k.startswith("__")}

    phi_init         = cfg["phi_init"]
    phi_model_offset = cfg.get("phi_model_offset", 0.1)
    max_neighbors    = cfg.get("max_neighbors", 8)
    overlap_eps      = cfg.get("overlap_eps", 1e-3)
    diagonal_sigma   = cfg.get("diagonal_sigma", True)

    group_name = cfg.get("group_name", "so3")
    G_q = get_generators(group_name, K_q)
    G_p = get_generators(group_name, K_p)

    fixed_center = tuple(s // 2 for s in domain_size) if fixed_location else None
    agents = []

    use_global_belief = cfg.get("use_global_belief_template", False)
    use_global_model  = cfg.get("use_global_model_template", False)

    if use_global_belief:
        global_mu_q, global_sigma_q = generate_mu_sigma_fields(
            domain_size=domain_size, rng=rng, K=K_q,
            mu_range=config.q_mu_range, sigma_range=config.q_sigma_range,
            mask=None, sigma_outside_val=cfg.get("sigma_outside_val", 1.0),
            dtype=dtype, smooth_sigma=cfg.get("init_smooth_sigma", 2.0),
            diagonal_sigma=diagonal_sigma,
        )

    if use_global_model:
        global_mu_p, global_sigma_p = generate_mu_sigma_fields(
            domain_size=domain_size, rng=rng, K=K_p,
            mu_range=config.p_mu_range, sigma_range=config.p_sigma_range,
            mask=None, sigma_outside_val=cfg.get("sigma_outside_val", 1.0),
            dtype=dtype, smooth_sigma=cfg.get("init_smooth_sigma", 2.0),
            diagonal_sigma=diagonal_sigma,
        )

    for n in range(N):
        center, radius, mask, mask_bool = create_agent_mask_and_center(
            domain_size, rng,
            radius_range=cfg.get("agent_radius_range", (5, 10)),
            fixed_center=fixed_center
        )

        if use_global_belief:
            mu_q = global_mu_q * mask[..., None]
            sigma_q = global_sigma_q * mask[..., None, None]
            mu_q += rng.normal(scale=cfg.get("belief_noise_scale_mu", 0.01), size=mu_q.shape)
            sigma_q += rng.normal(scale=cfg.get("belief_noise_scale_sigma", 0.01), size=sigma_q.shape)
            sigma_q = sanitize_sigma(sigma_q, eps=1e-6)
        else:
            mu_q, sigma_q = generate_mu_sigma_fields(
                domain_size=domain_size, rng=rng, K=K_q,
                mu_range=config.q_mu_range, sigma_range=config.q_sigma_range,
                mask=mask, sigma_outside_val=cfg.get("sigma_outside_val", 1e3),
                dtype=dtype, smooth_sigma=cfg.get("init_smooth_sigma", 2.0),
                diagonal_sigma=diagonal_sigma,
            )

        if use_global_model:
            mu_p = global_mu_p * mask[..., None]
            sigma_p = global_sigma_p * mask[..., None, None]
            mu_p += rng.normal(scale=cfg.get("model_noise_scale_mu", 0.01), size=mu_p.shape)
            sigma_p += rng.normal(scale=cfg.get("model_noise_scale_sigma", 0.01), size=sigma_p.shape)
            sigma_p = sanitize_sigma(sigma_p, eps=1e-6)
        else:
            mu_p, sigma_p = generate_mu_sigma_fields(
                domain_size=domain_size, rng=rng, K=K_p,
                mu_range=config.p_mu_range, sigma_range=config.p_sigma_range,
                mask=mask, sigma_outside_val=cfg.get("sigma_outside_val", 1e3),
                dtype=dtype, smooth_sigma=cfg.get("init_smooth_sigma", 2.0),
                diagonal_sigma=diagonal_sigma,
            )

        phi, phi_model = initialize_phi_pair(domain_size, lie_algebra_dim, rng, phi_init, phi_model_offset, mask)
        phi       = soft_clip_phi_norm(phi, max_norm=np.pi)
        phi_model = soft_clip_phi_norm(phi_model, max_norm=np.pi)

        agent = construct_agent_object(
            agent_id=n,
            center=center,
            radius=radius,
            mask=mask,
            mu_q=mu_q, sigma_q=sigma_q,
            mu_p=mu_p, sigma_p=sigma_p,
            phi=phi, phi_model=phi_model,
            dtype=dtype
        )

        agent.generators_q = G_q
        agent.generators_p = G_p

        # Morphism initialization
        morphism_seed = rng.integers(0, 1e9)
        sigma = cfg.get("morphism_sigma", 1.5)
        rank  = cfg.get("morphism_rank", min(K_q, K_p))

                
        Phi_0, Phi_tilde_0 = generate_morphism_pair(
            *domain_size,
            K_q=K_q,
            K_p=K_p,
            rank=rank,
            sigma=sigma,
            seed=morphism_seed,
            mask=mask,
            dtype=dtype,
        )


        agent.Phi_0        = Phi_0
        agent.Phi_tilde_0  = Phi_tilde_0
       
        agent.morphisms_dirty = False  # Φ/Φ̃ are current
        initialize_agent_gradients(agent, config)
        agents.append(agent)

    assign_neighbors_vectorized(agents, overlap_eps=overlap_eps, max_neighbors=max_neighbors)
    return agents

def create_agent_mask(
    center: tuple[int, ...],
    radius: float,
    domain_size: tuple[int, ...]
) -> np.ndarray:
    """
    Generate a soft Gaussian mask and zero out values below support_cutoff_eps from config.
    """
    mask = generate_soft_mask(center, radius, domain_size)
    cutoff = getattr(config, 'support_cutoff_eps', 1e-3)
    mask[mask < cutoff] = 0.0
    return mask

def generate_soft_mask(
    center: tuple[int, ...],
    radius: float,
    domain_size: tuple[int, ...]
) -> np.ndarray:
    """
    Create a smooth soft mask (Gaussian decay) on a periodic domain.
    Distances wrap around edges in each dimension.
    """
    dtype = getattr(config, 'dtype', np.float64)

    # Build per‐axis index grids via ogrid
    grids = np.ogrid[tuple(slice(0, s) for s in domain_size)]  # one array per axis

    # Accumulate squared wrap‐aware distances
    dist_sq = 0.0
    for d, size in enumerate(domain_size):
        coord = grids[d]
        # raw distance from center
        delta = np.abs(coord - center[d])
        # wrap‐around distance
        delta = np.minimum(delta, size - delta)
        dist_sq = dist_sq + delta**2

    dist = np.sqrt(dist_sq)

    # Avoid division by zero
    r = max(radius, np.finfo(float).eps)
    mask = np.exp(-(dist / r)**2).astype(dtype)
    return mask



def construct_agent_object(
    agent_id,
    center,
    radius,
    mask,
    mu_q, sigma_q,
    mu_p, sigma_p,
    phi, phi_model,
    dtype=np.float32
) -> Agent:
    """
    Build a level-0 Agent (no parents, no cross-scale maps yet).
    """
    a = Agent(
        id=agent_id,
        center=center,
        radius=radius,
        mask=mask.astype(dtype),

        mu_q_field=mu_q.astype(dtype),
        sigma_q_field=sigma_q.astype(dtype),
        mu_p_field=mu_p.astype(dtype),
        sigma_p_field=sigma_p.astype(dtype),

        phi=phi.astype(dtype),
        phi_model=phi_model.astype(dtype),

        neighbors=[],
    )
    # Ensure level-0 & empty relations
    a.level = 0
    a.parent_ids = []
    a.child_ids = []
    # Start with no Λ (will be built when a parent emerges)
    
    return a



def generate_mu_sigma_fields( 
    domain_size,
    rng,
    K,
    mu_range,
    sigma_range,
    mask=None,
    sigma_outside_val=None,
    dtype=np.float32,
    smooth_sigma=None,
    diagonal_sigma=None,
    eps=None,
    floor=None,
):
    """
    Generate spatially varying (μ, Σ) fields with shape:
        μ     : (H, W, K)
        Σ     : (H, W, K, K)

    Args:
        domain_size        : (H, W)
        rng                : np.random.Generator
        K                  : int, fiber dimension
        mu_range           : tuple[float, float]
        sigma_range        : tuple[float, float]
        mask               : optional (H, W) float mask
        sigma_outside_val  : float, fallback Σ diagonal value outside support (must be positive)
        dtype              : np.float32 or np.float64
        smooth_sigma       : float, Gaussian blur width for smoothness
        diagonal_sigma     : bool, if True, initializes Σ diagonally
        eps                : float, minimal eigenvalue threshold for SPD enforcement
        floor              : float, floor value for numerical stability (not used here, reserved)

    Returns:
        mu_field     : (H, W, K)
        sigma_field  : (H, W, K, K)
    """
    

    # Set safe, covariance-appropriate defaults
    eps = eps if eps is not None else 1e-6
    sigma_outside_val = sigma_outside_val if sigma_outside_val is not None else 1e3
    smooth_sigma = smooth_sigma if smooth_sigma is not None else 2.0
    diagonal_sigma = diagonal_sigma if diagonal_sigma is not None else False


    if sigma_outside_val <= 0.0:
        raise ValueError("sigma_outside_val must be positive for covariance")

    H, W = domain_size

    # Generate smooth mean field μ
    mu = rng.uniform(*mu_range, size=(H, W, K)).astype(dtype)
    for k in range(K):
        mu[..., k] = gaussian_filter(mu[..., k], sigma=smooth_sigma)

    # Generate covariance Σ field
    if diagonal_sigma:
        sigma_diag = rng.uniform(*sigma_range, size=(H, W, K)).astype(dtype)
        for k in range(K):
            sigma_diag[..., k] = gaussian_filter(sigma_diag[..., k], sigma=smooth_sigma)
        sigma_field = np.zeros((H, W, K, K), dtype=dtype)
        for k in range(K):
            # Enforce minimal eigenvalue via clipping on diagonal entries
            sigma_field[..., k, k] = np.clip(sigma_diag[..., k], eps, None)
    else:
        # Generate full covariance by A A^T + eps I construction
        A = rng.normal(0, 1, size=(H, W, K, K)).astype(dtype)
        for i in range(K):
            for j in range(K):
                A[..., i, j] = gaussian_filter(A[..., i, j], sigma=smooth_sigma)
        Sigma = np.einsum("...ik,...jk->...ij", A, A)
        for i in range(K):
            Sigma[..., i, i] += eps  # enforce positive definiteness
        # Normalize trace to desired scale
        trace = np.trace(Sigma, axis1=-2, axis2=-1)[..., None, None]
        target_trace = K * np.mean(sigma_range)
        Sigma *= target_trace / (trace + eps)
        sigma_field = Sigma

    # Final sanitization to ensure SPD
    
    sigma_field = sanitize_sigma(sigma_field, eps=eps)

    # Apply mask: zero μ and set fallback covariance outside support
    if mask is not None:
        mask = mask.astype(dtype)
        mu_zero = np.zeros_like(mu)
        sigma_outside = sigma_outside_val * np.eye(K, dtype=dtype)

        mu = mask[..., None] * mu + (1.0 - mask[..., None]) * mu_zero
        sigma_field = (
            mask[..., None, None] * sigma_field +
            (1.0 - mask[..., None, None]) * sigma_outside[None, None, :, :]
        )

    return mu, sigma_field







def soft_clip_mu(mu, mu_clip=3.0):
    """
    Softly compresses μ to avoid values far from 0,
    without hard clipping (preserves gradients).

    Returns:
        μ' = μ / (1 + |μ| / μ_clip)
    """
    return mu / (1 + np.abs(mu) / mu_clip)



def initialize_phi_field(
    domain_size: tuple[int, ...],
    lie_algebra_dim: int,
    rng: np.random.Generator,
    init_scale: float,
    mask: np.ndarray,
    smooth_sigma: float = None
) -> np.ndarray:
    """
    Initialize and smooth a φ field with Gaussian noise and apply mask.
    """
    dtype = getattr(config, 'dtype', mask.dtype)
    sigma = smooth_sigma or getattr(config, 'phi_init_smooth_sigma', 0.5)

    phi_raw = rng.normal(scale=init_scale, size=(*domain_size, lie_algebra_dim)).astype(dtype)
    phi_smooth = gaussian_filter(phi_raw, sigma=sigma)
    return phi_smooth * mask[..., None]


def initialize_phi_pair(
    domain_size: tuple[int, ...],
    lie_algebra_dim: int,
    rng: np.random.Generator,
    phi_init: float,
    phi_model_offset: float,
    mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Initialize φ and φ̃ (belief and model) with optional offset for φ̃.

    If config.identical_models is True, then φ̃ = 0 identically.
    If phi_model_offset == 0.0, then φ̃ is also initialized as zero.
    Otherwise, φ̃ is an independent Lie algebra sample: φ̃ = Smooth[φ + R·ξ],
    where R is a random SO(3) rotation (on the Lie algebra basis), and ξ is Gaussian noise.
    """
    phi = initialize_phi_field(domain_size, lie_algebra_dim, rng, phi_init, mask)

    if getattr(config, "identical_models", False) or phi_model_offset == 0.0:
        phi_model = np.zeros_like(phi)
    else:
        # Sample noise
        delta = rng.normal(scale=phi_model_offset, size=(*domain_size, lie_algebra_dim)).astype(phi.dtype)

        # Apply random SO(3) rotation matrix to each φ component (shared over grid)
        from scipy.spatial.transform import Rotation
        R = Rotation.random(random_state=rng).as_matrix()  # (3, 3)
        delta_rot = np.einsum("ab,...b->...a", R, delta)   # (H, W, 3)

        phi_model_raw = phi + delta_rot
        phi_model = gaussian_filter(
            phi_model_raw,
            sigma=getattr(config, 'phi_init_smooth_sigma', 0.5)
        )
        phi_model *= mask[..., None]

    return phi, phi_model



def assign_neighbors_vectorized(
    agents: list,
    overlap_eps: float,
    max_neighbors: int
) -> None:
    """
    Assigns neighbors to each agent by computing pairwise mask overlaps.
    Vectorized for efficiency.

    A pair (i, j) is considered overlapping if:
        overlap(i, j) > overlap_eps × num_pixels
    """
    if max_neighbors == 0:
        for agent in agents:
            agent.neighbors = []
        return

    N = len(agents)
    dtype = getattr(config, 'dtype', np.float32)

    # Flatten masks into (N, H×W)
    masks_flat = np.stack([agent.mask.flatten().astype(dtype) for agent in agents])

    # Compute pairwise overlap matrix
    overlap_matrix = masks_flat @ masks_flat.T  # shape: (N, N)
    pixel_thresh = overlap_eps * masks_flat.shape[1]

    # Boolean matrix: True if overlap > threshold
    above_thresh = overlap_matrix > pixel_thresh
    np.fill_diagonal(above_thresh, False)  # Exclude self-overlap

    for i in range(N):
        sorted_idx = np.argsort(-overlap_matrix[i])  # descending order
        valid_neighbors = [j for j in sorted_idx if above_thresh[i, j]]
        agents[i].neighbors = [{"id": j} for j in valid_neighbors[:max_neighbors]]




def create_agent_mask_and_center(domain_size, rng, radius_range, fixed_center=None):
    """
    Sample agent center and radius, and generate its soft mask.

    Args:
        domain_size  : tuple[int, int] = (H, W)
        rng          : np.random.Generator
        radius_range : (float, float)
        fixed_center : optional tuple[int, int]

    Returns:
        center : (int, int) coordinates of agent center
        radius : float
        mask   : (H, W) soft float32 mask
        mask_bool : (H, W) boolean version of mask for edge decay
    """
    radius = rng.uniform(*radius_range)
    center = fixed_center or tuple(rng.integers(0, s) for s in domain_size)
    mask = create_agent_mask(center, radius, domain_size)
    return center, radius, mask.astype(np.float32), mask.astype(bool)




def initialize_agent_gradients(agent, config=None):
    """
    Initialize all gradient fields once μ/Σ/φ are populated.
    Creates both canonical names and legacy *(_field) aliases (as views).
    Safe to call multiple times.
    """
    # --- infer shapes ---
    mu_q   = getattr(agent, "mu_q_field", None)
    sg_q   = getattr(agent, "sigma_q_field", None)
    mu_p   = getattr(agent, "mu_p_field", None)
    sg_p   = getattr(agent, "sigma_p_field", None)
    phi_q  = getattr(agent, "phi", None)
    phi_p  = getattr(agent, "phi_model", None)

    # Create zeros if missing shapes (skip silently if field missing)
    if mu_q is not None:
        agent.grad_mu_q = np.zeros_like(mu_q)
        agent.grad_mu_q_field = agent.grad_mu_q
    if sg_q is not None:
        agent.grad_sigma_q = np.zeros_like(sg_q)
        agent.grad_sigma_q_field = agent.grad_sigma_q
    if mu_p is not None:
        agent.grad_mu_p = np.zeros_like(mu_p)
        agent.grad_mu_p_field = agent.grad_mu_p
    if sg_p is not None:
        agent.grad_sigma_p = np.zeros_like(sg_p)
        agent.grad_sigma_p_field = agent.grad_sigma_p

    if phi_q is not None:
        agent.grad_phi = np.zeros_like(phi_q)
    if phi_p is not None:
        agent.grad_phi_tilde = np.zeros_like(phi_p)




        
