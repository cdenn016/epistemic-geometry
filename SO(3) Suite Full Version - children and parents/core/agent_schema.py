from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional
import numpy as np

@dataclass
class Agent:
    # ----------------------------------------------------------------------------------
    # Core identity & support
    # ----------------------------------------------------------------------------------
    id: int
    center: Tuple[int, ...]
    radius: float
    mask: np.ndarray                      # (H,W) float32 in [0,1]

    # ----------------------------------------------------------------------------------
    # Gauge fields (SO(3) algebra; last dim = 3)
    # ----------------------------------------------------------------------------------
    phi: np.ndarray                       # (H,W,3)
    phi_model: np.ndarray                 # (H,W,3)

    # ----------------------------------------------------------------------------------
    # Bundle fields (belief/model)
    # ----------------------------------------------------------------------------------
    mu_q_field: np.ndarray                # (H,W,Kq)
    sigma_q_field: np.ndarray             # (H,W,Kq,Kq)
    mu_p_field: np.ndarray                # (H,W,Kp)
    sigma_p_field: np.ndarray             # (H,W,Kp,Kp)

    # Optional inverses (filled in preprocess)
    sigma_q_inv: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    sigma_p_inv: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    # ----------------------------------------------------------------------------------
    # Morphisms (transported bundle maps; rebuilt in preprocess when dirty)
    # ----------------------------------------------------------------------------------
    bundle_morphism_q_to_p: Optional[np.ndarray] = field(default=None, init=False, repr=False)  # (H,W,Kp,Kq)
    bundle_morphism_p_to_q: Optional[np.ndarray] = field(default=None, init=False, repr=False)  # (H,W,Kq,Kp)
    morphisms_dirty: bool = field(default=True, init=False, repr=False)

    # ----------------------------------------------------------------------------------
    # Gradient buffers
    # ----------------------------------------------------------------------------------
    grad_mu_q: Optional[np.ndarray] = field(default=None, init=False, repr=False)      # (H,W,Kq)
    grad_sigma_q: Optional[np.ndarray] = field(default=None, init=False, repr=False)   # (H,W,Kq,Kq)
    grad_mu_p: Optional[np.ndarray] = field(default=None, init=False, repr=False)      # (H,W,Kp)
    grad_sigma_p: Optional[np.ndarray] = field(default=None, init=False, repr=False)   # (H,W,Kp,Kp)

    grad_phi: np.ndarray = field(init=False, repr=False)          # (H,W,3)
    grad_phi_tilde: np.ndarray = field(init=False, repr=False)    # (H,W,3)

    # Optional morphism grads (kept minimal)
    grad_Phi: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    grad_Phi_tilde: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    # ----------------------------------------------------------------------------------
    # Fisher / diagnostics (optional; used by update_terms_phi / diags)
    # ----------------------------------------------------------------------------------
    inverse_fisher_phi: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    inverse_fisher_phi_model: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    inverse_fisher_Phi: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    inverse_fisher_Phi_tilde: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    # Alignment diagnostics (masked arrays; optional)
    cached_alignment_kl: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    cached_model_alignment_kl: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    # ----------------------------------------------------------------------------------
    # Neighborhood / hierarchy
    # ----------------------------------------------------------------------------------
    neighbors: List[Dict[str, Any]] = field(default_factory=list)
    level: int = 0                                      # 0=child, >=1=parent
    parent_ids: List[int] = field(default_factory=list)
    child_ids: List[int] = field(default_factory=list)
    child_weights: Dict[int, float] = field(default_factory=dict, repr=False)
    labels: Dict[int, float] = field(default_factory=dict, repr=False)

    # ----------------------------------------------------------------------------------
    # Lifecycle & spawn
    # ----------------------------------------------------------------------------------
    birth_step: int = 0
    emerged: bool = False
    _age: int = 0
    _grace: int = 0
    _shrink_strikes: int = 0
    seed_mask: Optional[np.ndarray] = field(default=None, init=False, repr=False)       # (H,W)
    _core_map: Optional[np.ndarray] = field(default=None, init=False, repr=False)       # (H,W)
    _region_proposal: Optional[np.ndarray] = field(default=None, init=False, repr=False)# (H,W)

    # Multi-level parent routing (optional)
    parent_id_by_level: Dict[int, int] = field(default_factory=dict, repr=False)

    # Logging
    metrics_log: List[Dict[str, Any]] = field(default_factory=list)

    # ----------------------------------------------------------------------------------
    # Init hooks
    # ----------------------------------------------------------------------------------
    def __post_init__(self):
        self.grad_phi = np.zeros_like(self.phi, dtype=np.float32)
        self.grad_phi_tilde = np.zeros_like(self.phi_model, dtype=np.float32)

    # ----------------------------------------------------------------------------------
    # Tiny helpers
    # ----------------------------------------------------------------------------------
    @property
    def hw(self) -> Tuple[int, int]:
        return int(self.mask.shape[0]), int(self.mask.shape[1])

    def support(self, tau: float = 0.30) -> np.ndarray:
        return (np.asarray(self.mask, np.float32) > float(tau))

    def is_parent(self) -> bool:
        return int(getattr(self, "level", 0)) >= 1

    def is_active_parent(self, min_children: int = 2) -> bool:
        return self.is_parent() and len(getattr(self, "child_ids", ()) or ()) >= int(min_children)

    def mark_morphism_dirty(self) -> None:
        self.morphisms_dirty = True
        self.bundle_morphism_q_to_p = None
        self.bundle_morphism_p_to_q = None

    def log_metrics(self, step: int, stats: Dict[str, Any]) -> None:
        import copy
        self.metrics_log.append(copy.deepcopy({'step': int(step), **stats}))
