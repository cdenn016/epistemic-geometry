# -*- coding: utf-8 -*-
"""
runtime_context.py
Lightweight, level-aware run context for emergent parents & detectors.
Keeps heavy imports out to avoid cycles; DetectorState is created lazily.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Set, Literal
 # --- add near top of file ---
import hashlib, json


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers (kept local to avoid import cycles)
# ─────────────────────────────────────────────────────────────────────────────


class _Dirty:
    def __init__(self, *args, **kwargs):
        # ... your existing fields ...
        self.step_tag = getattr(self, "step_tag", -1)

    def reset_step(self, step: int) -> None:
        """Step-scoped reset hook so callers can safely do dirty.reset_step(step)."""
        try:
            if int(step) != getattr(self, "step_tag", -1):
                self.step_tag = int(step)
                # If you keep per-step flags, clear them here:
                for attr in ("phi_touched", "sigma_touched", "morphism_touched", "link_touched"):
                    if hasattr(self, attr):
                        v = getattr(self, attr)
                        if isinstance(v, dict) or isinstance(v, set):
                            v.clear()
        except Exception:
            # last-resort: at least set the tag
            self.step_tag = int(step)



Fiber = Literal["q", "p"]

@dataclass
class WorkFlags:
    """
    Step-scoped 'what changed' tracker.
    - NOT a cache invalidator
    - Used to *schedule* optional recomputations/warming/metrics/viz
    """
    step_tag: int = -1

    # per-agent field touches
    phi_touched: Dict[int, Set[Fiber]] = field(default_factory=dict)     # {aid: {"q","p"}}
    sigma_touched: Dict[int, Set[Fiber]] = field(default_factory=dict)   # {aid: {"q","p"}}
    morphism_touched: Set[int] = field(default_factory=set)              # agents whose Φ/Φ̃ priors changed

    # per-pair links (child,parent) that had structure changes (level graph, mask expands, etc.)
    link_touched: Set[Tuple[int, int]] = field(default_factory=set)

    def reset_step(self, step: int) -> None:
        if step != self.step_tag:
            self.step_tag = int(step)
            self.phi_touched.clear()
            self.sigma_touched.clear()
            self.morphism_touched.clear()
            self.link_touched.clear()

    # ---- mark helpers ----
    def touch_phi(self, aid: int, which: Fiber) -> None:
        self.phi_touched.setdefault(int(aid), set()).add(which)

    def touch_sigma(self, aid: int, which: Fiber) -> None:
        self.sigma_touched.setdefault(int(aid), set()).add(which)

    def touch_morphism(self, aid: int) -> None:
        self.morphism_touched.add(int(aid))

    def touch_link(self, child_id: int, parent_id: int) -> None:
        self.link_touched.add((int(child_id), int(parent_id)))

    # ---- query helpers (for schedulers/warmers) ----
    def phi_changed(self, aid: int, which: Optional[Fiber] = None) -> bool:
        flags = self.phi_touched.get(int(aid), set())
        return bool(flags if which is None else (which in flags))

    def sigma_changed(self, aid: int, which: Optional[Fiber] = None) -> bool:
        flags = self.sigma_touched.get(int(aid), set())
        return bool(flags if which is None else (which in flags))

    def morphism_changed(self, aid: int) -> bool:
        return int(aid) in self.morphism_touched

    def link_changed(self, child_id: int, parent_id: int) -> bool:
        return (int(child_id), int(parent_id)) in self.link_touched



def _shape_sig(a):
    return tuple(int(x) for x in np.asarray(a).shape)

def _arr_digest(a: np.ndarray, *, tol: float | None = None) -> str:
    a = np.asarray(a, np.float32)
    if tol and tol > 0:
        q = np.round(a / tol) * tol
        b = q.tobytes()
    else:
        b = a.tobytes()
    return hashlib.sha256(b).hexdigest()

def _config_digest(cfg: dict) -> str:
    keys = [
        "group_name", "phi_clip", "periodic_wrap", "so3_irreps_are_orthogonal",
        "mask_edge_soften_sigma"
    ]
    sub = {k: cfg.get(k, None) for k in keys}
    s = json.dumps(sub, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(s.encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Central cache hub (per-step invalidation, namespaced)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CacheHub:
    step_tag: int = -1
    spaces: Dict[str, Dict[Any, Any]] = field(default_factory=lambda: {
    "exp": {}, "omega": {}, "lambda": {}, "morphism": {}, "theta": {}, "fisher": {}, "misc": {}, "jinv": {}
    })
    
    # persistent (not cleared each step)
    persist: Dict[str, Dict[Any, Any]] = field(default_factory=lambda: {
        "morph_base": {}
    })
    
    hits: int = 0
    misses: int = 0
    invalidations: int = 0


    def ns(self, name: str) -> Dict[Any, Any]:
        return self.spaces.setdefault(name, {})

    def nsp(self, name: str) -> Dict[Any, Any]:
        """Persistent namespace (not cleared on step)."""
        return self.persist.setdefault(name, {})

    #in sync step
    def clear_on_step(self, step: int) -> None:
        if step != self.step_tag:
            for d in self.spaces.values():
                d.clear()
            self.step_tag = step

    def stats(self) -> Dict[str, int]:
        return {k: len(v) for k, v in self.spaces.items()}

    
# ---- standardized key builders ----
    def key_E(self, *, agent_id: int, step: int, phi: np.ndarray,
              cfg: dict, wrap_flag: bool, tol: float | None = None):
        return ("E", int(agent_id), int(step),
                _shape_sig(phi), bool(wrap_flag), _config_digest(cfg),
                _arr_digest(phi, tol=tol))

    def key_Jinv(self, **kw):
        k = self.key_E(**kw)
        return ("Jinv",) + k[1:]  # same parts except artifact tag

    def key_Omega(self, *, agent_i: int, agent_j: int, step: int,
                  phi_i: np.ndarray, phi_j: np.ndarray, cfg: dict,
                  wrap_flag: bool, tol: float | None = None):
        di = _arr_digest(phi_i, tol=tol)
        dj = _arr_digest(phi_j, tol=tol)
        pair_sha = hashlib.sha256((di + "|" + dj).encode()).hexdigest()
        return ("Omega", int(agent_i), int(agent_j), int(step),
                _shape_sig(phi_i), bool(wrap_flag), _config_digest(cfg), pair_sha)

    # runtime_context.py — inside CacheHub
    def key_Fisher(self, *, agent_id: int, step: int,
                   mu: np.ndarray, Sigma: np.ndarray,
                   cfg: dict, which: str = "q", tol: float | None = None):
        # digest both μ and Σ; “which” disambiguates q vs p fiber
       
        mu  = np.asarray(mu,    np.float32)
        Sig = np.asarray(Sigma, np.float32)
        mu_sha  = _arr_digest(mu,  tol=tol)
        Sig_sha = _arr_digest(Sig, tol=tol)
        shape   = _shape_sig(Sig)  # (..., K, K)
        return ("Fisher", which, int(agent_id), int(step),
                shape, _config_digest(cfg), mu_sha, Sig_sha)

    def key_Lambda(self, *, child_id: int, parent_id: int, step: int,
                  phi_child: np.ndarray, phi_parent: np.ndarray,
                  cfg: dict, which: str, up: bool, tol: float | None = None):
        dc = _arr_digest(phi_child, tol=tol); dp = _arr_digest(phi_parent, tol=tol)
        return ("Lambda", which, "up" if up else "dn", int(child_id), int(parent_id),
                int(step), _config_digest(cfg), dc, dp)
    
    def key_Theta(self, *, child_id: int, parent_id: int, step: int,
                  phi_child_q: np.ndarray, phi_parent_q: np.ndarray,
                  phi_child_p: np.ndarray, phi_parent_p: np.ndarray,
                  cfg: dict, tol: float | None = None):
        # Θ depends on both fibers via Λ and Φ; digest both q/p pairs
        dqc = _arr_digest(phi_child_q, tol=tol); dqp = _arr_digest(phi_parent_q, tol=tol)
        dpc = _arr_digest(phi_child_p, tol=tol); dpp = _arr_digest(phi_parent_p, tol=tol)
        return ("Theta", int(child_id), int(parent_id), int(step),
                _config_digest(cfg), dqc, dqp, dpc, dpp)

    # ---- typed get/put with counters ----
    def get(self, ns_name: str, key):
        ns = self.ns(ns_name)
        val = ns.get(key)
        if val is None:
            self.misses += 1
        else:
            self.hits += 1
        return val

    def put(self, ns_name: str, key, value):
        self.ns(ns_name)[key] = value

    def invalidate_like(self, ns_name: str, predicate):
        ns = self.ns(ns_name)
        todel = [k for k in list(ns.keys()) if predicate(k)]
        for k in todel:
            ns.pop(k, None)
        self.invalidations += len(todel)

    def full_stats(self) -> Dict[str, int]:
        out = self.stats()
        out.update({"hits": self.hits, "misses": self.misses, "invalidations": self.invalidations})
        return out
   
   





# ─────────────────────────────────────────────────────────────────────────────
# Graph/link carriers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LevelGraph:
    level: int
    agent_ids: List[int]                                     # ordered ids at this level
    neighbors: Dict[int, List[int]] = field(default_factory=dict)  # id -> [neighbor ids]



@dataclass
class CrossScaleLink:
    low_level: int
    high_level: int
    child_ids: List[int]                                     # ids at low_level
    parent_ids: List[int]                                    # ids at high_level
    labels: np.ndarray                                       # shape (len(child_ids),), parent id or -1





# ─────────────────────────────────────────────────────────────────────────────
# Unified Runtime Context (legacy views + level-graphs + cross-scale links)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RuntimeCtx:
    # ---- Legacy lvl-1 views (kept for backward compat) ---------------------

    global_step: int = 0


    # ---- Multi-level stores (sources of truth) -----------------------------
    parents_by_level: Dict[int, Dict[int, Any]] = field(default_factory=dict)       # L -> {pid: Agent}
    next_parent_id_by_level: Dict[int, int] = field(default_factory=dict)           # L -> next pid
    detector_state_by_level: Dict[int, Any] = field(default_factory=dict)           # L -> DetectorState
    children_latest_by_level: Dict[int, Any] = field(default_factory=dict)          # (L-1) -> list[Agent] for viz

    # New, level-agnostic registries
    agents_by_level: Dict[int, Dict[int, Any]] = field(default_factory=dict)        # L -> {id: Agent}
    graphs: Dict[int, LevelGraph] = field(default_factory=dict)                     # L -> LevelGraph
    xlinks: Dict[Tuple[int, int], CrossScaleLink] = field(default_factory=dict)     # (L_lo, L_hi) -> link

    # Central cache hub
    cache: CacheHub = field(default_factory=CacheHub)
    dirty: _Dirty = field(default_factory=_Dirty)
    
   
        
    # -----------------------------------------------------------------------
    # Level-aware accessors (new, backward-compatible)
    # -----------------------------------------------------------------------
    def get_detector_state(self, level: int):
        """Return the DetectorState for a level, creating it lazily."""
        ds = self.detector_state_by_level.get(level)
        if ds is None:
            # Lazy import avoids cycles just like _lazy_detector_factory
            try:
                from detector import DetectorState  # local import
                ds = DetectorState()
            except Exception:
                class _Stub:
                    pass
                ds = _Stub()
            self.detector_state_by_level[level] = ds
        return ds
    
    def get_parent_registry(self, level: int):
        """
        Return (registry_dict, next_parent_id) for a level.
        Callers can mutate the dict in-place; remember to persist with set_parent_registry.
        """
        reg = self.parents_by_level.setdefault(level, {})
        next_pid = int(self.next_parent_id_by_level.get(level, 1))
        return reg, next_pid
    
    def set_parent_registry(self, level: int, registry: dict, next_parent_id: int):
        """Persist registry and next_parent_id for a level."""
        self.parents_by_level[level] = registry or {}
        self.next_parent_id_by_level[level] = int(next_parent_id)

    
    
    def register_agents(self, level: int, agents: List[Any]) -> None:
        """Install/refresh the agent registry for a level."""
        lvl = self.agents_by_level.setdefault(level, {})
        lvl.clear()
        for a in agents:
            aid = int(getattr(a, "id"))
            lvl[aid] = a

    def build_neighbors(self, level: int, neighbor_map: Dict[int, List[int]]) -> None:
        """Install a neighbor graph for a level. IDs must be in agents_by_level[level]."""
        ids = list(self.agents_by_level.get(level, {}).keys())
        nm: Dict[int, List[int]] = {i: [j for j in neighbor_map.get(i, []) if j in ids] for i in ids}
        self.graphs[level] = LevelGraph(level=level, agent_ids=ids, neighbors=nm)

    def neighbors_of(self, level: int, agent_id: int) -> List[int]:
        """Return same-level neighbor ids (fallback to empty list)."""
        g = self.graphs.get(level)
        if not g:
            return []
        return g.neighbors.get(agent_id, [])

    def agent(self, level: int, agent_id: int) -> Any:
        """Fetch agent by (level, id)."""
        return self.agents_by_level[level][agent_id]

    def agents(self, level: int) -> List[Any]:
        """Ordered list of agents at a level."""
        if level not in self.graphs:
            return list(self.agents_by_level.get(level, {}).values())
        g = self.graphs[level]
        return [self.agents_by_level[level][aid] for aid in g.agent_ids]

    # -----------------------------------------------------------------------
    # Cross-scale: labels & Λ fields
    # -----------------------------------------------------------------------
    def set_crossscale_labels(
        self, low_level: int, high_level: int,
        child_ids: List[int], parent_ids: List[int], labels: np.ndarray
    ) -> None:
        """Create/refresh the cross-scale link with labeling."""
        labels = np.asarray(labels).astype(int)
        self.xlinks[(low_level, high_level)] = CrossScaleLink(
            low_level=low_level, high_level=high_level,
            child_ids=[int(x) for x in child_ids],
            parent_ids=[int(x) for x in parent_ids],
            labels=labels,
        )

    

    def crossscale_link_touching(self, level: int) -> List[CrossScaleLink]:
        """Return all links where this level participates (as low or high)."""
        out: List[CrossScaleLink] = []
        for (lo, hi), lk in self.xlinks.items():
            if lo == level or hi == level:
                out.append(lk)
        return out

    def parent_of_child(self, low_level: int, high_level: int, child_id: int) -> int:
        """Return parent id for child or -1 if unlabeled."""
        lk = self.xlinks.get((low_level, high_level))
        if not lk:
            return -1
        try:
            idx = lk.child_ids.index(int(child_id))
        except ValueError:
            return -1
        return int(lk.labels[idx]) if 0 <= idx < len(lk.labels) else -1

    
    def cache_stats(self) -> dict:
        """Return sizes by namespace for debugging."""
        return {} if not getattr(self, "cache", None) else self.cache.stats()
    
    

    def with_level(self, L: int, detector_factory: Optional[Callable[[], Any]] = None) -> "RuntimeCtx":
        self.mount_level(L, detector_factory=detector_factory)
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Convenience
# ─────────────────────────────────────────────────────────────────────────────

def ensure_runtime_defaults(ctx: Optional[RuntimeCtx]) -> RuntimeCtx:
    if ctx is None:
        ctx = RuntimeCtx()
    if ctx.global_step is None:
        ctx.global_step = 0
    if not hasattr(ctx, "dirty") or ctx.dirty is None:
        ctx.dirty = WorkFlags(step_tag=ctx.global_step)
    return ctx




__all__ = [
    "CacheHub",
    "LevelGraph",
    "CrossScaleLink",
    "RuntimeCtx",
    "ensure_runtime_defaults",
]
