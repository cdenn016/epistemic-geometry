# -*- coding: utf-8 -*-
"""
Parent utilities: geometry, set/IoU helpers, proposal gating, match/scoring,
and a few spawn-time conveniences.

No legacy mirrors; no runtime_ctx/Agent constructors here.
"""

from __future__ import annotations
import hashlib
import numpy as np
from core.numerical_utils import resize_nn  # used by child_activity_map
import core.config as config

# ---------------------------------------------------------------------------
# Basic geometry on toroidal domains
# ---------------------------------------------------------------------------




# ---------- small primitives ----------
def as_parent_dict(pars):
    if isinstance(pars, dict):
        return pars
    d = {}
    for P in (pars or []):
        pid = int(getattr(P, "id", len(d)))
        d[pid] = P
    return d

def iou_bool(a, b):
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    if not a.any() and not b.any(): return 0.0
    inter = int((a & b).sum()); union = int(a.sum() + b.sum() - inter)
    return inter / max(1, union)

def centroid_bool(mask_bool):
    yyx = np.argwhere(np.asarray(mask_bool, bool))
    if yyx.size == 0: return None
    yx = yyx.mean(0)
    return float(yx[0]), float(yx[1])

def mask_hash_bool(mask_bool: np.ndarray) -> int:
    """Stable 64-bit hash for boolean masks."""
    mv = np.packbits(np.asarray(mask_bool, bool).astype(np.uint8), bitorder="little")
    return int(hashlib.blake2b(mv, digest_size=8).hexdigest(), 16)


# ---------- coarse-grain trigger ----------
def need_cg_now(P, *, step_now, abs_core, rel_core, min_core_px, mask_tau):
    """
    Decide if parent P needs coarse-graining this step.
      - newborn freeze window passed
      - core area >= min_core_px
      - mask core changed since last CG beyond threshold
      - flagged dirty (morphisms_dirty OR _need_cg/_needs_cg)
    """
    if int(getattr(P, "freeze_until", -1)) > int(step_now):
        return False

    pm = np.asarray(getattr(P, "mask", 0.0), np.float32)
    if pm.size == 0:
        return False

    thr  = max(float(abs_core), float(rel_core) * float(pm.max()))
    core = (pm > thr)
    if int(core.sum()) < int(min_core_px):
        return False

    last_h = getattr(P, "_last_cg_core_hash", None)
    cur_h  = mask_hash_bool(core)
    changed_core = (last_h != cur_h)

    explicit_dirty = bool(
        getattr(P, "morphisms_dirty", False)
        or getattr(P, "_need_cg", False)
        or getattr(P, "_needs_cg", False)
    )

    if changed_core or explicit_dirty:
        P._pending_core_hash = cur_h    # stash for post-commit
        return True
    return False


# ---------- construction / spawn ----------
def _make_parent_template(
    ctx, tmpl, *, pid: int, H: int, W: int, Kq: int, Kp: int,
    generators_q, generators_p
):
    """
    Construct a brand-new level-1 parent with safe defaults (no per-agent caches).
    """
    AgentCls = tmpl.__class__
    dtype = getattr(tmpl.mu_q_field, "dtype", np.float32)

    # zero means; identity covariances (SPD)
    eye_q = np.broadcast_to(np.eye(Kq, dtype=dtype), (H, W, Kq, Kq)).copy()
    eye_p = np.broadcast_to(np.eye(Kp, dtype=dtype), (H, W, Kp, Kp)).copy()

    # algebra fields (SO(3): d=3)
    phi0   = np.zeros((H, W, 3), dtype=dtype)
    phi_t0 = np.zeros((H, W, 3), dtype=dtype)

    P = AgentCls(
        id=pid,
        center=(0, 0),
        radius=0.0,
        mask=np.zeros((H, W), np.float32),

        # gauge fields
        phi=phi0,
        phi_model=phi_t0,

        # bundle fields
        mu_q_field=np.zeros((H, W, Kq), dtype=dtype),
        sigma_q_field=eye_q,
        mu_p_field=np.zeros((H, W, Kp), dtype=dtype),
        sigma_p_field=eye_p,
    )

    # lifecycle / bookkeeping
    P.level = 1
    P.emerged = True
    P._age = 0
    P._grace = int(getattr(config, "parent_grace_steps", 7))
    P.birth_step = int(getattr(ctx, "global_step", 0))
    P._region_proposal = None

    # hierarchy placeholders
    P.child_ids     = []
    P.child_weights = {}
    P.parent_ids    = []
    P.neighbors     = []

    # morphisms are built later by preprocess (ctx/cache aware)

    P.morphisms_dirty = True

    # gradient buffers
    P.grad_mu_q    = np.zeros_like(P.mu_q_field,    dtype=dtype)
    P.grad_sigma_q = np.zeros_like(P.sigma_q_field, dtype=dtype)
    P.grad_mu_p    = np.zeros_like(P.mu_p_field,    dtype=dtype)
    P.grad_sigma_p = np.zeros_like(P.sigma_p_field, dtype=dtype)

    # SPD tidy
    def _spd_tidy(S):
        S = 0.5 * (S + np.swapaxes(S, -1, -2))
        k = S.shape[-1]
        return S + (1e-8 * np.eye(k, dtype=S.dtype))
    P.sigma_q_field = _spd_tidy(P.sigma_q_field)
    P.sigma_p_field = _spd_tidy(P.sigma_p_field)

    # debug breadcrumbs (guarded)
    if bool(getattr(config, "debug_spawn_log", False)):
        try:
            import traceback, time
            P._created_where = "parent_spawn._make_parent_template"
            P._created_when  = time.time()
            P._created_stack = "".join(traceback.format_stack(limit=12))
        except Exception:
            P._created_where = "parent_spawn._make_parent_template"
            P._created_when  = None
            P._created_stack = None

    return P


def _spawn_parent(_RC, agents, *, pid, H, W, Kq, Kp, Gq, Gp, prop, dtype,
                  support_tau, gkids, peak_child_weights, step_now,
                  per_x, per_y, chosen_centers):
    """Create & initialize a brand-new parent, return (P, support_bool)."""
    P = _make_parent_template(
        _RC, agents[0],
        pid=pid, H=H, W=W, Kq=Kq, Kp=Kp,
        generators_q=Gq, generators_p=Gp
    )

    # lifecycle
    P._region_proposal = prop
    P.emerged = True
    P.birth_step = int(step_now)
    P._age = 0
    P._grace = int(getattr(config, "parent_freeze_steps", 0))

    # ids/weights
    cw = peak_child_weights
    if isinstance(cw, dict) and cw:
        P.child_weights = {int(k): float(v) for k, v in cw.items()}
        P.child_ids = tuple(sorted(P.child_weights.keys()))
    else:
        P.child_ids = tuple(sorted(gkids))
        P.child_weights = {} if len(gkids) == 0 else {int(cid): 1.0 / len(gkids) for cid in gkids}

    # mask + centroid
    supp = apply_mask_and_centroid(
        P, prop, dtype=dtype, support_tau=support_tau,
        H=H, W=W, chosen_centers=chosen_centers, pid=pid,
        per_x=per_x, per_y=per_y, fallback_cyx=(None, None)
    )
    return P, supp





# ---------- matching / weights ----------
def update_parent_weights(P, peak_child_weights, *, replace_thr, weight_ema):
    """
    Update P.child_weights (+ids) using either hard replace (if J>=replace_thr)
    or EMA blend; returns True if overwrite in place, False if blended.
    """
    cw = peak_child_weights
    if not (isinstance(cw, dict) and cw):
        return False  # handled by caller

    old = set(getattr(P, "child_ids", ()))
    new = set(int(k) for k in cw.keys())

    try:
        J = jaccard_sets(old, new) if old else 1.0
    except Exception:
        J = 1.0 if not old else 0.0

    if J >= float(replace_thr):
        P.child_weights = {int(k): float(v) for k, v in cw.items()}
        P.child_ids = tuple(sorted(P.child_weights.keys()))
        return True

    # EMA blend
    weights = dict(getattr(P, "child_weights", {}))
    if not weights and old:
        weights = {int(k): 1.0 / len(old) for k in old}
    keys = set(weights.keys()) | set(int(x) for x in cw.keys())
    tot = 0.0
    for k in keys:
        v_old = float(weights.get(k, 0.0))
        v_new = float(cw.get(k, 0.0))
        weights[k] = (1.0 - float(weight_ema)) * v_old + float(weight_ema) * v_new
        tot += weights[k]
    if tot > 0.0:
        inv = 1.0 / tot
        for k in weights:
            weights[k] *= inv
    P.child_weights = weights
    P.child_ids = tuple(sorted(weights.keys()))
    return False


def match_best_parent(seed_bool, gkids, *, exists_support, exists_kids,
                      H, W, per_x, per_y, R2, match_iou_thr, child_jac_thr):
    """Return (best_pid, best_score_tuple) or (None, (-1,-1))."""
    best_pid, best_sc = None, (-1.0, -1.0)
    for pid, supp in exists_support.items():
        py, px = centroid_toroidal(supp, H, W, per_y=per_y, per_x=per_x)
        close = (dist2_toroidal(py, px, getattr(seed_bool, "cy", py), getattr(seed_bool, "cx", px),
                                H, W, per_y, per_x) <= R2) if hasattr(seed_bool, "cy") else False
        sc = match_score_for_parent(
            seed_bool, gkids, supp, exists_kids[pid],
            close=close, match_iou_thr=match_iou_thr, child_jac_thr=child_jac_thr
        )
        if sc is not None and sc > best_sc:
            best_sc, best_pid = sc, pid
    return best_pid, best_sc


# ---------- orchestrator helpers ----------
# ---------- orchestrator helpers ----------
def ensure_parent_registry(ctx, agents, level: int):
    """
    Level-aware parent registry accessor.
    Returns (existing_registry_dict, next_parent_id, prev_ids_set, H, W).
    """
    H, W = np.asarray(agents[0].mask, np.float32).shape[:2]
    existing, next_parent_id = ctx.get_parent_registry(level)
    # Normalize to dict if a legacy structure snuck in.
    if not isinstance(existing, dict):
        existing = as_parent_dict(existing)
        ctx.set_parent_registry(level, existing, next_parent_id)
    prev_ids = set(existing.keys())
    return existing, int(next_parent_id), prev_ids, H, W






def compute_cover2_map(agents, tau_sup):
    Ms = np.stack(
        [(np.asarray(ch.mask, np.float32) >= float(tau_sup)).astype(np.uint8) for ch in agents],
        axis=0
    )
    return (Ms.sum(0) >= 2)


def existing_cores_and_centroids(existing, abs_core, rel_core):
    cores, cents = [], []
    for P in existing.values():
        pm = np.asarray(getattr(P, "mask", 0.0), np.float32)
        mmax = float(pm.max()) if pm.size else 0.0
        thr  = max(float(abs_core), float(rel_core) * mmax)
        core = (pm > thr)
        cores.append(core)
        yyx = np.argwhere(core)
        if yyx.size == 0: cents.append(None)
        else:
            yx = yyx.mean(0); cents.append((float(yx[0]), float(yx[1])))
    return cores, cents


def filter_spawn_regions(regions, cover2, min_cover2_px, existing_cores, existing_centroids, spawn_nms_iou, spawn_min_dist_px):
    kept = []
    for R in (regions or []):
        m = np.asarray(getattr(R, "mask", 0.0), np.float32) > 0.0
        if not m.any(): continue
        if isinstance(cover2, np.ndarray):
            if int((m & cover2).sum()) < int(min_cover2_px): continue
        cyx = centroid_bool(m)
        novel = True
        for ec, c0 in zip(existing_cores, existing_centroids):
            if not np.asarray(ec).any(): continue
            if iou_bool(m, ec) >= float(spawn_nms_iou):
                novel = False; break
            if cyx is not None and c0 is not None:
                if (abs(cyx[0]-c0[0]) + abs(cyx[1]-c0[1])) < float(spawn_min_dist_px):
                    novel = False; break
        if novel: kept.append(R)
    return kept


def merge_parents_into_registry(ctx, parents_changed, level: int):
    parents_changed = as_parent_dict(parents_changed)
    existing, next_parent_id = ctx.get_parent_registry(level)
    step_now = int(getattr(ctx, "global_step", 0))
    freeze_steps = int(getattr(config, "parent_freeze_steps", 3))
    for pid, P in parents_changed.items():
        existing[int(getattr(P, "id", pid))] = P
        pm = np.asarray(getattr(P, "mask", 0.0), np.float32)
        P.seed_mask = pm.copy()
        if not hasattr(P, "_age"):   P._age = 0
        P._born_step   = step_now
        P.freeze_until = step_now + freeze_steps
        if not hasattr(P, "_grace"): P._grace = freeze_steps
        # mark dirty in a consistent way
        setattr(P, "_need_cg", True)
        setattr(P, "_needs_cg", True)
        setattr(P, "morphisms_dirty", True)
    # Persist registry & next id back to the level store.
    ctx.set_parent_registry(level, existing, next_parent_id)
    return parents_changed, step_now


def select_cg_targets_stable(parents_changed, prev_ids, *, step_now, abs_core, rel_core, min_core_px, mask_tau):
    # only those that existed before this spawn call
    items_all = [(pid, P) for (pid, P) in parents_changed.items() if int(pid) in prev_ids]
    cg_items  = []
    for pid, P in items_all:
        if need_cg_now(P, step_now=step_now, abs_core=abs_core, rel_core=rel_core,
                       min_core_px=min_core_px, mask_tau=mask_tau):
            cg_items.append(P)
    return items_all, cg_items






def novelty_or_skip(seed_bool, *, exists_support, H, W, min_novel_px, min_novel_frac):
    """Novelty gate over existing supports; returns (ok, novel_px, novel_frac)."""
    return novelty_gate_ok(
        seed_bool, exists_support, H=H, W=W,
        min_novel_px=min_novel_px, min_novel_frac=min_novel_frac
    )


def continuity_try_rescue(cy, cx, seed_bool, *, exists_support, H, W,
                          per_x, per_y, cont_r2, cont_iou_thr):
    """Return best_pid or None for continuity-rescue."""
    cont = continuity_match_pid(
        cy, cx, seed_bool, exists_support,
        H=H, W=W, per_x=per_x, per_y=per_y,
        cont_r2=cont_r2, cont_iou_thr=cont_iou_thr
    )
    return (cont[1] if cont is not None else None)


def conservative_child_set(agents, seed_bool, *, mask_tau_child, min_overlap_px):
    """Intersection-based kid set from seed (conservative)."""
    return set(conservative_child_set_from_seed(
        agents, seed_bool, mask_tau=mask_tau_child, min_overlap_px=min_overlap_px
    ))


def apply_mask_and_centroid(P, prop, *, dtype, support_tau, H, W,
                            chosen_centers, pid, per_x, per_y, fallback_cyx):
    """
    Set mask, return support_bool; also appends centroid to chosen_centers.
    """
    P.mask = np.clip(prop, 0.0, 1.0).astype(dtype, copy=False)
    supp = (np.asarray(P.mask, np.float32) > float(support_tau))
    append_center_from_mask(
        chosen_centers, supp, pid, H, W,
        per_x=per_x, per_y=per_y, fallback=fallback_cyx
    )
    return supp


def assert_parent_invariants(P, H, W) -> bool:
    def _shp(x):
        try: return tuple(np.asarray(x).shape)
        except: return None
    ok = True
    msg = []
    for name in ("mu_q_field","sigma_q_field","mu_p_field","sigma_p_field","phi","phi_model","mask"):
        val = getattr(P, name, None)
        if val is None: ok = False; msg.append(f"{name}=None")
    if ok:
        mq, sq = _shp(P.mu_q_field), _shp(P.sigma_q_field)
        mp, sp = _shp(P.mu_p_field), _shp(P.sigma_p_field)
        if (mq is None or len(mq)!=3 or mq[0]!=H or mq[1]!=W): ok=False; msg.append(f"mu_q_field{mq}")
        if (sq is None or len(sq)!=4 or sq[0]!=H or sq[1]!=W or sq[-1]!=sq[-2]): ok=False; msg.append(f"sigma_q_field{sq}")
        if (mp is None or len(mp)!=3 or mp[0]!=H or mp[1]!=W): ok=False; msg.append(f"mu_p_field{mp}")
        if (sp is None or len(sp)!=4 or sp[0]!=H or sp[1]!=W or sp[-1]!=sp[-2]): ok=False; msg.append(f"sigma_p_field{sp}")
    if (not ok) and bool(getattr(config, "debug_parent_invariants", False)):
        where = getattr(P, "_created_where", "?")
        print(f"[PARENT_INVARIANT_FAIL] pid={getattr(P,'id','?')} where={where} details={';'.join(msg)}")
        stk = getattr(P, "_created_stack", None)
        if stk: print(stk)
    return ok



# ---------- thin wrappers (keep; no QQ-duplicates) ----------
def seed_proposal_or_skip(peak, *, H, W, agents, local_tau, min_local_px,
                          mask_tau_child, support_tau, min_seed_px):
    """
    Returns:
      - (prop, seed_bool, seed_px) on success
      - None on failure
    """
    prop = peak.get("proposal", peak.get("mask"))
    out = gate_and_seed_proposal(
        prop, H=H, W=W, local_tau=local_tau, min_local_px=min_local_px,
        agents=agents, mask_tau_child=mask_tau_child,
        support_tau=support_tau, min_seed_px=min_seed_px
    )
    if out is None:
        return None
    seed_bool, seed_px = out
    return prop, seed_bool, seed_px








# ---------------------------------------------------------------------------
# Basic geometry on toroidal domains
# ---------------------------------------------------------------------------

def dist2_toroidal(y0: float, x0: float, y1: float, x1: float,
                   H: int, W: int, per_y: bool, per_x: bool) -> float:
    """Squared distance with optional wrap in y/x."""
    dy = abs(y0 - y1)
    dx = abs(x0 - x1)
    if per_y:
        dy = min(dy, H - dy)
    if per_x:
        dx = min(dx, W - dx)
    return dy * dy + dx * dx


def centroid_toroidal(mask_bool: np.ndarray, H: int, W: int, *, per_y: bool, per_x: bool):
    """
    Robust centroid on a torus by projecting to both wraps and picking the
    lower-variance projection. For typical compact regions, regular mean works.
    """
    m = np.asarray(mask_bool, bool)
    if not m.any():
        return float(H / 2.0), float(W / 2.0)

    yy, xx = np.nonzero(m)
    if not per_y and not per_x:
        return float(yy.mean()), float(xx.mean())

    def _wrap_mean(coords, N, wrap):
        if not wrap:
            return coords.mean()
        ang = coords * (2.0 * np.pi / N)
        c = np.cos(ang).mean()
        s = np.sin(ang).mean()
        mu = np.arctan2(s, c) * (N / (2.0 * np.pi))
        return float((mu + N) % N)

    cy = _wrap_mean(yy.astype(np.float64), H, per_y)
    cx = _wrap_mean(xx.astype(np.float64), W, per_x)
    return float(cy), float(cx)


def append_center_from_mask(centers: list, mask_bool: np.ndarray, pid: int,
                            H: int | None = None, W: int | None = None, *,
                            per_x: bool, per_y: bool, fallback=None):
    """
    Compute (cy,cx) toroidally for a boolean mask and append (pid, cy, cx).
    Falls back to provided center (or image center) if mask is empty.
    """
    m = np.asarray(mask_bool, bool)
    h0, w0 = m.shape[:2]
    H = int(h0 if H is None else H)
    W = int(w0 if W is None else W)

    if m.any():
        cy, cx = centroid_toroidal(m, H, W, per_y=per_y, per_x=per_x)
    else:
        if fallback is not None:
            cy, cx = map(float, fallback)
        else:
            cy, cx = float(h0 / 2.0), float(w0 / 2.0)

    centers.append((float(cy), float(cx), int(pid)))


def spawn_spacing_radius(allow_overlap, default_px: int) -> int:
    """Return the minimum spacing radius in pixels for seeds."""
    return 0 if bool(allow_overlap) else int(default_px)


# ---------------------------------------------------------------------------
# Set/boolean helpers used by match/spawn logic
# ---------------------------------------------------------------------------




def jaccard_sets(A: set, B: set) -> float:
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / float(len(A | B))



# ---------------------------------------------------------------------------
# Activity & coverage helpers (used by detector/spawn prefilter)
# ---------------------------------------------------------------------------

def QQchild_activity_map(
    ch, H, W,
    # belief (q) thresholds
    mu_tau=1e-3, trsig_tau=1e-3,
    # misc
    eps=1e-6,
    # model (p) thresholds (None ⇒ pull from config or fall back to q thresholds)
    mu_tau_p=None, trsig_tau_p=None,
    # mixing controls (None ⇒ pull from config)
    mix_mode=None,                 # 'weighted_or' | 'and' | 'or' | 'model_only' | 'belief_only'
    w_model=None, w_belief=None,   # weights for 'weighted_or'
    score_tau=None                 # threshold for 'weighted_or'
):
    """
    Boolean "activity" core for a child based on (μ, tr Σ) gates, optionally mixing
    belief/model fibers. Always gated by the child's mask (eps).
    """
   
    

    m = getattr(ch, "mask", None)
    if m is None:
        return None
    m_ok = (np.asarray(m, np.float32) > eps)

    mu_q, Sig_q = getattr(ch, "mu_q_field", None), getattr(ch, "sigma_q_field", None)
    mu_p, Sig_p = getattr(ch, "mu_p_field", None), getattr(ch, "sigma_p_field", None)

    if ((mu_q is None or Sig_q is None) and (mu_p is None or Sig_p is None)):
        return m_ok

    if mu_tau_p is None:
        mu_tau_p = float(getattr(config, "activity_mu_tau", mu_tau))
    if trsig_tau_p is None:
        trsig_tau_p = float(getattr(config, "activity_trsig_tau", trsig_tau))
    if w_model is None:
        w_model = float(getattr(config, "activity_model_weight", 0.1))
    if w_belief is None:
        w_belief = float(getattr(config, "activity_belief_weight", 0.9))
    if mix_mode is None:
        mix_mode = getattr(config, "activity_mix_mode", "weighted_or")
    if score_tau is None:
        score_tau = float(getattr(config, "activity_score_tau", 0.5))

    def _strength(mu, Sig, mu_thr, tr_thr):
        mu = np.asarray(mu); Sig = np.asarray(Sig)
        if mu.shape[:2] != (H, W):  mu = resize_nn(mu, (H, W))
        if Sig.shape[:2] != (H, W): Sig = resize_nn(Sig, (H, W))
        mu_mag = np.sqrt((mu ** 2).sum(-1)) if mu.ndim == 3 else np.abs(mu).astype(np.float32)
        trsig  = np.trace(Sig, axis1=-2, axis2=-1) if Sig.ndim == 4 else Sig.astype(np.float32)
        return (mu_mag > float(mu_thr)) | (trsig > float(tr_thr))

    b = _strength(mu_q, Sig_q, mu_tau, trsig_tau) if (mu_q is not None and Sig_q is not None) else None
    p = _strength(mu_p, Sig_p, mu_tau_p, trsig_tau_p) if (mu_p is not None and Sig_p is not None) else None

    if mix_mode == "model_only":
        act_core = p if p is not None else (b if b is not None else False)
    elif mix_mode == "belief_only":
        act_core = b if b is not None else (p if p is not None else False)
    elif mix_mode == "and":
        act_core = (b if b is not None else True) & (p if p is not None else True)
    elif mix_mode == "or":
        act_core = (b if b is not None else False) | (p if p is not None else False)
    else:
        b_f = b.astype(np.float32) if b is not None else 0.0
        p_f = p.astype(np.float32) if p is not None else 0.0
        score = w_model * p_f + w_belief * b_f
        act_core = (score >= float(score_tau))

    return m_ok & act_core


def cover_ge2_mask(children, H: int, W: int, mask_tau: float) -> np.ndarray:
    """Boolean mask: pixels covered by at least two children (m >= mask_tau)."""
    cover = np.zeros((H, W), np.int16)
    for ch in children:
        m = np.asarray(getattr(ch, "mask", 0.0), np.float32)
        cover += (m >= float(mask_tau)).astype(np.int16)
    return (cover >= 2)


def clamp_to_multichild(proposal: np.ndarray, children, H: int, W: int, mask_tau: float) -> np.ndarray:
    """Clamp proposal to ≥2-child cover (belt & suspenders)."""
    prop = np.asarray(proposal, np.float32)
    return prop * cover_ge2_mask(children, H, W, mask_tau).astype(np.float32)


def conservative_child_set_from_seed(children, seed_bool: np.ndarray, *,
                                     mask_tau: float, min_overlap_px: int) -> list[int]:
    """Return sorted list of child ids that overlap the seed by at least min_overlap_px."""
    ids = []
    for ch in children:
        m = np.asarray(getattr(ch, "mask", 0.0), np.float32)
        ov_px = int(np.logical_and(seed_bool, m >= float(mask_tau)).sum())
        if ov_px >= int(min_overlap_px):
            try:
                ids.append(int(getattr(ch, "id")))
            except Exception:
                pass
    return sorted(ids)


# ---------------------------------------------------------------------------
# Spawn/match helpers (imported by parent_spawn.py)
# ---------------------------------------------------------------------------

def child_union_for_ids(agents, child_ids, H: int, W: int, mask_tau: float = 0.10) -> np.ndarray:
    """Union of child supports over ids (mask >= mask_tau)."""
    U = np.zeros((H, W), bool)
    for cid in child_ids:
        try:
            C = agents[cid]
        except Exception:
            continue
        m = np.asarray(getattr(C, "mask", 0.0), np.float32)
        if m.shape[:2] != (H, W):
            # If this ever hits, upstream standardization is broken.
            raise ValueError("child mask shape mismatch in child_union_for_ids")
        U |= (m >= float(mask_tau))
    return U


def proposal_overlaps_children(local_bool: np.ndarray, child_union: np.ndarray,
                               min_px: int = 6, min_frac: float = 0.10) -> tuple[bool, int]:
    L = np.asarray(local_bool, bool)
    U = np.asarray(child_union, bool)
    ov = int((L & U).sum())
    area = int(L.sum())
    return (ov >= int(min_px)) and (ov / float(max(1, area)) >= float(min_frac)), ov


def respect_spawn_spacing(cy: float, cx: float, gkids_set: set,
                          chosen_centers: list[tuple[float, float, int]],
                          exists_kids: dict[int, set],
                          H: int, W: int, per_x: bool, per_y: bool, *,
                          min_interseed_px: int, jac_thr: float, subset_thr: float) -> bool:
    """
    Enforce spatial spacing AND child-set distinctness near other recent seeds.
    """
    R2 = float(max(1, min_interseed_px)) ** 2
    for (py, px, pid_near) in (chosen_centers or []):
        if dist2_toroidal(cy, cx, py, px, H, W, per_y, per_x) >= R2:
            continue
        kids_near = exists_kids.get(pid_near, set())
        inter = len(kids_near & gkids_set)
        if inter == 0:
            continue
        j = inter / float(len(kids_near | gkids_set))
        sub = max(inter / float(len(kids_near or {1})),
                  inter / float(len(gkids_set or {1})))
        if (j >= float(jac_thr)) or (sub >= float(subset_thr)):
            return False
    return True


def norm_active_regions(regions, H: int, W: int):
    """
    Normalize active-region dicts into a consistent structure:
      {'mask': (H,W) float32 in [0,1], 'child_ids': set[int], 'proposal': (H,W) float32}
    """
    out = []
    for R in (regions or []):
        m = np.asarray(getattr(R, "mask", 0.0), np.float32)
        if m.shape[:2] != (H, W):
            try:
                m = resize_nn(m, (H, W)).astype(np.float32)
            except Exception:
                continue
        m = np.clip(m, 0.0, 1.0)
        kids = set(int(x) for x in (getattr(R, "child_ids", []) or []))
        out.append({"mask": m, "child_ids": kids, "proposal": m})
    return out


def dedup_peaks_by_childset(peaks, *, jaccard_thr: float = 0.90):
    """
    Keep peaks with unique/novel child sets (Jaccard < thr to any kept).
    """
    kept = []
    for P in (peaks or []):
        ks = set(int(x) for x in (P.get("child_ids") or []))
        novel = True
        for Q in kept:
            if jaccard_sets(ks, Q["child_ids"]) >= float(jaccard_thr):
                novel = False
                break
        if novel:
            kept.append({"mask": np.asarray(P.get("mask", 0.0), np.float32),
                         "child_ids": ks,
                         "proposal": np.asarray(P.get("proposal", P.get("mask", 0.0)), np.float32),
                         "child_weights": P.get("child_weights", None)})
    return kept


def prepare_parent_match_state(parents_by_id: dict, support_tau: float):
    """
    Build (exists_support: pid->bool mask, exists_kids: pid->set) from current parents.
    """
    exists_support = {}
    exists_kids = {}
    for pid, P in (parents_by_id or {}).items():
        m = np.asarray(getattr(P, "mask", 0.0), np.float32)
        exists_support[int(pid)] = (m > float(support_tau))
        exists_kids[int(pid)] = set(int(x) for x in (getattr(P, "child_ids", []) or []))
    return exists_support, exists_kids


def gate_and_seed_proposal(prop: np.ndarray, *, H: int, W: int,
                           local_tau: float, min_local_px: int,
                           agents, mask_tau_child: float,
                           support_tau: float, min_seed_px: int):
    """
    Gate pixelwise proposal to a boolean seed; returns:
      - (seed_bool, seed_px) on success
      - None on failure
    """
    p = np.asarray(prop, np.float32)
    if p.shape[:2] != (H, W):
        try:
            p = resize_nn(p, (H, W)).astype(np.float32)
        except Exception:
            return None

    seed_bool = (p >= float(local_tau))
    if int(seed_bool.sum()) < int(min_local_px):
        return None

    # require overlap with ≥2-child cover
    Ms = [(np.asarray(ch.mask, np.float32) >= float(mask_tau_child)).astype(np.uint8) for ch in agents]
    cover2 = (np.stack(Ms, 0).sum(0) >= 2)
    seed_px = int((seed_bool & cover2).sum())
    if seed_px < int(min_seed_px):
        return None

    return seed_bool, seed_px



def novelty_gate_ok(seed_bool: np.ndarray, exists_support: dict[int, np.ndarray], *,
                    H: int, W: int, min_novel_px: int, min_novel_frac: float):
    """
    Seed must be sufficiently **novel** w.r.t. existing parent supports.
    """
    S = int(np.asarray(seed_bool, bool).sum())
    if S < 1:
        return False, 0, 0.0
    for supp in (exists_support or {}).values():
        J = iou_bool(seed_bool, np.asarray(supp, bool))
        if J >= float(min_novel_frac):
            # if very similar IoU, also ensure some minimum non-overlapping area
            diff_px = int((np.asarray(seed_bool, bool) & ~np.asarray(supp, bool)).sum())
            if diff_px < int(min_novel_px):
                return False, diff_px, J
    diff_px = S  # novel by default
    return True, diff_px, 1.0 - 0.0


def continuity_match_pid(cy: float, cx: float, seed_bool: np.ndarray, exists_support: dict[int, np.ndarray],
                         *, H: int, W: int, per_x: bool, per_y: bool,
                         cont_r2: float, cont_iou_thr: float):
    """
    Find best continuity match by proximity (within cont_r2) and IoU threshold.
    Return (score, pid) or None.
    """
    best = None
    for pid, supp in (exists_support or {}).items():
        py, px = centroid_toroidal(supp, H, W, per_y=per_y, per_x=per_x)
        if dist2_toroidal(cy, cx, py, px, H, W, per_y, per_x) > float(cont_r2):
            continue
        iou = iou_bool(seed_bool, supp)
        if iou >= float(cont_iou_thr):
            sc = (iou, -dist2_toroidal(cy, cx, py, px, H, W, per_y, per_x))
            if best is None or sc > best[0]:
                best = (sc, int(pid))
    return best


def match_score_for_parent(seed_bool: np.ndarray, gkids: set,
                           supp: np.ndarray, kids_old: set, *,
                           close: bool, match_iou_thr: float, child_jac_thr: float):
    """
    Joint score used for parent matching: (IoU, child-set Jaccard).
    """
    iou = iou_bool(seed_bool, supp)
    if iou < float(match_iou_thr):
        return None
    j = jaccard_sets(set(gkids), set(kids_old))
    if j < float(child_jac_thr) and not close:
        return None
    return (iou, j)


# ---------------------------------------------------------------------------
# Legacy compatibility shims (minimized)
# ---------------------------------------------------------------------------

def ensure_hw(a, hw, reduce=None, dtype=np.float32):
    """Kept as a no-op to avoid breaking any lingering calls; upstream should standardize."""
    return np.asarray(a, dtype)


# ---------------------------------------------------------------------------
# Connected-components with optional periodic wrap (used by parent_semantics)
# ---------------------------------------------------------------------------
import numpy as np

def _label4(mask: np.ndarray, periodic: bool = False) -> np.ndarray:
    """
    4-connected component labeling with optional toroidal (periodic) wrap.
    Returns int32 labels in {0,1,2,...}, where 0 == background.
    """
    m = (np.asarray(mask) != 0)
    H, W = m.shape
    if H == 0 or W == 0:
        return np.zeros_like(m, dtype=np.int32)

    # Union-Find (Disjoint Set)
    N = H * W
    parent = np.arange(N, dtype=np.int32)
    rank = np.zeros(N, dtype=np.int16)

    def _idx(y, x): return y * W + x

    def _find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    # First pass: planar unions (up, left) on foreground
    for y in range(H):
        row = m[y]
        for x in range(W):
            if not row[x]:
                continue
            i = _idx(y, x)
            if x > 0 and row[x - 1]:
                _union(i, _idx(y, x - 1))
            if y > 0 and m[y - 1, x]:
                _union(i, _idx(y - 1, x))

    # Wrap unions if periodic
    if periodic:
        # left<->right
        for y in range(H):
            if m[y, 0] and m[y, W - 1]:
                _union(_idx(y, 0), _idx(y, W - 1))
        # top<->bottom
        for x in range(W):
            if m[0, x] and m[H - 1, x]:
                _union(_idx(0, x), _idx(H - 1, x))

    # Second pass: assign compact labels
    lbl = np.zeros(N, dtype=np.int32)
    k = 0
    for i in range(N):
        if not m.flat[i]:
            continue
        r = _find(i)
        if lbl[r] == 0:
            k += 1
            lbl[r] = k
        lbl[i] = lbl[r]
    return lbl.reshape(H, W)

def cc_label(mask: np.ndarray, periodic: bool = False):
    """
    Backward-compatible adapter:
    returns (labels, n_components) like scipy.ndimage.label.
    """
    labels = _label4(mask, periodic=periodic)
    n = int(labels.max()) if labels.size else 0
    return labels, n
