# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 12:00:40 2025

@author: chris and christine
"""


from typing import List, Tuple, Optional
import numpy as np
from core.gaussian_core import resolve_support_tau
import core.config as config



def _resize_to_hw(a, H, W):
    a = np.asarray(a, np.float32)
    if a.shape == (H, W): 
        return a
    if a.shape[0] >= H and a.shape[1] >= W:
        return a[:H, :W].astype(np.float32, copy=False)
    py, px = max(0, H - a.shape[0]), max(0, W - a.shape[1])
    return np.pad(a, ((0, py), (0, px)), mode="edge")[:H, :W].astype(np.float32, copy=False)





def region_confidence(
    mask_bool, *,
    A_global=None,              # (H,W) agreement in [0,1] or None
    cover=None,                 # (H,W) #active kids, int
    mean_weight=None,           # (H,W) mean child mask weight in [0,1]
    prior_p=0.10,               # prior probability a random pixel is "parentish"
    eps=1e-6
):
    """
    Compute a probabilistic confidence for a candidate region.
    Combines: mean agreement, spatial coherence, and a simple LL ratio vs. null.
    Returns dict: { 'conf': float in [0,1], 'parts': {...} } for diagnostics.
    """
    m = np.asarray(mask_bool, bool)
    area = int(m.sum())
    if area == 0:
        return {'conf': 0.0, 'parts': {}}

    # --- 1) agreement summary (posterior for "signal" pixels) ---
    if A_global is not None and getattr(A_global, "shape", None) == mask_bool.shape:
        A_vals = np.asarray(A_global, np.float32)[m]
        A_mu   = float(np.nanmean(A_vals)) if A_vals.size else 0.0
        A_std  = float(np.nanstd(A_vals)) if A_vals.size else 1.0
    else:
        A_mu, A_std = 0.0, 1.0

    # --- 2) spatial coherence proxy: boundary ratio (lower is more coherent) ---
    # 4-neigh boundary pixels inside the mask
    rollN = np.roll(m,  1, axis=0)
    rollS = np.roll(m, -1, axis=0)
    rollW = np.roll(m,  1, axis=1)
    rollE = np.roll(m, -1, axis=1)
    neighbors_in = (rollN + rollS + rollW + rollE)
    boundary = m & (neighbors_in < 4)
    boundary_ratio = float(boundary.sum()) / float(area + eps)

    # --- 3) crude likelihood ratio vs. null using agreement as "success prob" ---
    # Treat A as pixelwise Bernoulli success prob; region uses mean A as parameter.
    # Compare P(data|region) with P(data|null=prior_p), using mean agreement as proxy.
    p1 = max(eps, min(1.0 - eps, A_mu))
    p0 = max(eps, min(1.0 - eps, float(prior_p)))
    # Approx Binomial LL with sufficient stats: sum(A) ~ area*A_mu (proxy)
    S = max(0.0, A_mu * area)
    F = max(0.0, (1.0 - A_mu) * area)
    ll1 = S * np.log(p1) + F * np.log(1.0 - p1)
    ll0 = S * np.log(p0) + F * np.log(1.0 - p0)
    llr = float(ll1 - ll0)

    # --- fuse into confidence in [0,1] via a calibrated sigmoid ---
    # penalize ragged regions (higher boundary_ratio) and low agreement
    z = (llr / max(1.0, area)) + 1.5*(A_mu - 0.5) - 1.0*boundary_ratio
    conf = 1.0 / (1.0 + np.exp(-3.0 * z))  # temperature 3.0; adjust in config later

    return {
        'conf': float(np.clip(conf, 0.0, 1.0)),
        'parts': dict(A_mu=A_mu, A_std=A_std, boundary_ratio=boundary_ratio, llr=llr, area=area)
    }













def compute_cover_weight(children, H, W, mask_tau_mask, config):
    """
    Returns:
      cover  : (H,W) int32  (# active kids at pixel, mask>=tau)
      weight : (H,W) float32 (sum of mask weights over active kids)
      kid_act: list[(H,W) bool] for each child (mask>=tau)
    """
    cover  = np.zeros((H, W), np.int32)
    weight = np.zeros((H, W), np.float32)
    kid_act = []
    for ch in children:
        m = np.asarray(getattr(ch, "mask", 0.0), np.float32)
        a = (m >= float(mask_tau_mask))
        cover  += a.astype(np.int32)
        weight += m * a.astype(np.float32)
        kid_act.append(a)
    return cover, weight, kid_act

def aggregate_child_agreement(children, kid_act, H, W, alpha, tau_q, tau_p, tau_blend):
    """
    For each child, return an agreement map A_child in [0,1] using caches only:
      1) prefer ch.spawn_A_final
      2) else from softmin KL caches -> exp(-KL/τ) (using τ_q/τ_p and α blend)
      3) else from cached_alignment_kl / cached_model_alignment_kl
    Returns: list[A_child or None] aligned with children order.
    """
    A_list = []
    for ch, a in zip(children, kid_act):
        # 1) direct agreement cache
        A = getattr(ch, "spawn_A_final", None)
        if A is not None:
            A = _resize_to_hw(A, H, W)
            A = np.clip(np.nan_to_num(A, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
            A_list.append(A)
            continue

        # 2) softmin KL caches
        q = getattr(ch, "spawn_softmin_kl_q", None)
        p = getattr(ch, "spawn_softmin_kl_p", None)
        if (q is not None) or (p is not None):
            if q is not None:
                q = _resize_to_hw(q, H, W)
                Aq = np.exp(-np.asarray(q, np.float32) / max(tau_q, 1e-8))
            else:
                Aq = None
            if p is not None:
                p = _resize_to_hw(p, H, W)
                Ap = np.exp(-np.asarray(p, np.float32) / max(tau_p, 1e-8))
            else:
                Ap = None

            if (Aq is not None) and (Ap is not None):
                A = np.clip((alpha * Aq + (1.0 - alpha) * Ap).astype(np.float32), 0.0, 1.0)
            elif Aq is not None:
                A = np.clip(Aq.astype(np.float32), 0.0, 1.0)
            elif Ap is not None:
                A = np.clip(Ap.astype(np.float32), 0.0, 1.0)
            else:
                A = None

            A_list.append(A)
            continue

        # 3) light fallback KL caches
        q = getattr(ch, "cached_alignment_kl", None)
        p = getattr(ch, "cached_model_alignment_kl", None)
        if (q is None) and (p is None):
            A_list.append(None)
            continue

        if q is not None:
            q = _resize_to_hw(q, H, W)
            Aq = np.exp(-np.asarray(q, np.float32) / max(tau_q, 1e-8))
        else:
            Aq = None
        if p is not None:
            p = _resize_to_hw(p, H, W)
            Ap = np.exp(-np.asarray(p, np.float32) / max(tau_p, 1e-8))
        else:
            Ap = None

        if (Aq is not None) and (Ap is not None):
            A = np.clip((alpha * Aq + (1.0 - alpha) * Ap).astype(np.float32), 0.0, 1.0)
        elif Aq is not None:
            A = np.clip(Aq.astype(np.float32), 0.0, 1.0)
        else:
            A = np.clip(Ap.astype(np.float32), 0.0, 1.0)
        A_list.append(A)
    return A_list

def apply_weight_gate(cover, weight, min_kids, cfg_weight_tau):
    """
    Returns:
      keep        : (H,W) bool mask of pixels kept after cover & weight gates
      mean_weight : (H,W) float32 mean weight per pixel
      thr         : float weight threshold used (auto or fixed)
    """
    H, W = cover.shape
    keep = (cover >= int(min_kids))
    mean_weight = np.zeros((H, W), np.float32)
    nz = cover > 0
    mean_weight[nz] = (weight[nz] / cover[nz].astype(np.float32))
    if cfg_weight_tau is None:
        ww = mean_weight[(cover >= int(min_kids)) & (mean_weight > 0)]
        thr = float(np.percentile(ww, 60)) if ww.size else 0.0
    else:
        thr = float(cfg_weight_tau)
    keep &= (mean_weight >= thr)
    return keep, mean_weight, thr

def erode_core_bool(core, iters, periodic):
    """4-neighborhood erosion; PBC-safe if periodic=True."""
    core = np.asarray(core, bool, order="C")
    if iters <= 0 or not core.any():
        return core
    if periodic:
        for _ in range(iters):
            e = core.copy()
            e &= np.roll(core,  1, axis=0)  # N
            e &= np.roll(core, -1, axis=0)  # S
            e &= np.roll(core,  1, axis=1)  # W
            e &= np.roll(core, -1, axis=1)  # E
            core = e
    else:
        for _ in range(iters):
            north = np.pad(core[1:, :],  ((0,1),(0,0)), constant_values=False)
            south = np.pad(core[:-1, :], ((1,0),(0,0)), constant_values=False)
            west  = np.pad(core[:, 1:],  ((0,0),(0,1)), constant_values=False)
            east  = np.pad(core[:, :-1], ((0,0),(1,0)), constant_values=False)
            core = core & north & south & west & east
    return core



