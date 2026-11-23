"""Exact (NumPy) safety checks on predictions — never replaced by the SDF loss."""
from __future__ import annotations

import os
import sys
from typing import Dict

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from pathsafe_ellipse.v2.ellipse import exact_clearance  # noqa: E402


def exact_safety_stats(center: np.ndarray, S: np.ndarray, occupancy: np.ndarray,
                       r_scene: float, valid: np.ndarray) -> Dict[str, float]:
    """Collision rate of the *raw* predictions under the generator's exact test."""
    v = np.asarray(valid, dtype=bool)
    if v.sum() == 0:
        return {"n": 0}
    occ = np.asarray(occupancy).astype(np.uint8)
    H, W = occ.shape
    n_safe = 0
    min_vals = []
    center_blocked = 0
    for i in np.flatnonzero(v):
        q = np.asarray(center[i], dtype=np.float64)
        ok, min_q, _bad, _idx = exact_clearance(np.asarray(S[i], dtype=np.float64), q,
                                               occ, r_scene)
        n_safe += int(bool(ok))
        if np.isfinite(min_q):
            min_vals.append(min_q)
        cc = int(round(q[0] / r_scene))
        rr = int(round(q[1] / r_scene))
        if 0 <= rr < H and 0 <= cc < W and occ[rr, cc] > 0:
            center_blocked += 1
    n = int(v.sum())
    return {
        "n": n,
        "exact_safety_rate": n_safe / n,
        "raw_collision_rate": 1.0 - n_safe / n,
        "center_collision_rate": center_blocked / n,
        "min_quadratic": {"mean": float(np.mean(min_vals)) if min_vals else None,
                          "p05": float(np.percentile(min_vals, 5)) if min_vals else None},
    }


def border_violation_rate(center: np.ndarray, S: np.ndarray, extent, valid: np.ndarray,
                          margin: float = 0.0) -> float:
    v = np.asarray(valid, dtype=bool)
    if v.sum() == 0:
        return 0.0
    c = np.asarray(center)[v]
    S = np.asarray(S)[v]
    rx = np.sqrt(np.maximum(S[:, 0, 0], 0.0))
    ry = np.sqrt(np.maximum(S[:, 1, 1], 0.0))
    W, H = extent
    bad = ((c[:, 0] - rx - margin < 0) | (c[:, 0] + rx + margin > W)
           | (c[:, 1] - ry - margin < 0) | (c[:, 1] + ry + margin > H))
    return float(bad.mean())


def continuity_pass_rate(center: np.ndarray, S: np.ndarray, valid: np.ndarray,
                         rho_max: float = 0.85) -> Dict[str, float]:
    """Fraction of adjacent pairs whose midpoints are inside both ellipses."""
    v = np.asarray(valid, dtype=bool)
    n = len(center)
    if n < 2:
        return {"pair_rate": 0.0, "path_rate": 0.0}
    ok_pairs = []
    for i in range(n - 1):
        if not (v[i] and v[i + 1]):
            continue
        m = 0.5 * (np.asarray(center[i]) + np.asarray(center[i + 1]))
        r0 = midpoint_metric_np(S[i], center[i], m)
        r1 = midpoint_metric_np(S[i + 1], center[i + 1], m)
        ok_pairs.append(r0 <= rho_max and r1 <= rho_max)
    if not ok_pairs:
        return {"pair_rate": 0.0, "path_rate": 0.0}
    return {"pair_rate": float(np.mean(ok_pairs)),
            "path_rate": float(all(ok_pairs))}


def midpoint_metric_np(S: np.ndarray, q: np.ndarray, m: np.ndarray) -> float:
    d = np.asarray(m, np.float64) - np.asarray(q, np.float64)
    return float(np.sqrt(max(d @ np.linalg.inv(np.asarray(S, np.float64)) @ d, 0.0)))
