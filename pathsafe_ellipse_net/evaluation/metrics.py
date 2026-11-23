"""Evaluation metrics: centre, ellipse shape, ray constraint, IoU."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _dist_stats(values: np.ndarray) -> Dict[str, Optional[float]]:
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"mean": None, "median": None, "p95": None}
    return {"mean": float(v.mean()), "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95))}


def ellipse_iou_grid(q0, S0, q1, S1, n_grid: int = 96) -> float:
    """Approximate IoU of two ellipses by rasterising inside the union bbox."""
    def extent(q, S):
        rx, ry = np.sqrt(max(S[0, 0], 0.0)), np.sqrt(max(S[1, 1], 0.0))
        return q[0] - rx, q[0] + rx, q[1] - ry, q[1] + ry

    e0, e1 = extent(q0, S0), extent(q1, S1)
    x0, x1 = min(e0[0], e1[0]), max(e0[1], e1[1])
    y0, y1 = min(e0[2], e1[2]), max(e0[3], e1[3])
    if not np.isfinite([x0, x1, y0, y1]).all() or x1 - x0 <= 1e-9 or y1 - y0 <= 1e-9:
        return 0.0
    xs = np.linspace(x0, x1, n_grid)
    ys = np.linspace(y0, y1, n_grid)
    gx, gy = np.meshgrid(xs, ys)

    def mask(q, S):
        M = np.linalg.inv(np.asarray(S, np.float64))
        dx, dy = gx - q[0], gy - q[1]
        return (M[0, 0] * dx * dx + 2 * M[0, 1] * dx * dy + M[1, 1] * dy * dy) <= 1.0

    m0, m1 = mask(q0, S0), mask(q1, S1)
    union = np.logical_or(m0, m1).sum()
    return float(np.logical_and(m0, m1).sum() / union) if union else 0.0


def angle_error(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Axis orientation error in radians, folded to ``[0, pi/2]``."""
    d = np.mod(np.abs(np.asarray(pred, np.float64) - np.asarray(gt, np.float64)), np.pi)
    return np.minimum(d, np.pi - d)


def summarize_predictions(center: np.ndarray, axes: np.ndarray, angle: np.ndarray,
                          S: np.ndarray, gt_center: np.ndarray, gt_axes: np.ndarray,
                          gt_angle: np.ndarray, gt_S: np.ndarray, valid: np.ndarray,
                          path_xy: Optional[np.ndarray] = None,
                          ray_onehot: Optional[np.ndarray] = None,
                          ratio: Optional[np.ndarray] = None,
                          lidar: Optional[np.ndarray] = None,
                          r_max_m: float = 3.0, eps_r_m: float = 0.05,
                          iou_limit: int = 400) -> Dict[str, object]:
    """Centre / shape metrics over the valid points of a concatenated prediction set."""
    v = np.asarray(valid, dtype=bool)
    if v.sum() == 0:
        return {"n": 0}
    center = np.asarray(center, np.float64)
    gt_center = np.asarray(gt_center, np.float64)
    pred_ab = np.asarray(axes, np.float64)[v]
    gt_ab = np.asarray(gt_axes, np.float64)[v]

    ce = np.linalg.norm(center[v] - gt_center[v], axis=1)
    ang_err = angle_error(np.asarray(angle)[v], np.asarray(gt_angle)[v])
    idx_v = np.flatnonzero(v)[:iou_limit]
    ious = np.array([ellipse_iou_grid(center[i], np.asarray(S)[i],
                                      gt_center[i], np.asarray(gt_S)[i]) for i in idx_v])
    area_ratio = (pred_ab[:, 0] * pred_ab[:, 1]) / np.maximum(gt_ab[:, 0] * gt_ab[:, 1], 1e-9)

    out: Dict[str, object] = {
        "n": int(v.sum()),
        "center_error": _dist_stats(ce),
        "axes_error_a": _dist_stats(np.abs(pred_ab[:, 0] - gt_ab[:, 0])),
        "axes_error_b": _dist_stats(np.abs(pred_ab[:, 1] - gt_ab[:, 1])),
        "angle_error_deg": {"mean": float(np.degrees(ang_err.mean())),
                            "median": float(np.degrees(np.median(ang_err)))},
        "area_ratio": {"mean": float(area_ratio.mean()),
                       "median": float(np.median(area_ratio))},
        "ellipse_iou": {"mean": float(ious.mean()) if ious.size else None,
                        "median": float(np.median(ious)) if ious.size else None,
                        "n": int(ious.size)},
    }
    if ratio is not None:
        r = np.asarray(ratio, np.float64)[v]
        out["ratio"] = {"mean": float(r.mean()), "p05": float(np.percentile(r, 5)),
                        "p95": float(np.percentile(r, 95))}
    if ray_onehot is not None and path_xy is not None:
        onehot = np.asarray(ray_onehot, np.float64)
        rays = onehot.shape[-1]
        idx = onehot[v].argmax(-1)
        out["direction_hist"] = np.bincount(idx, minlength=rays).tolist()
        u = np.stack([np.cos(2 * np.pi * idx / rays), np.sin(2 * np.pi * idx / rays)], 1)
        off = center[v] - np.asarray(path_xy, np.float64)[v]
        step = (off * u).sum(1)
        lateral = np.linalg.norm(off - step[:, None] * u, axis=1)
        out["ray_constraint"] = {"lateral_m": _dist_stats(lateral),
                                 "step_m": _dist_stats(step)}
        if lidar is not None:
            usable = np.clip(np.asarray(lidar, np.float64)[v][np.arange(idx.size), idx]
                             - eps_r_m, 0.0, r_max_m)
            out["ray_constraint"]["over_range_rate"] = float(
                np.mean(step > usable + 1e-6))
    return out
