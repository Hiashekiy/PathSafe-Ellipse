"""Path geometry features computed from the input path only (plan section 2.2).

    g_i = [t_x, t_y, kappa_tilde, s_tilde, ds_tilde, x_norm, y_norm]

Nothing here may look at ``center_offset``/``q_xy``/labels: these are network
inputs, and using label information would be leakage.
"""
from __future__ import annotations

import numpy as np


def path_geometry(path_xy: np.ndarray, map_extent_xy: tuple) -> np.ndarray:
    """``(N,7)`` float32 features of a metric path inside a metric map extent."""
    p = np.asarray(path_xy, dtype=np.float64).reshape(-1, 2)
    n = len(p)
    if n == 0:
        return np.zeros((0, 7), np.float32)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1) if n > 1 else np.zeros(0)
    mean_seg = float(np.mean(seg)) if seg.size else 1.0
    mean_seg = max(mean_seg, 1e-6)

    tangent = np.zeros((n, 2), np.float64)
    if n == 1:
        tangent[0] = (1.0, 0.0)
    else:
        tangent[1:-1] = p[2:] - p[:-2]
        tangent[0] = p[1] - p[0]
        tangent[-1] = p[-1] - p[-2]
        norm = np.linalg.norm(tangent, axis=1, keepdims=True)
        tangent = tangent / np.maximum(norm, 1e-9)

    # turning angle between consecutive tangents -> curvature per metre
    kappa = np.zeros(n)
    if n > 2:
        dtheta = np.arctan2(tangent[1:, 1], tangent[1:, 0]) - np.arctan2(
            tangent[:-1, 1], tangent[:-1, 0])
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
        kappa[1:] = np.abs(dtheta) / np.maximum(seg, 1e-9)
    kappa_tilde = np.tanh(kappa * mean_seg)          # bounded, scale free

    s_cum = np.concatenate([[0.0], np.cumsum(seg)]) if n > 1 else np.zeros(1)
    total = float(s_cum[-1]) if s_cum.size else 0.0
    s_tilde = s_cum / total if total > 1e-9 else np.zeros(n)

    ds = np.zeros(n)
    if n > 1:
        ds[1:] = seg / mean_seg
        ds[0] = ds[1]
    ds_tilde = np.clip(ds, 0.0, 5.0)

    W, H = float(map_extent_xy[0]), float(map_extent_xy[1])
    x_norm = p[:, 0] / max(W, 1e-9)
    y_norm = p[:, 1] / max(H, 1e-9)

    return np.stack([tangent[:, 0], tangent[:, 1], kappa_tilde, s_tilde, ds_tilde,
                     x_norm, y_norm], axis=1).astype(np.float32)


GEOM_DIM = 7
