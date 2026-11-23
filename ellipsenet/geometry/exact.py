"""Exact (analytic) ellipse-vs-occupied-square test, vendored into ``ellipsenet``.

Originally part of the dataset generator (``src/pathsafe_ellipse/v2/ellipse.py`` and
``src/pathsafe_ellipse/local_features.py``); copied here verbatim so that ``ellipsenet``
and its acceptance scripts have no dependency on the retired V1/V2 code base.

The test is the *acceptance authority* for "does the predicted ellipse touch a blocked
cell": the SDF-sampled safety term is a differentiable proxy, this one is exact.

    box_quadratic_min       exact min of (x-q)^T M (x-q) over axis-aligned boxes
    exact_clearance         (is_clear, min_quadratic, violating cells, indices)
    supercover_ray_cast     the DDA used by the dataset's LiDAR model (audit only)
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def box_quadratic_min(M: np.ndarray, q: np.ndarray, x0, x1, y0, y1) -> np.ndarray:
    """Exact min of ``(x-q)^T M (x-q)`` over axis-aligned boxes (vectorised)."""
    m11, m12, m22 = M[0, 0], M[0, 1], M[1, 1]
    cx, cy = q[0], q[1]
    best = np.full(np.broadcast(x0, x1, y0, y1).shape, np.inf)

    def evaluate(x, y):
        dx = x - cx
        dy = y - cy
        return m11 * dx * dx + 2.0 * m12 * dx * dy + m22 * dy * dy

    inside = (cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1)
    best = np.where(inside, 0.0, best)
    for xf in (x0, x1):
        y_star = cy - (m12 * (xf - cx)) / m22 if m22 > 0 else cy
        best = np.minimum(best, evaluate(xf, np.clip(y_star, y0, y1)))
        best = np.minimum(best, evaluate(xf, y0))
        best = np.minimum(best, evaluate(xf, y1))
    for yf in (y0, y1):
        x_star = cx - (m12 * (yf - cy)) / m11 if m11 > 0 else cx
        best = np.minimum(best, evaluate(np.clip(x_star, x0, x1), yf))
        best = np.minimum(best, evaluate(x0, yf))
        best = np.minimum(best, evaluate(x1, yf))
    return best


def ellipse_bbox_cells(occupancy: np.ndarray, r_scene: float, q: np.ndarray,
                       radii: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Occupied cell centres (metres) inside the ellipse bounding box."""
    n = occupancy.shape[0]
    c0 = int(np.floor((q[0] - radii[0]) / r_scene)) - 1
    c1 = int(np.ceil((q[0] + radii[0]) / r_scene)) + 1
    r0 = int(np.floor((q[1] - radii[1]) / r_scene)) - 1
    r1 = int(np.ceil((q[1] + radii[1]) / r_scene)) + 1
    c0 = max(c0, 0); r0 = max(r0, 0)
    c1 = min(c1, n - 1); r1 = min(r1, n - 1)
    if c1 < c0 or r1 < r0:
        return np.empty((0, 2)), np.empty(0, dtype=int)
    sub = occupancy[r0:r1 + 1, c0:c1 + 1]
    rr, cc = np.nonzero(sub)
    if rr.size == 0:
        return np.empty((0, 2)), np.empty(0, dtype=int)
    centres = np.column_stack([(cc + c0) * r_scene, (rr + r0) * r_scene])
    flat_idx = (rr + r0) * n + (cc + c0)
    return centres, flat_idx


def exact_clearance(S: np.ndarray, q: np.ndarray, occupancy: np.ndarray,
                    r_scene: float, eps: float = 0.0
                    ) -> Tuple[bool, float, np.ndarray, np.ndarray]:
    """Exact test: ``(is_clear, min_quadratic, violating_centres, violating_idx)``.

    ``min_quadratic > 1`` means the ellipse does not intersect any occupied square
    inside its bounding box (continuous analytic test, not sampling).
    """
    S = (np.asarray(S, float) + np.asarray(S, float).T) / 2.0
    eig = np.linalg.eigvalsh(S)
    if eig.min() <= 0:
        return False, -np.inf, np.empty((0, 2)), np.empty(0, dtype=int)
    radii = np.sqrt(np.maximum(np.diag(S), 0.0))   # exact bounding half-extents
    centres, idx = ellipse_bbox_cells(occupancy, r_scene, q, radii)
    if centres.size == 0:
        return True, np.inf, np.empty((0, 2)), np.empty(0, dtype=int)
    M = np.linalg.inv(S)
    h = 0.5 * r_scene
    vals = box_quadratic_min(M, q, centres[:, 0] - h, centres[:, 0] + h,
                             centres[:, 1] - h, centres[:, 1] + h)
    min_val = float(vals[int(np.argmin(vals))])
    bad = vals <= (1.0 + eps)
    return (not np.any(bad)), min_val, centres[bad], idx[bad]


def supercover_ray_cast(occ: np.ndarray, x0: float, y0: float, ux: float, uy: float,
                        max_range: float) -> Tuple[float, bool]:
    """Supercover DDA used by the dataset's LiDAR model (audit/diagnostic only).

    Returns ``(distance, hit)`` where ``distance`` is the ray parameter at which the
    ray enters the first obstacle cell (cells), or ``max_range`` when the ray leaves
    the map / exceeds the range without hitting.  Cells span ``[i-0.5, i+0.5)``, so
    integer coordinates are cell centres -- this is why the stored LiDAR range is the
    distance to the blocked cell *boundary*, not to its centre.
    """
    h, w = occ.shape
    ix, iy = int(round(x0)), int(round(y0))
    if ix < 0 or iy < 0 or ix >= w or iy >= h:
        return max_range, False
    if occ[iy, ix]:
        return 0.0, True

    step_x = 1 if ux > 1e-12 else (-1 if ux < -1e-12 else 0)
    step_y = 1 if uy > 1e-12 else (-1 if uy < -1e-12 else 0)
    inf = float("inf")
    t_delta_x = 1.0 / abs(ux) if step_x else inf
    t_delta_y = 1.0 / abs(uy) if step_y else inf
    t_max_x = ((ix + 0.5 * step_x) - x0) / ux if step_x else inf
    t_max_y = ((iy + 0.5 * step_y) - y0) / uy if step_y else inf
    t = 0.0

    while True:
        if t_max_x < t_max_y - 1e-12:
            t = t_max_x
            ix += step_x
            t_max_x += t_delta_x
            if t > max_range or ix < 0 or ix >= w:
                return max_range, False
            if occ[iy, ix]:
                return t, True
        elif t_max_y < t_max_x - 1e-12:
            t = t_max_y
            iy += step_y
            t_max_y += t_delta_y
            if t > max_range or iy < 0 or iy >= h:
                return max_range, False
            if occ[iy, ix]:
                return t, True
        else:  # exact corner crossing: both orthogonal cells must be checked
            t = t_max_x
            if t > max_range:
                return max_range, False
            side_a = (0 <= ix + step_x < w) and occ[iy, ix + step_x]
            side_b = (0 <= iy + step_y < h) and occ[iy + step_y, ix]
            if side_a or side_b:
                return t, True
            ix += step_x
            iy += step_y
            t_max_x += t_delta_x
            t_max_y += t_delta_y
            if ix < 0 or ix >= w or iy < 0 or iy >= h:
                return max_range, False
            if occ[iy, ix]:
                return t, True
