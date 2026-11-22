"""Exact geometric acceptance tests for a candidate ellipse (sections 8.4 / 8.5).

* ``point_inside_metric``: rho = ||L^{-1}(p - c)||_2 must satisfy rho <= eta.
* ``ellipse_cell_intersections``: exact minimum of the ellipse quadratic form
  over every obstacle cell square inside the ellipse bounding box.  The
  minimum of a convex quadratic over an axis-aligned box is attained on the
  boundary, so it is enough to minimise analytically on each of the four faces
  and take the smallest value.  A value ``<= 1`` means the cell and the ellipse
  intersect, i.e. the ellipse is unsafe.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class SafetyReport:
    safe: bool
    min_quad: float
    n_obstacle_cells: int
    point_metric: float
    degenerate: bool
    n_axes_ok: bool


def quadratic_matrix(L: np.ndarray) -> np.ndarray:
    """``M = (L L^T)^{-1}`` of the ellipse quadratic form ``(x-c)^T M (x-c) <= 1``."""
    L = np.asarray(L, dtype=float)
    return np.linalg.inv(L @ L.T)


def point_metric(L: np.ndarray, c: np.ndarray, p: np.ndarray) -> float:
    """Normalised distance rho of ``p`` from the ellipse centre."""
    try:
        L_inv = np.linalg.inv(np.asarray(L, dtype=float))
    except np.linalg.LinAlgError:
        return float("inf")
    return float(np.linalg.norm(L_inv @ (np.asarray(p, dtype=float) - np.asarray(c, dtype=float))))


def box_quadratic_min(M: np.ndarray, c: np.ndarray, x0, x1, y0, y1):
    """Minimum of ``(x-c)^T M (x-c)`` over the axis-aligned box (vectorised).

    ``x0, x1, y0, y1`` are broadcastable arrays.  Returns the exact minimum
    because a convex quadratic attains its box minimum on a face and the 1-D
    face minimum is analytic.
    """
    m11, m12, m22 = M[0, 0], M[0, 1], M[1, 1]
    cx, cy = c[0], c[1]
    best = np.full(np.broadcast(x0, x1, y0, y1).shape, np.inf)

    def evaluate(x, y):
        dx = x - cx
        dy = y - cy
        return m11 * dx * dx + 2.0 * m12 * dx * dy + m22 * dy * dy

    # the unconstrained minimiser (the ellipse centre) may sit inside the box
    inside = (cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1)
    best = np.where(inside, 0.0, best)

    # faces x = x0 and x = x1 : minimise over y analytically
    for xf in (x0, x1):
        y_star = cy - (m12 * (xf - cx)) / m22 if m22 > 0 else cy
        y_c = np.clip(y_star, y0, y1)
        best = np.minimum(best, evaluate(xf, y_c))
        best = np.minimum(best, evaluate(xf, y0))
        best = np.minimum(best, evaluate(xf, y1))
    # faces y = y0 and y = y1 : minimise over x analytically
    for yf in (y0, y1):
        x_star = cx - (m12 * (yf - cy)) / m11 if m11 > 0 else cx
        x_c = np.clip(x_star, x0, x1)
        best = np.minimum(best, evaluate(x_c, yf))
        best = np.minimum(best, evaluate(x0, yf))
        best = np.minimum(best, evaluate(x1, yf))
    return best


def ellipse_obstacle_intersections(
    L: np.ndarray,
    c: np.ndarray,
    patch: np.ndarray,
    safety_epsilon: float = 1.0e-9,
) -> Tuple[bool, float, int]:
    """Exact (continuous) check: does the ellipse touch any obstacle cell square?

    Returns ``(has_intersection, min_quad, n_cells_checked)``.
    """
    patch = np.asarray(patch)
    h, w = patch.shape
    # semi-axis bounds of the ellipse -> axis-aligned bounding box in cells
    radii = np.sqrt(np.maximum(np.sum(L * L, axis=1), 0.0))
    x_lo = int(np.floor(c[0] - radii[0] - 0.5))
    x_hi = int(np.ceil(c[0] + radii[0] + 0.5))
    y_lo = int(np.floor(c[1] - radii[1] - 0.5))
    y_hi = int(np.ceil(c[1] + radii[1] + 0.5))
    x_lo = max(x_lo, 0); y_lo = max(y_lo, 0)
    x_hi = min(x_hi, w - 1); y_hi = min(y_hi, h - 1)
    if x_hi < x_lo or y_hi < y_lo:
        return False, np.inf, 0

    sub = patch[y_lo:y_hi + 1, x_lo:x_hi + 1]
    ys, xs = np.nonzero(sub)
    if xs.size == 0:
        return False, np.inf, 0

    cells_x = xs + x_lo
    cells_y = ys + y_lo
    M = quadratic_matrix(L)
    min_q = box_quadratic_min(
        M, np.asarray(c, dtype=float),
        cells_x - 0.5, cells_x + 0.5, cells_y - 0.5, cells_y + 0.5,
    )
    value = float(np.min(min_q))
    return value <= (1.0 + safety_epsilon), value, int(xs.size)


def evaluate_ellipse(
    L: np.ndarray,
    c: np.ndarray,
    patch: np.ndarray,
    path_point: np.ndarray,
    eta: float,
    min_axis_cells: float,
    max_axis_cells: float,
    safety_epsilon: float = 1.0e-9,
) -> Tuple[SafetyReport, List[str]]:
    """Full acceptance test used by candidate selection and sample filtering.

    Size validity is the single fixed rule of the implementation plan: the short
    semi-axis must exceed ``min_axis_cells`` and the long one stay below
    ``max_axis_cells`` (both in cells).  No scale-adaptive thresholds.
    """
    reasons: List[str] = []
    L = np.asarray(L, dtype=float)
    L = (L + L.T) / 2.0
    degenerate = False
    if not np.all(np.isfinite(L)):
        reasons.append("ellipse_nonfinite")
        degenerate = True
        eig = np.array([-1.0, -1.0])
    else:
        eig = np.linalg.eigvalsh(L)
        if eig.min() <= 0:
            reasons.append("ellipse_not_positive_definite")
            degenerate = True

    axes_ok = bool(eig.min() >= min_axis_cells and eig.max() <= max_axis_cells
                   and eig.max() >= eig.min())
    if not axes_ok and not degenerate:
        reasons.append("ellipse_degenerate")

    rho = point_metric(L, c, path_point) if not degenerate else float("inf")
    if rho > eta:
        reasons.append("path_point_outside_ellipse")

    if degenerate:
        return SafetyReport(False, float("inf"), 0, rho, True, axes_ok), reasons

    collides, min_q, n_cells = ellipse_obstacle_intersections(L, c, patch, safety_epsilon)
    if collides:
        reasons.append("ellipse_collision")
    return SafetyReport(not collides, min_q, n_cells, rho, False, axes_ok), reasons
