"""Inference-time verification of a predicted safe ellipse.

The regressor outputs an ellipse; it cannot *guarantee* geometric safety (no
hard constraint is enforced during training, and the error grows exactly where
the corridor is tight).  This module is the deterministic guard that runs after
the network:

1. :func:`verify`  - exact continuous test "does the ellipse touch any inflated
   obstacle cell?"  Same routine that produced the training labels, so there is
   no train/inference mismatch.
2. :func:`project_to_safe` - if it does, scale the ellipse about the path point
   until it is clear.  Scaling both the centre offset and the shape matrix by
   ``s`` leaves ``rho = ||L^-1 (p - c)||`` unchanged, so the path point stays
   inside the ellipse and only area is traded for safety.  The result is
   collision-free by construction (bisection on ``s``, each step verified).

Typical cost: one exact test is ~0.15 ms for a 128x128 patch; the projection
needs ~12 of them (~2 ms), which is negligible next to the network forward pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .ellipse_validation import ellipse_obstacle_intersections

PATCH_CENTER = np.array([64.0, 64.0])


@dataclass
class SafeEllipse:
    L: np.ndarray            # shape matrix in cells (2x2, symmetric PD)
    center: np.ndarray       # centre in patch coordinates (cells)
    scale: float             # 1.0 = prediction used as-is, <1 = shrunk
    safe: bool               # verified collision-free
    min_quadratic: float     # min of the ellipse quadratic over obstacle cells

    @property
    def semi_axes_cells(self) -> Tuple[float, float]:
        eig = np.sort(np.linalg.eigvalsh(self.L))[::-1]
        return float(eig[0]), float(eig[1])


def verify(L: np.ndarray, center: np.ndarray, patch: np.ndarray,
           safety_epsilon: float = 1.0e-9) -> Tuple[bool, float]:
    """Exact check: ``(collision_free, min_quadratic)``.

    ``min_quadratic > 1`` means the ellipse does not intersect any inflated
    obstacle cell in the patch (continuous test, not point sampling).
    """
    collides, min_q, _ = ellipse_obstacle_intersections(np.asarray(L, float),
                                                        np.asarray(center, float),
                                                        patch, safety_epsilon)
    return (not collides), float(min_q)


def project_to_safe(L: np.ndarray, center: np.ndarray, patch: np.ndarray,
                    iters: int = 12, safety_epsilon: float = 1.0e-9) -> SafeEllipse:
    """Shrink the ellipse about the patch centre until it is collision-free."""
    L = np.asarray(L, dtype=float)
    center = np.asarray(center, dtype=float)
    ok, min_q = verify(L, center, patch, safety_epsilon)
    if ok:
        return SafeEllipse(L=L, center=center, scale=1.0, safe=True, min_quadratic=min_q)

    lo, hi = 0.0, 1.0                        # lo safe (degenerate), hi unsafe
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if verify(L * mid, PATCH_CENTER + mid * (center - PATCH_CENTER), patch,
                  safety_epsilon)[0]:
            lo = mid
        else:
            hi = mid
    L_s = L * lo
    c_s = PATCH_CENTER + lo * (center - PATCH_CENTER)
    ok_s, min_q_s = verify(L_s, c_s, patch, safety_epsilon)
    return SafeEllipse(L=L_s, center=c_s, scale=float(lo), safe=bool(ok_s),
                       min_quadratic=min_q_s)


def from_parameters(a_m: float, b_m: float, angle_rad: float, center_offset_m: Tuple[float, float],
                    resolution: float) -> Tuple[np.ndarray, np.ndarray]:
    """Build ``(L, center_cells)`` from the network's physical output."""
    ca, sa = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[ca, -sa], [sa, ca]])
    L = R @ np.diag([a_m / resolution, b_m / resolution]) @ R.T
    center = PATCH_CENTER + np.asarray(center_offset_m, dtype=float) / resolution
    return L, center
