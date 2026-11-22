"""Local 128x128 patches, supercover ray casting and local clearance fields."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import scipy.ndimage as ndimage


@dataclass
class PatchFeatures:
    """A 128x128 local patch centred on a path point."""

    patch: np.ndarray            # (128, 128) uint8 inflated occupancy
    clearance: np.ndarray        # (128, 128) float32 EDT on the patch [cells]
    ray_directions: np.ndarray   # (K, 2) float32 unit directions
    ray_distances: np.ndarray    # (K,) float32 normalised distance in [0, 1]
    ray_hits: np.ndarray         # (K,) uint8
    ray_distances_raw: np.ndarray  # (K,) float32 distance in cells
    local_xy: Tuple[float, float]  # path point in patch coordinates (32.0-based)


PATCH_CENTER = 64.0


def extract_patch(occupancy: np.ndarray, point_xy: Tuple[int, int], patch_size: int) -> np.ndarray:
    """Crop a ``patch_size`` patch centred on ``point_xy``; out-of-map -> obstacle."""
    half = patch_size // 2
    cx, cy = point_xy
    h, w = occupancy.shape
    patch = np.ones((patch_size, patch_size), dtype=np.uint8)

    x_min = max(0, cx - half)
    x_max = min(w, cx + half)
    y_min = max(0, cy - half)
    y_max = min(h, cy + half)

    px_min = half - (cx - x_min)
    px_max = half + (x_max - cx)
    py_min = half - (cy - y_min)
    py_max = half + (y_max - cy)
    if x_max > x_min and y_max > y_min:
        patch[py_min:py_max, px_min:px_max] = occupancy[y_min:y_max, x_min:x_max]
    return patch


def patch_clearance(patch: np.ndarray) -> np.ndarray:
    """EDT of the free space inside a patch; the patch border counts as obstacle."""
    padded = np.pad(patch.astype(bool), 1, constant_values=True)
    edt = ndimage.distance_transform_edt(~padded)
    return edt[1:-1, 1:-1].astype(np.float32)


def _cast_one(occ: np.ndarray, x0: float, y0: float, ux: float, uy: float,
              max_range: float) -> Tuple[float, bool]:
    """Supercover DDA from ``(x0, y0)`` along ``(ux, uy)``.

    Returns ``(distance, hit)`` where ``distance`` is the ray parameter at which
    the ray enters the first obstacle cell (cells), or ``max_range`` when the
    ray leaves the patch / exceeds the range without hitting an obstacle.
    Cells span ``[i-0.5, i+0.5)`` so integer coordinates are cell centres.
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


def ray_directions(ray_count: int) -> np.ndarray:
    angles = 2.0 * math.pi * np.arange(ray_count) / ray_count
    return np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)


def build_patch_features(
    occupancy_inflated: np.ndarray,
    point_xy: Tuple[int, int],
    patch_size: int,
    ray_count: int,
    ray_max_range: float,
) -> PatchFeatures:
    patch = extract_patch(occupancy_inflated, point_xy, patch_size)
    clearance = patch_clearance(patch)
    dirs = ray_directions(ray_count)
    origin = (PATCH_CENTER, PATCH_CENTER)
    raw = np.zeros(ray_count, dtype=np.float32)
    hits = np.zeros(ray_count, dtype=np.uint8)
    for k in range(ray_count):
        d, hit = _cast_one(patch, origin[0], origin[1], float(dirs[k, 0]), float(dirs[k, 1]), ray_max_range)
        raw[k] = d
        hits[k] = 1 if hit else 0
    dist = np.where(hits > 0, np.minimum(raw, ray_max_range) / ray_max_range, 1.0).astype(np.float32)
    return PatchFeatures(
        patch=patch,
        clearance=clearance,
        ray_directions=dirs,
        ray_distances=dist,
        ray_hits=hits,
        ray_distances_raw=raw,
        local_xy=origin,
    )


def supercover_line_cells(p0: Tuple[float, float], p1: Tuple[float, float],
                          max_steps: int = 512) -> List[Tuple[int, int]]:
    """Cells crossed by the segment ``p0 -> p1`` (supercover DDA, both included)."""
    x0, y0 = p0
    x1, y1 = p1
    cells: List[Tuple[int, int]] = []
    ix, iy = int(round(x0)), int(round(y0))
    ex, ey = int(round(x1)), int(round(y1))
    cells.append((ix, iy))
    dx = x1 - x0
    dy = y1 - y0
    step_x = 1 if dx > 1e-12 else (-1 if dx < -1e-12 else 0)
    step_y = 1 if dy > 1e-12 else (-1 if dy < -1e-12 else 0)
    inf = float("inf")
    t_delta_x = 1.0 / abs(dx) if step_x else inf
    t_delta_y = 1.0 / abs(dy) if step_y else inf
    t_max_x = ((ix + 0.5 * step_x) - x0) / dx if step_x else inf
    t_max_y = ((iy + 0.5 * step_y) - y0) / dy if step_y else inf
    steps = 0
    while steps < max_steps:
        steps += 1
        if t_max_x < 1.0 - 1e-12 or t_max_y < 1.0 - 1e-12:
            if t_max_x < t_max_y - 1e-12:
                if t_max_x > 1.0:
                    break
                ix += step_x
                t_max_x += t_delta_x
                cells.append((ix, iy))
            elif t_max_y < t_max_x - 1e-12:
                if t_max_y > 1.0:
                    break
                iy += step_y
                t_max_y += t_delta_y
                cells.append((ix, iy))
            else:
                if t_max_x > 1.0:
                    break
                cells.append((ix + step_x, iy))
                cells.append((ix, iy + step_y))
                ix += step_x
                iy += step_y
                t_max_x += t_delta_x
                t_max_y += t_delta_y
                cells.append((ix, iy))
        else:
            break
    if cells[-1] != (ex, ey):
        cells.append((ex, ey))
    return cells
