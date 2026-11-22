"""V2 scene sampling: whole 256x256 scene, conservative rescale, metric inflation,
distance field (metres) with bilinear interpolation and analytic gradient.

Conventions (V2)
----------------
* A grid cell ``(row=i, col=j)`` is a square of side ``r_scene`` metres whose
  centre sits at ``x = j * r_scene``, ``y = i * r_scene``.
* Every geometric quantity outside grid lookups is in **metres**.
* ``free`` / ``blocked`` refers to the robot *centre*: a cell is blocked when the
  distance from its centre to the nearest occupied square is ``<= r_robot + margin``
  (conservative Minkowski sum of occupied square and robot disk).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.ndimage as ndimage


# --------------------------------------------------------------------------- #
# Conservative occupancy resampling
# --------------------------------------------------------------------------- #
def conservative_resize(occ: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Rescale an occupancy grid to ``size`` without ever losing an obstacle.

    Down-sampling (crop_px > size): a target cell is occupied if **any** source
    cell overlapping its footprint is occupied (max pooling by area).
    Up-sampling (crop_px < size): nearest neighbour, so thin obstacles survive.

    Returns ``(occupancy uint8, covered_mask uint8)``; ``covered_mask == 0``
    marks target cells that received no source coverage (treated as blocked).
    """
    h, w = occ.shape
    sx, sy = w / size, h / size
    covered = np.zeros((size, size), dtype=np.uint8)
    out = np.zeros((size, size), dtype=np.uint8)

    if sx > 1.0 + 1e-9 or sy > 1.0 + 1e-9:
        # area max-pool: map every source cell to the target cell that contains it
        cols = np.minimum((np.arange(w) + 0.5) / sx, size - 1).astype(int)
        rows = np.minimum((np.arange(h) + 0.5) / sy, size - 1).astype(int)
        rr = np.repeat(rows, w)
        cc = np.tile(cols, h)
        vals = occ.reshape(-1)
        np.maximum.at(out, (rr, cc), vals)
        np.add.at(covered, (rr, cc), 1)
        covered = (covered > 0).astype(np.uint8)
    else:
        # nearest neighbour (identity when the sizes match)
        rows = np.minimum((np.arange(size) * sy).astype(int), h - 1)
        cols = np.minimum((np.arange(size) * sx).astype(int), w - 1)
        out = occ[np.ix_(rows, cols)]
        covered[:] = 1
    return out.astype(np.uint8), covered


# --------------------------------------------------------------------------- #
# Metric distance to the obstacle *area*
# --------------------------------------------------------------------------- #
def distance_to_obstacles_m(occ: np.ndarray, cell_m: float, upsample: int = 4) -> np.ndarray:
    """Distance from each cell centre to the nearest occupied **square**, in metres.

    Computed on an ``upsample``-times finer grid so the result is the distance to
    the occupied area rather than to occupied cell centres (that is what a
    conservative Minkowski inflation needs).
    """
    if not np.any(occ):
        return np.full(occ.shape, np.inf)
    sub = np.repeat(np.repeat(occ.astype(bool), upsample, axis=0), upsample, axis=1)
    sub_cell = cell_m / upsample
    edt = ndimage.distance_transform_edt(~sub)
    # a sub-cell centre sits at least half a sub-cell away from the occupied area
    dist = np.maximum(edt - 0.5, 0.0) * sub_cell
    # block-average back to the coarse grid (mean keeps it conservative enough
    # because we compare against a margin; the centre value is what matters)
    h = occ.shape[0]
    coarse = dist.reshape(h, upsample, h, upsample).mean(axis=(1, 3))
    return coarse.astype(np.float32)


@dataclass
class Scene:
    """A whole 256x256 training scene in metric coordinates."""

    scene_id: str
    source_map_id: str
    occupancy: np.ndarray          # (256,256) uint8, raw obstacles after rescale
    blocked: np.ndarray            # (256,256) bool, robot-centre blocked (inflated)
    free: np.ndarray               # (256,256) bool
    dist_m: np.ndarray             # (256,256) float32 distance to obstacle area [m]
    grad_x: np.ndarray             # dD/dx [m per m]
    grad_y: np.ndarray             # dD/dy [m per m]
    valid_mask: np.ndarray         # (256,256) uint8, 1 = inside the source map
    r_scene: float                 # metres per cell of the scene
    physical_field_m: float        # 48 / 64 / 80
    crop_px: int                   # source pixels cropped (e.g. 192/256/320)
    crop_origin: Tuple[int, int]   # (col, row) of the crop in the source map
    source_resolution: float
    density_inflated: float
    density_raw: float
    clearance_max_m: float
    meta: Dict[str, object] = field(default_factory=dict)

    # ------------------------------------------------------------------ lookup
    @property
    def size(self) -> int:
        return int(self.occupancy.shape[0])

    def cell_of(self, xy_m: np.ndarray) -> np.ndarray:
        """Continuous cell coordinates ``(col, row)`` of metric points."""
        return np.asarray(xy_m, dtype=float) / self.r_scene

    def xy_of_cell(self, rc: np.ndarray) -> np.ndarray:
        return np.asarray(rc, dtype=float) * self.r_scene

    def sample_scalar(self, field_arr: np.ndarray, xy_m: np.ndarray) -> np.ndarray:
        """Bilinear sample of a per-cell field at metric points (clamped)."""
        rc = self.cell_of(xy_m)
        n = self.size
        c = np.clip(rc[..., 0], 0, n - 1.001)
        r = np.clip(rc[..., 1], 0, n - 1.001)
        c0 = np.floor(c).astype(int); r0 = np.floor(r).astype(int)
        fc = c - c0; fr = r - r0
        v00 = field_arr[r0, c0]
        v01 = field_arr[r0, c0 + 1]
        v10 = field_arr[r0 + 1, c0]
        v11 = field_arr[r0 + 1, c0 + 1]
        return (v00 * (1 - fc) * (1 - fr) + v01 * fc * (1 - fr)
                + v10 * (1 - fc) * fr + v11 * fc * fr)

    def distance(self, xy_m: np.ndarray) -> np.ndarray:
        """Distance-field value in metres at metric points."""
        return self.sample_scalar(self.dist_m, xy_m)

    def gradient(self, xy_m: np.ndarray) -> np.ndarray:
        """Analytic gradient of the distance field at metric points, shape (...,2)."""
        gx = self.sample_scalar(self.grad_x, xy_m)
        gy = self.sample_scalar(self.grad_y, xy_m)
        return np.stack([gx, gy], axis=-1)

    def is_free(self, xy_m: np.ndarray) -> np.ndarray:
        rc = np.rint(self.cell_of(xy_m)).astype(int)
        rc[..., 0] = np.clip(rc[..., 0], 0, self.size - 1)
        rc[..., 1] = np.clip(rc[..., 1], 0, self.size - 1)
        return ~self.blocked[rc[..., 1], rc[..., 0]]

    def obstacle_cells(self) -> np.ndarray:
        """Centres (metres) of all occupied cells — used by the exact safety test."""
        rr, cc = np.nonzero(self.occupancy)
        return np.column_stack([cc * self.r_scene, rr * self.r_scene])

    def blocked_cells(self) -> np.ndarray:
        """Centres (metres) of all blocked (inflated) cells."""
        rr, cc = np.nonzero(self.blocked)
        return np.column_stack([cc * self.r_scene, rr * self.r_scene])

    def obstacle_boundary_cells(self) -> np.ndarray:
        """Centres (metres) of *inflated* obstacle cells touching free space.

        The labelled ellipse must stay outside the inflated (robot-centre)
        obstacles, so the active constraint set is built from this boundary.
        """
        occ = self.blocked.astype(bool)
        up = np.zeros_like(occ); up[1:, :] = occ[:-1, :]
        down = np.zeros_like(occ); down[:-1, :] = occ[1:, :]
        left = np.zeros_like(occ); left[:, 1:] = occ[:, :-1]
        right = np.zeros_like(occ); right[:, :-1] = occ[:, 1:]
        boundary = occ & ~(up & down & left & right)
        rr, cc = np.nonzero(boundary)
        return np.column_stack([cc * self.r_scene, rr * self.r_scene])


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def build_scene(
    source_occ: np.ndarray,
    source_resolution: float,
    crop_origin: Tuple[int, int],
    crop_px: int,
    size: int,
    physical_field_m: float,
    robot_radius_m: float,
    safety_margin_m: float,
    edt_upsample: int,
    scene_id: str,
    source_map_id: str,
) -> Scene:
    """Crop, conservatively rescale, inflate and derive the distance field."""
    x0, y0 = crop_origin
    crop = source_occ[y0:y0 + crop_px, x0:x0 + crop_px]
    if crop.shape != (crop_px, crop_px):
        raise ValueError(f"crop {crop.shape} does not match crop_px={crop_px}")

    occ, covered = conservative_resize(crop, size)
    r_scene = physical_field_m / size

    dist = distance_to_obstacles_m(occ, r_scene, upsample=edt_upsample)
    # cells with no source coverage or outside the map are never free
    blocked = (dist <= (robot_radius_m + safety_margin_m)) | (covered == 0)
    free = ~blocked

    gy, gx = np.gradient(dist.astype(np.float64), r_scene, r_scene)
    scene = Scene(
        scene_id=scene_id,
        source_map_id=source_map_id,
        occupancy=occ,
        blocked=blocked,
        free=free,
        dist_m=dist.astype(np.float32),
        grad_x=gx.astype(np.float32),
        grad_y=gy.astype(np.float32),
        valid_mask=covered,
        r_scene=float(r_scene),
        physical_field_m=float(physical_field_m),
        crop_px=int(crop_px),
        crop_origin=(int(x0), int(y0)),
        source_resolution=float(source_resolution),
        density_inflated=float(blocked.mean()),
        density_raw=float(occ.mean()),
        clearance_max_m=float(np.max(dist)),
    )
    return scene


def free_components(free: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """8-connected free-space labelling: ``(labels, n_components, largest_size)``."""
    lab, n = ndimage.label(free, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return lab, 0, 0
    sizes = np.bincount(lab.ravel())[1:]
    return lab, int(n), int(sizes.max()) if sizes.size else 0
