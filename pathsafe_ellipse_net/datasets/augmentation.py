"""Horizontal / vertical flip augmentation (plan section 2.4).

Everything is transformed consistently: map, path, tangent (inside the geometry
features), LiDAR ray order, centre offset, semi-axes/angle and shape matrix.

Rays are indexed by direction ``phi_j = 2 pi j / 16``:

* flip x (reflection about the y axis)  : ``phi -> pi - phi``   -> ``j' = (8 - j) % 16``
* flip y (reflection about the x axis)  : ``phi -> -phi``       -> ``j' = (-j) % 16``
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def ray_permutation(n_rays: int, flip_x: bool, flip_y: bool) -> np.ndarray:
    """Index array such that ``new_lidar[..., k] = old_lidar[..., perm[k]]``."""
    j = np.arange(n_rays)
    if flip_x:
        j = (n_rays // 2 - j) % n_rays
    if flip_y:
        j = (-j) % n_rays
    return j.astype(np.int64)


def flip_angle(angle: np.ndarray, flip_x: bool, flip_y: bool) -> np.ndarray:
    a = np.asarray(angle, dtype=np.float64)
    if flip_x:
        a = np.pi - a
    if flip_y:
        a = -a
    return a


def flip_shape_matrix(S: np.ndarray, flip_x: bool, flip_y: bool) -> np.ndarray:
    R = np.eye(2)
    if flip_x:
        R = np.diag([-1.0, 1.0]) @ R
    if flip_y:
        R = np.diag([1.0, -1.0]) @ R
    return R @ S @ R.T


def flip_sample(sample: Dict[str, np.ndarray], flip_x: bool, flip_y: bool,
                map_size: tuple) -> Dict[str, np.ndarray]:
    """Return a flipped copy of one dataset item (arrays are not modified in place)."""
    out = dict(sample)
    H, W = map_size
    extent_x = float(sample["extent"][0])
    extent_y = float(sample["extent"][1])

    occ = np.asarray(sample["occupancy"]).copy()
    sdf = np.asarray(sample["sdf"]).copy()
    if flip_x:
        occ = occ[:, :, ::-1].copy()
        sdf = sdf[:, :, ::-1].copy()
    if flip_y:
        occ = occ[:, ::-1, :].copy()
        sdf = sdf[:, ::-1, :].copy()
    out["occupancy"], out["sdf"] = occ, sdf

    p = np.asarray(sample["path_xy"]).copy()
    off = np.asarray(sample["center_offset"]).copy()
    geom = np.asarray(sample["geometry"]).copy()
    lidar = np.asarray(sample["lidar"]).copy()
    ab = np.asarray(sample["ab"]).copy()
    angle = np.asarray(sample["angle"]).copy()
    S = np.asarray(sample["S"]).copy()

    if flip_x:
        p[:, 0] = extent_x - p[:, 0]
        off[:, 0] = -off[:, 0]
        geom[:, 0] = -geom[:, 0]                      # tangent x
        geom[:, 4] = 1.0 - geom[:, 4]                 # x_norm
    if flip_y:
        p[:, 1] = extent_y - p[:, 1]
        off[:, 1] = -off[:, 1]
        geom[:, 1] = -geom[:, 1]
        geom[:, 5] = 1.0 - geom[:, 5]

    perm = ray_permutation(lidar.shape[-1], flip_x, flip_y)
    lidar = lidar[..., perm]
    angle = flip_angle(angle, flip_x, flip_y)
    S = np.stack([flip_shape_matrix(s, flip_x, flip_y) for s in S]) if len(S) else S

    out.update({"path_xy": p, "center_offset": off, "geometry": geom, "lidar": lidar,
                "ab": ab, "angle": angle.astype(np.float32), "S": S,
                "flip_x": bool(flip_x), "flip_y": bool(flip_y)})
    return out
