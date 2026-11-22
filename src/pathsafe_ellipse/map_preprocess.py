"""Map preprocessing: robot-footprint inflation, distance transform, obstacle boundaries."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict

import numpy as np
import scipy.ndimage as ndimage

from .map_io import MapRecord, load_moving_ai_map

STRUCT_3X3 = np.ones((3, 3), dtype=bool)


@dataclass
class InflatedMap:
    """Cached inflated occupancy grid plus its derived fields."""

    key: str
    occupancy_raw: np.ndarray       # (H, W) uint8, 0 free / 1 obstacle
    occupancy_inflated: np.ndarray  # (H, W) uint8, robot-centre planning map
    clearance: np.ndarray           # (H, W) float32, EDT on inflated map [cells]
    resolution: float
    robot_radius: float
    kernel_radius: int
    iterations: int


def inflate_once(occ: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Binary-dilate the obstacle mask with a 3x3 square structuring element."""
    if iterations <= 0:
        return occ.copy()
    mask = ndimage.binary_dilation(occ.astype(bool), structure=STRUCT_3X3, iterations=iterations)
    return mask.astype(np.uint8)


def distance_transform(occ_inflated: np.ndarray) -> np.ndarray:
    """Euclidean distance to the nearest obstacle cell centre, in cells.

    The map border is treated as obstacle, matching "everything outside the map
    is blocked" (a 1-cell pad is used for the transform).
    """
    padded = np.pad(occ_inflated.astype(bool), 1, constant_values=True)
    edt = ndimage.distance_transform_edt(~padded)
    return edt[1:-1, 1:-1].astype(np.float32)


def obstacle_boundary_mask(occ: np.ndarray) -> np.ndarray:
    """Obstacle cells that touch a free cell or the map border."""
    mask = occ.astype(bool)
    if not np.any(mask):
        return np.zeros_like(mask)
    up = np.zeros_like(mask); up[1:, :] = mask[:-1, :]
    down = np.zeros_like(mask); down[:-1, :] = mask[1:, :]
    left = np.zeros_like(mask); left[:, 1:] = mask[:, :-1]
    right = np.zeros_like(mask); right[:, :-1] = mask[:, 1:]
    interior = up & down & left & right
    return mask & (~interior)


def _cache_paths(cache_dir: str, key: str) -> Dict[str, str]:
    return {
        "npz": os.path.join(cache_dir, f"{key}.npz"),
        "json": os.path.join(cache_dir, f"{key}.json"),
    }


def preprocess_map(
    record: MapRecord,
    map_dir: str,
    cache_dir: str,
    robot_radius: float,
    kernel_radius: int,
    iterations: int,
    use_cache: bool = True,
) -> InflatedMap:
    """Load one map and produce the inflated planning map (cached on disk)."""
    key = os.path.splitext(record.file_name)[0]
    paths = _cache_paths(cache_dir, key)
    fingerprint = hashlib.md5(
        json.dumps(
            {
                "md5": record.md5,
                "robot_radius": robot_radius,
                "kernel_radius": kernel_radius,
                "iterations": iterations,
                "resolution": record.resolution,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()

    if use_cache and os.path.exists(paths["npz"]) and os.path.exists(paths["json"]):
        with open(paths["json"], "r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if cached.get("fingerprint") == fingerprint:
            data = np.load(paths["npz"])
            return InflatedMap(
                key=key,
                occupancy_raw=np.unpackbits(data["raw_packed"], axis=1)[:, : record.width].astype(np.uint8),
                occupancy_inflated=np.unpackbits(data["inflated_packed"], axis=1)[:, : record.width].astype(np.uint8),
                clearance=data["clearance"].astype(np.float32),
                resolution=cached["resolution"],
                robot_radius=robot_radius,
                kernel_radius=kernel_radius,
                iterations=iterations,
            )

    grid, meta = load_moving_ai_map(os.path.join(map_dir, record.rel_path))
    inflated = inflate_once(grid, iterations=iterations)
    clearance = distance_transform(inflated)

    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(
        paths["npz"],
        raw_packed=np.packbits(grid, axis=1),
        inflated_packed=np.packbits(inflated, axis=1),
        clearance=clearance,
    )
    with open(paths["json"], "w", encoding="utf-8") as fh:
        json.dump(
            {
                "fingerprint": fingerprint,
                "key": key,
                "resolution": float(meta["resolution"] or record.resolution),
                "robot_radius": robot_radius,
                "kernel_radius": kernel_radius,
                "iterations": iterations,
                "kernel": "3x3 square, all ones",
                "width": record.width,
                "height": record.height,
                "source_rel_path": record.rel_path,
            },
            fh,
            indent=2,
        )
    return InflatedMap(
        key=key,
        occupancy_raw=grid,
        occupancy_inflated=inflated,
        clearance=clearance,
        resolution=float(meta["resolution"] or record.resolution),
        robot_radius=robot_radius,
        kernel_radius=kernel_radius,
        iterations=iterations,
    )
