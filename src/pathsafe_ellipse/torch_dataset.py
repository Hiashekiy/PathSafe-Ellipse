"""In-memory split container: bit-packed patches unpacked batch-wise.

The patches dominate memory (128x128 per sample), so the splits store them
bit-packed.  Unpacking a batch costs ~1 ms, which makes DataLoader worker
processes (and their pipes) unnecessary.
"""
from __future__ import annotations

import os
from typing import Dict, Iterator, Optional, Tuple

import numpy as np

REQUIRED_FIELDS = ("ray_distances", "ray_hits", "path_tangent", "ellipse_ab",
                   "ellipse_angle", "ellipse_center_offset", "angle_valid")


class SplitData:
    def __init__(self, path: str):
        with np.load(path) as data:
            self.arrays: Dict[str, np.ndarray] = {k: data[k] for k in data.files}
        self.path = path
        if not self.arrays:                     # empty split (e.g. tiny dev runs)
            self.packed = np.zeros((0, 128, 16), dtype=np.uint8)
            self.n = 0
            return
        if "map_patch_packed" in self.arrays:
            self.packed = self.arrays["map_patch_packed"]
        elif "map_patch" in self.arrays:
            self.packed = np.packbits(self.arrays["map_patch"].reshape(-1, 128, 128), axis=2)
            self.arrays["map_patch_packed"] = self.packed
            del self.arrays["map_patch"]
        else:
            raise KeyError(f"{path}: no map_patch / map_patch_packed field")
        missing = [f for f in REQUIRED_FIELDS if f not in self.arrays]
        if missing:
            raise KeyError(f"{path}: missing fields {missing}")
        self.n = int(self.packed.shape[0])

    # ------------------------------------------------------------------ access
    def patches(self, idx: np.ndarray) -> np.ndarray:
        packed = self.packed[idx] if isinstance(idx, np.ndarray) else self.packed[[idx]]
        return np.unpackbits(packed, axis=2).reshape(len(packed), 128, 128)[:, None, :, :]

    def batch(self, idx: np.ndarray) -> Dict[str, np.ndarray]:
        a = self.arrays
        return {
            "patch": self.patches(idx).astype(np.float32),
            "rays": a["ray_distances"][idx].astype(np.float32),
            "hits": a["ray_hits"][idx].astype(np.float32),
            "tangent": a["path_tangent"][idx].astype(np.float32),
            "center_offset": a["ellipse_center_offset"][idx].astype(np.float32),
            "ab": a["ellipse_ab"][idx].astype(np.float32),
            "sincos": a["ellipse_angle"][idx].astype(np.float32),
            "angle_valid": a["angle_valid"][idx].astype(np.float32),
        }

    def target_batch(self, idx: np.ndarray) -> Dict[str, np.ndarray]:
        b = self.batch(idx)
        return {k: b[k] for k in ("center_offset", "ab", "sincos", "angle_valid")}

    def epoch(self, batch_size: int, rng: np.random.Generator,
              drop_last: bool = True) -> Iterator[np.ndarray]:
        order = rng.permutation(self.n)
        limit = self.n - (self.n % batch_size if drop_last else 0)
        for start in range(0, limit, batch_size):
            yield order[start:start + batch_size]


def stats_note() -> str:
    return ("split arrays are loaded once; patches stay bit-packed in RAM and are "
            "unpacked per batch (no DataLoader workers, no shared-memory pipes)")
