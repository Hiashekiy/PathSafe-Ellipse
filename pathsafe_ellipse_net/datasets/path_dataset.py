"""Path-level dataset over the V3 trajectory NPZs (one sample = one trajectory)."""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ..geometry.sdf import signed_distance_field
from .augmentation import flip_sample
from .geometry_features import path_geometry


def list_trajectories(root: str) -> List[str]:
    return sorted(p for p in glob.glob(os.path.join(root, "**", "*.npz"), recursive=True)
                  if ".tmp." not in os.path.basename(p))


def read_meta(path: str) -> Dict[str, object]:
    with np.load(path, allow_pickle=False) as z:
        raw = z["meta_json"]
    return json.loads(str(raw.item() if raw.ndim == 0 else raw))


class TrajectoryDataset(Dataset):
    """Returns one padded-free trajectory item; batching happens in the collate fn.

    ``label_valid`` filtered targets are still returned (the loss masks them) so the
    dataset never silently drops a query point.
    """

    def __init__(self, files: Sequence[str], cfg: Dict, augment: bool = False,
                 sdf_cache_size: int = 64):
        self.files = list(files)
        self.cfg = cfg
        self.augment = augment
        self._sdf_cache: Dict[str, np.ndarray] = {}
        self._sdf_cache_size = int(sdf_cache_size)

    def __len__(self) -> int:
        return len(self.files)

    def _sdf(self, key: str, occupancy: np.ndarray, r_scene: float) -> np.ndarray:
        cached = self._sdf_cache.get(key)
        if cached is not None:
            return cached
        sdf = signed_distance_field(occupancy, r_scene)[None]        # (1,H,W)
        if len(self._sdf_cache) >= self._sdf_cache_size:
            self._sdf_cache.pop(next(iter(self._sdf_cache)))
        self._sdf_cache[key] = sdf
        return sdf

    def __getitem__(self, index: int) -> Dict[str, object]:
        path = self.files[index]
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta_json"].item()))
            occ = z["occupancy"].astype(np.float32)
            p = z["path_xy"].astype(np.float32)
            lidar = z["lidar_ranges"].astype(np.float32)
            offset = z["center_offset"].astype(np.float32)
            ab = z["ellipse_ab"].astype(np.float32)
            angle = z["ellipse_angle"].astype(np.float32)
            S = z["ellipse_S"].astype(np.float32)
            valid = z["label_valid"].astype(bool)
            safety = z["safety_verified"].astype(bool)
            cont = z["continuity_verified"].astype(bool)

        r_scene = float(meta["r_scene"])
        H, W = int(occ.shape[0]), int(occ.shape[1])
        extent = (W * r_scene, H * r_scene)
        sample = {
            "occupancy": occ[None],                                   # (1,H,W)
            "sdf": self._sdf(path, occ, r_scene),
            "path_xy": p,
            "lidar": lidar,                                           # metres
            "geometry": path_geometry(p, extent),
            "center_offset": offset,
            "ab": ab,
            "angle": angle,
            "S": S,
            "valid": valid,
            "safety": safety,
            "continuity": cont,
            "extent": extent,
            "r_scene": r_scene,
            "n_points": int(len(p)),
            "scene_id": str(meta["scene_id"]),
            "map_group": str(meta["map_group"]),
            "path_id": int(meta["path_id"]),
            "file": os.path.basename(path),
        }
        if self.augment:
            # the flip must depend on the SCENE, not on the trajectory: trajectories of
            # one scene share a single encoded map, so they must share the geometry too
            # (crc32 keeps it stable across processes, unlike hash())
            import zlib
            rng = np.random.default_rng(zlib.crc32(sample["scene_id"].encode()))
            fx, fy = bool(rng.integers(0, 2)), bool(rng.integers(0, 2))
            if fx or fy:
                sample = flip_sample(sample, fx, fy, (H, W))
        return sample


class SceneGroupedBatchSampler:
    """Yield batches whose items mostly come from the same scenes.

    The map encoder then runs once per scene instead of once per trajectory
    (plan section 9).  Batches are still filled to ``batch_size``; only the order is
    grouped, so nothing is dropped.
    """

    def __init__(self, scene_ids: Sequence[str], batch_size: int, shuffle: bool = True,
                 seed: int = 0, max_per_scene: int = 8):
        self.scene_ids = list(scene_ids)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.max_per_scene = int(max_per_scene)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        by_scene: Dict[str, List[int]] = {}
        for i, sid in enumerate(self.scene_ids):
            by_scene.setdefault(str(sid), []).append(i)
        keys = list(by_scene)
        rng = np.random.default_rng(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(keys)
        batch: List[int] = []
        for k in keys:
            idxs = by_scene[k]
            if self.shuffle:
                idxs = [idxs[j] for j in rng.permutation(len(idxs))]
            for i in idxs[: self.max_per_scene]:
                batch.append(i)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch

    def __len__(self) -> int:
        return int(np.ceil(len(self.scene_ids) / self.batch_size))


def split_by_group(groups: Sequence[str], frac=(0.8, 0.1, 0.1), seed: int = 0):
    """Split *map groups* (never points) into train/val/test index lists."""
    uniq = sorted(set(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr = max(1, int(round(frac[0] * n)))
    n_va = max(1, int(round(frac[1] * n))) if n > 2 else 0
    train_g = set(uniq[:n_tr])
    val_g = set(uniq[n_tr:n_tr + n_va])
    tr = [i for i, g in enumerate(groups) if g in train_g]
    va = [i for i, g in enumerate(groups) if g in val_g]
    te = [i for i, g in enumerate(groups) if g not in train_g and g not in val_g]
    return tr, va, te
