"""V2.1 dataset assembly: whole-scene NPZ label files -> training tensors.

Features come from the ORIGINAL A* point ``p_i`` only (local sensing), never from
the label centre, so the network has to predict where the ellipse should go:

    lidar   : 16 ranges / max_range                      (16,)
    tangent : unit path tangent at the station            (2,)
    patch   : occupancy crop centred on p_i, resampled    (1, P, P)
    extra   : clearance at p_i, offset-free scalars       (2,)

Targets:
    offset  : q_i - p_i                                  (2,)
    axes    : log(a), log(b)                             (2,)
    angle   : cos(2θ), sin(2θ)                           (2,)

Splitting is by MAP GROUP so no map appears in two splits.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .pilot import load_scene_npz

PATCH_PX = 64          # 64 x 64 around p_i (at r_scene -> 64 * 0.19..0.31 m)


def _patch(npz, p_xy: np.ndarray, r_scene: float, px: int = PATCH_PX) -> np.ndarray:
    """Occupancy crop centred on ``p_xy`` (nearest-neighbour, zero padded)."""
    occ = npz["occupancy"]
    n = occ.shape[0]
    cx = p_xy[0] / r_scene
    cy = p_xy[1] / r_scene
    half = px // 2
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    out = np.zeros((px, px), dtype=np.float32)
    xs0, ys0 = max(0, x0), max(0, y0)
    xs1, ys1 = min(n, x0 + px), min(n, y0 + px)
    if xs1 > xs0 and ys1 > ys0:
        out[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0] = occ[ys0:ys1, xs0:xs1]
    return out


def load_split_files(root: str) -> List[str]:
    return sorted(glob.glob(os.path.join(root, "*.npz")))


def build_table(files: List[str], with_patch: bool = True,
                only_valid: bool = True) -> Dict[str, np.ndarray]:
    """Flatten label rows of many scene NPZs into arrays."""
    lidar, tan, clr, patch = [], [], [], []
    off, axes, ang, groups, scenes, pids, pts = [], [], [], [], [], [], []
    for path in files:
        npz = load_scene_npz(path)
        meta = npz["meta"]
        rs = float(meta["r_scene_m"])
        max_r = float(meta["lidar_max_range_m"])
        keep = (npz["label_valid"].astype(bool) if only_valid
                else np.ones(len(npz["q_xy"]), dtype=bool))
        for k in np.flatnonzero(keep):
            p = npz["p_xy"][k].astype(np.float32)
            q = npz["q_xy"][k].astype(np.float32)
            lidar.append(npz["lidar_ranges"][k].astype(np.float32) / max_r)
            tan.append(npz["path_tangent"][k].astype(np.float32))
            clr.append(np.float32(npz["clearance"][k]))
            if with_patch:
                patch.append(_patch(npz, p, rs))
            ov = (q - p).astype(np.float32)
            off.append(ov)
            ab = npz["ellipse_ab"][k].astype(np.float32)
            axes.append(np.log(np.maximum(ab, 1e-3)).astype(np.float32))
            th = float(npz["ellipse_angle"][k])
            ang.append(np.array([np.cos(2 * th), np.sin(2 * th)], dtype=np.float32))
            groups.append(str(meta["map_group"]))
            scenes.append(str(meta["scene_id"]))
            pids.append(int(npz["path_id"][k]))
            pts.append(int(npz["point_id"][k]))
    out = {
        "lidar": np.stack(lidar) if lidar else np.zeros((0, 16), np.float32),
        "tangent": np.stack(tan) if tan else np.zeros((0, 2), np.float32),
        "clearance_p": np.asarray(clr, dtype=np.float32).reshape(-1, 1),
        "offset": np.stack(off) if off else np.zeros((0, 2), np.float32),
        "log_axes": np.stack(axes) if axes else np.zeros((0, 2), np.float32),
        "angle2": np.stack(ang) if ang else np.zeros((0, 2), np.float32),
        "group": np.asarray(groups),
        "scene_id": np.asarray(scenes),
        "path_id": np.asarray(pids, dtype=np.int32),
        "point_id": np.asarray(pts, dtype=np.int32),
    }
    if with_patch:
        out["patch"] = (np.stack(patch)[:, None, :, :] if patch
                        else np.zeros((0, 1, PATCH_PX, PATCH_PX), np.float32))
    return out


def split_by_group(groups: np.ndarray, seed: int = 0,
                   frac=(0.7, 0.15, 0.15)) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train_mask, val_mask) with test the remainder, split by map group."""
    uniq = np.array(sorted(set(groups.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr = max(1, int(round(frac[0] * n)))
    n_va = max(1, int(round(frac[1] * n)))
    tr = set(uniq[:n_tr].tolist())
    va = set(uniq[n_tr:n_tr + n_va].tolist())
    mask_tr = np.array([g in tr for g in groups])
    mask_va = np.array([g in va for g in groups])
    mask_te = ~(mask_tr | mask_va)
    return mask_tr, mask_va, mask_te


def summarise(table: Dict[str, np.ndarray]) -> Dict[str, object]:
    ab = np.exp(table["log_axes"])
    area = np.pi * ab[:, 0] * ab[:, 1]
    err = np.linalg.norm(table["offset"], axis=1)
    return {
        "n": int(len(area)),
        "scenes": int(len(set(table["scene_id"].tolist()))),
        "groups": int(len(set(table["group"].tolist()))),
        "area_p50": float(np.median(area)) if len(area) else None,
        "area_min": float(area.min()) if len(area) else None,
        "area_max": float(area.max()) if len(area) else None,
        "offset_mean": float(err.mean()) if len(err) else None,
        "offset_p95": float(np.percentile(err, 95)) if len(err) else None,
        "a_p50": float(np.median(ab[:, 0])) if len(area) else None,
        "b_p50": float(np.median(ab[:, 1])) if len(area) else None,
        "capped_share": float(np.mean(ab[:, 0] > 5.9)) if len(area) else None,
        "tiny_share": float(np.mean(area < 1.0)) if len(area) else None,
    }


@dataclass
class Standardiser:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardiser":
        return cls(mean=x.mean(axis=0), std=x.std(axis=0) + 1e-6)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def state(self) -> Dict[str, list]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def save_meta(path: str, payload: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=float)
