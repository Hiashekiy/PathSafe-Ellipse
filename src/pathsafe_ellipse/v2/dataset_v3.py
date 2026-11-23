"""V3 dataset: one training sample == one complete trajectory.

Reads the trajectory-level NPZs written by :func:`pilot.save_trajectory_npz` and
returns them unchanged:

    occupancy      (256, 256)   inflated occupancy, 0 = traversable, 1 = blocked
    path_xy        (N, 2)       resampled A* trajectory (metres)
    centerline_xy  (N, 2)       optimised centre line   (metres)
    lidar_ranges   (N, 16)      LiDAR cast from path_xy[i]
    center_offset  (N, 2)       centerline_xy - path_xy
    ellipse_S      (N, 2, 2)    ellipse shape matrices (m^2)
    ellipse_ab, ellipse_angle, ellipse_area          (N, 2) / (N,) / (N,)
    safety_verified, continuity_verified, label_valid, solver_status   (N,)

The map is never cropped, no clearance feature is produced and a trajectory is
never split into independent single-point samples.  ``N`` varies per trajectory;
batching is the DataLoader's job (see :func:`collate_trajectories`, which pads to
the longest trajectory of the batch and returns an explicit ``mask``).
"""
from __future__ import annotations

import glob
import os
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .dataset_v21 import split_by_group          # noqa: F401  (same group policy)
from .pilot import load_trajectory_npz

#: map arrays of a trajectory NPZ
MAP_FIELDS = ("occupancy",)
#: per-point arrays (all of length N, one row per query point)
POINT_FIELDS = ("path_xy", "centerline_xy", "lidar_ranges", "center_offset",
                "ellipse_S", "ellipse_ab", "ellipse_angle", "ellipse_area",
                "safety_verified", "continuity_verified", "label_valid",
                "solver_status")
#: single-value arrays
SCALAR_FIELDS = ("start_xy", "goal_xy")


def list_trajectory_files(root: str) -> List[str]:
    """Every trajectory NPZ below ``root`` (sorted, recursive)."""
    return sorted(glob.glob(os.path.join(root, "**", "*.npz"), recursive=True))


def load_trajectory(path: str) -> Dict[str, object]:
    """Load one trajectory sample (arrays + decoded ``meta``)."""
    return load_trajectory_npz(path)


def check_sample(sample: Dict[str, object], size: Optional[int] = 256,
                 offset_tol: float = 1e-6) -> List[str]:
    """Structural acceptance checks of spec 6.1; returns the failures (``[] = ok``)."""
    problems: List[str] = []
    for key in MAP_FIELDS + POINT_FIELDS + SCALAR_FIELDS:
        if key not in sample:
            problems.append(f"missing array '{key}'")
    if problems:
        return problems

    occ = np.asarray(sample["occupancy"])
    if occ.ndim != 2 or (size is not None and occ.shape != (size, size)):
        problems.append(f"occupancy shape {occ.shape} != {(size, size)}")
    if not np.all(np.isin(np.unique(occ), (0, 1))):
        problems.append("occupancy must be 0/1")
    if "meta" not in sample:
        problems.append("missing meta_json")

    n = int(np.asarray(sample["path_xy"]).shape[0])
    if n < 2:
        problems.append(f"trajectory has {n} query points (< 2)")
    for key in POINT_FIELDS:
        arr = np.asarray(sample[key])
        if int(arr.shape[0]) != n:
            problems.append(f"'{key}' has {arr.shape[0]} rows, path_xy has {n}")

    p = np.asarray(sample["path_xy"], dtype=np.float64).reshape(-1, 2)
    q = np.asarray(sample["centerline_xy"], dtype=np.float64).reshape(-1, 2)
    off = np.asarray(sample["center_offset"], dtype=np.float64).reshape(-1, 2)
    S = np.asarray(sample["ellipse_S"], dtype=np.float64).reshape(-1, 2, 2)
    ab = np.asarray(sample["ellipse_ab"], dtype=np.float64).reshape(-1, 2)
    lidar = np.asarray(sample["lidar_ranges"], dtype=np.float64)
    for name, arr in (("path_xy", p), ("centerline_xy", q), ("center_offset", off),
                      ("ellipse_S", S), ("ellipse_ab", ab), ("lidar_ranges", lidar),
                      ("ellipse_angle", np.asarray(sample["ellipse_angle"], np.float64)),
                      ("ellipse_area", np.asarray(sample["ellipse_area"], np.float64)),
                      ("start_xy", np.asarray(sample["start_xy"], np.float64)),
                      ("goal_xy", np.asarray(sample["goal_xy"], np.float64))):
        if not np.all(np.isfinite(arr)):
            problems.append(f"'{name}' contains NaN/Inf")
    if np.any(lidar < 0):
        problems.append("lidar_ranges has negative ranges")
    if not np.allclose(off, q - p, atol=offset_tol, rtol=0.0):
        problems.append("center_offset != centerline_xy - path_xy")
    if np.any(np.linalg.eigvalsh(S).min(axis=-1) <= 0):
        problems.append("ellipse_S is not positive definite")
    if np.any(np.linalg.norm(off, axis=1) > 1e3):
        problems.append("center_offset implausibly large")
    return problems


class TrajectoryDataset:
    """One item per trajectory NPZ (variable ``N``); nothing is cropped or split."""

    def __init__(self, root: str, files: Optional[Sequence[str]] = None,
                 size: Optional[int] = 256):
        self.root = root
        self.files = list(files) if files is not None else list_trajectory_files(root)
        if not self.files:
            raise FileNotFoundError(f"no trajectory NPZ under {root}")
        self.size = size

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = load_trajectory(self.files[index])
        sample["path"] = self.files[index]
        return sample

    def meta(self, index: int) -> Dict[str, object]:
        with np.load(self.files[index], allow_pickle=False) as z:
            raw = z["meta_json"]
        import json
        return json.loads(str(raw.item() if raw.ndim == 0 else raw))

    def __iter__(self) -> Iterable[Dict[str, object]]:
        for i in range(len(self)):
            yield self[i]


def collate_trajectories(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Pad a batch of variable-length trajectories to ``N_max``.

    Returns the padded arrays plus ``mask`` (B, N_max) telling the network which
    query points are real; nothing is truncated.
    """
    n_max = max(int(np.asarray(b["path_xy"]).shape[0]) for b in batch)
    b_size = len(batch)
    ray = int(np.asarray(batch[0]["lidar_ranges"]).shape[-1])
    out: Dict[str, object] = {
        "occupancy": np.stack([np.asarray(b["occupancy"], np.uint8) for b in batch]),
        "mask": np.zeros((b_size, n_max), bool),
        "path_xy": np.zeros((b_size, n_max, 2), np.float32),
        "centerline_xy": np.zeros((b_size, n_max, 2), np.float32),
        "lidar_ranges": np.zeros((b_size, n_max, ray), np.float32),
        "center_offset": np.zeros((b_size, n_max, 2), np.float32),
        "ellipse_S": np.zeros((b_size, n_max, 2, 2), np.float32),
        "ellipse_ab": np.zeros((b_size, n_max, 2), np.float32),
        "ellipse_angle": np.zeros((b_size, n_max), np.float32),
        "ellipse_area": np.zeros((b_size, n_max), np.float32),
        "safety_verified": np.zeros((b_size, n_max), np.uint8),
        "continuity_verified": np.zeros((b_size, n_max), np.uint8),
        "label_valid": np.zeros((b_size, n_max), np.uint8),
        "solver_status": np.full((b_size, n_max), "", dtype="U32"),
        "start_xy": np.zeros((b_size, 2), np.float32),
        "goal_xy": np.zeros((b_size, 2), np.float32),
        "meta": [b.get("meta", {}) for b in batch],
        "path": [b.get("path", "") for b in batch],
    }
    for i, b in enumerate(batch):
        n = int(np.asarray(b["path_xy"]).shape[0])
        out["mask"][i, :n] = True
        for key in ("path_xy", "centerline_xy", "center_offset", "ellipse_ab",
                    "ellipse_S", "ellipse_angle", "ellipse_area", "lidar_ranges",
                    "safety_verified", "continuity_verified", "label_valid",
                    "solver_status"):
            out[key][i, :n] = np.asarray(b[key])
        for key in SCALAR_FIELDS:
            out[key][i] = np.asarray(b[key])
    return out


# --------------------------------------------------------------------------- #
# Dataset-level statistics (spec 6.4, no clearance metrics)
# --------------------------------------------------------------------------- #
def _dist(values: np.ndarray) -> Dict[str, Optional[float]]:
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"mean": None, "p05": None, "p50": None, "p95": None,
                "min": None, "max": None}
    return {"mean": float(v.mean()), "p05": float(np.percentile(v, 5)),
            "p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)),
            "min": float(v.min()), "max": float(v.max())}


def summarise_files(files: Sequence[str]) -> Dict[str, object]:
    """Aggregate the trajectory statistics required by the acceptance report."""
    areas, a_axis, b_axis, offsets, lidar_all = [], [], [], [], []
    safety, cont, valid, points_per_traj = [], [], [], []
    groups, scenes, statuses = set(), set(), {}
    for path in files:
        s = load_trajectory(path)
        meta = s.get("meta", {})
        groups.add(str(meta.get("map_group", "")))
        scenes.add(str(meta.get("scene_id", "")))
        p = np.asarray(s["path_xy"], np.float64)
        q = np.asarray(s["centerline_xy"], np.float64)
        points_per_traj.append(len(p))
        areas.append(np.asarray(s["ellipse_area"], np.float64))
        ab = np.asarray(s["ellipse_ab"], np.float64)
        a_axis.append(ab[:, 0]); b_axis.append(ab[:, 1])
        offsets.append(np.linalg.norm(q - p, axis=1))
        lidar_all.append(np.asarray(s["lidar_ranges"], np.float64))
        safety.append(np.asarray(s["safety_verified"], np.uint8).astype(bool))
        cont.append(np.asarray(s["continuity_verified"], np.uint8).astype(bool))
        valid.append(np.asarray(s["label_valid"], np.uint8).astype(bool))
        for st in np.asarray(s["solver_status"]).tolist():
            statuses[str(st)] = statuses.get(str(st), 0) + 1

    def cat(parts):
        return np.concatenate(parts) if parts else np.zeros(0)

    area, aa, bb = cat(areas), cat(a_axis), cat(b_axis)
    off, lid = cat(offsets), cat(lidar_all)
    saf, cnf, val = cat(safety), cat(cont), cat(valid)
    n_points = int(area.size)
    ppt = np.asarray(points_per_traj, np.float64)
    return {
        "n_files": len(files),
        "n_scenes": len(scenes),
        "n_map_groups": len(groups),
        "n_trajectories": len(points_per_traj),
        "n_query_points": n_points,
        "points_per_trajectory": _dist(ppt),
        "label_valid_rate": float(val.mean()) if n_points else None,
        "safety_verified_rate": float(saf.mean()) if n_points else None,
        "continuity_verified_rate": float(cnf.mean()) if n_points else None,
        "ellipse_area": _dist(area),
        "semi_major_a": _dist(aa),
        "semi_minor_b": _dist(bb),
        "center_offset_norm": _dist(off),
        "lidar_range": _dist(lid),
        "lidar_ray_mean": [float(v) for v in np.mean(lid, axis=0)] if n_points else [],
        "solver_status_counts": statuses,
    }
