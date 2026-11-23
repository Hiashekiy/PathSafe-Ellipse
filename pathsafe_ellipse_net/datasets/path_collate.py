"""Batch collation for variable-length trajectories (padding + masks)."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


def collate_paths(batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
    """Pad to the longest trajectory of the batch; return masks and pair masks.

    Trajectories of the same scene are stacked once in ``unique_occupancy`` /
    ``unique_sdf`` and referenced by ``map_index``, so the map encoder runs once per
    *scene* instead of once per trajectory (plan section 9).
    """
    B = len(batch)
    n_max = max(int(b["n_points"]) for b in batch)
    rays = int(np.asarray(batch[0]["lidar"]).shape[-1])
    H, W = np.asarray(batch[0]["occupancy"]).shape[-2:]

    scene_ids = [b["scene_id"] for b in batch]
    order: List[str] = []
    map_index = torch.zeros(B, dtype=torch.long)
    for i, sid in enumerate(scene_ids):
        if sid not in order:
            order.append(sid)
        map_index[i] = order.index(sid)
    unique_occ = torch.zeros((len(order), 1, H, W))
    unique_sdf = torch.zeros((len(order), 1, H, W))
    for u, sid in enumerate(order):
        ref = batch[scene_ids.index(sid)]
        unique_occ[u] = torch.from_numpy(np.asarray(ref["occupancy"]))
        unique_sdf[u] = torch.from_numpy(np.asarray(ref["sdf"]))

    out: Dict[str, object] = {
        "occupancy": unique_occ,          # (U,1,H,W)  — unique scenes only
        "sdf": unique_sdf,
        "map_index": map_index,           # (B,)  which unique map each item uses
        "path_xy": torch.zeros((B, n_max, 2)),
        "lidar": torch.zeros((B, n_max, rays)),
        "geometry": torch.zeros((B, n_max, int(np.asarray(batch[0]["geometry"]).shape[-1]))),
        "center_offset": torch.zeros((B, n_max, 2)),
        "ab": torch.zeros((B, n_max, 2)),
        "angle": torch.zeros((B, n_max)),
        "S": torch.zeros((B, n_max, 2, 2)),
        "valid": torch.zeros((B, n_max), dtype=torch.bool),
        "safety": torch.zeros((B, n_max), dtype=torch.bool),
        "continuity": torch.zeros((B, n_max), dtype=torch.bool),
        "mask": torch.zeros((B, n_max), dtype=torch.bool),
        "r_scene": torch.zeros((B,)),
    }
    for i, b in enumerate(batch):
        n = int(b["n_points"])
        for key in ("path_xy", "lidar", "geometry", "center_offset", "ab", "angle", "S",
                    "valid", "safety", "continuity"):
            out[key][i, :n] = torch.from_numpy(np.asarray(b[key]))
        out["mask"][i, :n] = True
        out["r_scene"][i] = float(b["r_scene"])

    out["pair_mask"] = out["mask"][:, 1:] & out["mask"][:, :-1]
    out["meta"] = [{"scene_id": b["scene_id"], "map_group": b["map_group"],
                    "path_id": b["path_id"], "file": b["file"],
                    "extent": b["extent"], "n_points": b["n_points"]} for b in batch]
    return out
