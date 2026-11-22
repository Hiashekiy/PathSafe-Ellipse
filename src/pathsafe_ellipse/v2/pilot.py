"""Pilot artefact I/O: metadata block, per-scene NPZ with paths + labels, index.

The NPZ reuses the *formal* pipeline field names (``anchor_id``, ``path_id``,
``p_xy``, ``q_xy``, ``ellipse_S``, ``ellipse_ab``, ``ellipse_area`` ...) plus
``point_id`` and ``label_valid`` so a dataset built from the pilot is
indistinguishable from a production one apart from the version metadata.
"""
from __future__ import annotations

import json
import os
import subprocess
from typing import Dict, List, Optional

import numpy as np

from .centerline import PathData
from .scene import Scene

LABEL_FIELDS = ("point_id", "anchor_id", "path_id", "p_xy", "q_xy", "center_offset",
                "path_tangent", "lidar_ranges", "ellipse_S", "ellipse_ab", "ellipse_angle",
                "ellipse_area", "clearance", "solver_status", "safety_verified",
                "continuity_verified", "label_valid", "generation_time_ms",
                "rho_left", "rho_right")
PATH_FIELDS = ("path_id", "n_stations", "n_anchors", "n_labels", "n_ellipse_safe",
               "n_continuity_verified", "n_both_verified", "n_inserted", "length_m",
               "clearance_p_mean", "clearance_q_mean", "clearance_gain_mean",
               "max_offset_m", "length_ratio", "centerline_ms", "ellipse_ms_median",
               "homotopy_verified", "station_order_ok", "accepted")


def git_commit(short: bool = True) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short" if short else "--verify", "HEAD"],
                             cwd=os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                             "..", "..", "..")),
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def scene_metadata(scene: Scene, cfg: Dict, seed: int, source_rel_path: str,
                   map_group: str, free_components: int) -> Dict[str, object]:
    """Everything needed to re-create this scene's labels bit for bit."""
    cc = cfg["centerline"]
    return {
        "label_version": str(cfg["project"].get("label_version", "v2")),
        "git_commit": git_commit(),
        "centerline_method": str(cc.get("method", "legacy_weighted")),
        "r_max_m": float(cc.get("r_max_m", float("nan"))),
        "d_safe_m": (float(cfg["robot"]["radius_m"]) + float(cfg["robot"]["safety_margin_m"])
                     + float(cc.get("numeric_margin_m", 0.01))),
        "segment_check_step_cells": float(cc.get("segment_check_step_cells", 0.25)),
        "homotopy_check_step_cells": float(cc.get("homotopy_check_step_cells", 0.25)),
        "w_clear": float(cc.get("w_clear", float("nan"))),
        "w_smooth": float(cc.get("w_smooth", float("nan"))),
        "w_pull": float(cc.get("w_pull", float("nan"))),
        "max_iter": int(cc.get("max_iter", 0)),
        "multi_start_count": int(cc.get("multi_start_count", 1)),
        "max_optimised_stations": cc.get("max_optimised_stations"),
        "seed": int(seed),
        "source_map_id": str(scene.source_map_id),
        "source_rel_path": str(source_rel_path),
        "map_group": str(map_group),
        "scene_id": str(scene.scene_id),
        "physical_field_m": float(scene.physical_field_m),
        "source_resolution_m": float(scene.source_resolution),
        "r_scene_m": float(scene.r_scene),
        "crop_px": int(scene.crop_px),
        "crop_origin": [int(v) for v in scene.crop_origin],
        "robot_radius_m": float(cfg["robot"]["radius_m"]),
        "safety_margin_m": float(cfg["robot"]["safety_margin_m"]),
        "edt_upsample": int(cfg["robot"]["edt_upsample"]),
        "density_inflated": float(scene.density_inflated),
        "density_raw": float(scene.density_raw),
        "free_components": int(free_components),
        "lidar_rays": int(cfg["lidar"]["ray_count"]),
        "lidar_max_range_m": float(cfg["lidar"]["max_range_m"]),
        "anchor_spacing": {k: float(v) for k, v in cfg["anchor"].items()},
        "rho_max": float(cfg["continuity"]["rho_max"]),
        "resample_spacing": {k: float(v) for k, v in cfg["resample"].items()},
    }


def save_scene_npz(res, scene: Scene, cfg: Dict, out_dir: str, meta: Dict[str, object],
                   prefix: str = "") -> str:
    """Write one scene: scene grid, every path polyline and every label row."""
    os.makedirs(out_dir, exist_ok=True)
    name = f"{prefix}{scene.scene_id}.npz"
    dst = os.path.join(out_dir, name)
    arrays: Dict[str, np.ndarray] = {
        "occupancy": scene.occupancy.astype(np.uint8),
        "blocked": scene.blocked.astype(np.uint8),
        "valid_mask": scene.valid_mask.astype(np.uint8),
        "dist_m": scene.dist_m.astype(np.float32),
        "meta_json": np.asarray(json.dumps(meta), dtype="U"),
    }
    # labels (same field names as the production pipeline)
    for k in LABEL_FIELDS:
        if k in res.labels:
            arrays[k] = np.asarray(res.labels[k])
    # paths, flattened with offsets so they stay retrievable by path_id
    paths: List[Dict[str, object]] = res.paths
    p_parts, q_parts, p_offsets, q_offsets = [], [], [0], [0]
    for rec in paths:
        p = np.asarray(rec.get("p_xy", np.zeros((0, 2))), dtype=np.float32).reshape(-1, 2)
        q = np.asarray(rec.get("q_xy", np.zeros((0, 2))), dtype=np.float32).reshape(-1, 2)
        p_parts.append(p)
        q_parts.append(q)
        p_offsets.append(p_offsets[-1] + len(p))
        q_offsets.append(q_offsets[-1] + len(q))
    # NOTE: a rejected path contributes an empty polyline.  Stacking must ignore
    # empty parts but keep the offsets aligned with the per-path lengths, otherwise
    # one rejected path 0 wipes out every polyline in the file.
    def _stack(parts):
        non_empty = [np.asarray(x, dtype=np.float32).reshape(-1, 2) for x in parts
                     if len(x)]
        return np.vstack(non_empty) if non_empty else np.zeros((0, 2), np.float32)

    arrays["path_p_xy"] = _stack(p_parts)
    arrays["path_q_xy"] = _stack(q_parts)
    arrays["path_p_offsets"] = np.asarray(p_offsets, dtype=np.int32)
    arrays["path_q_offsets"] = np.asarray(q_offsets, dtype=np.int32)
    if paths:
        for key in ("path_id", "n_stations", "n_anchors", "n_labels", "n_ellipse_safe",
                    "n_continuity_verified", "n_both_verified", "n_inserted", "length_m",
                    "clearance_p_mean", "clearance_q_mean", "clearance_gain_mean",
                    "max_offset_m", "length_ratio", "centerline_ms", "ellipse_ms_median"):
            vals = [rec.get(key) for rec in paths]
            arrays[f"path_{key}"] = np.asarray(
                [np.nan if v is None else v for v in vals], dtype=np.float64)
        for key in ("homotopy_verified", "station_order_ok", "accepted"):
            arrays[f"path_{key}"] = np.asarray([bool(rec.get(key, False)) for rec in paths],
                                               dtype=np.uint8)
        arrays["path_label_status"] = np.asarray(
            [str(rec.get("label_status", "")) for rec in paths], dtype="U32")
        arrays["path_status"] = np.asarray([str(rec.get("status", "")) for rec in paths],
                                           dtype="U64")
        arrays["path_start_xy"] = np.asarray([rec.get("start_xy", np.zeros(2)) for rec in paths],
                                             dtype=np.float32).reshape(-1, 2)
        arrays["path_goal_xy"] = np.asarray([rec.get("goal_xy", np.zeros(2)) for rec in paths],
                                            dtype=np.float32).reshape(-1, 2)
    np.savez_compressed(dst, **arrays)
    return dst


def load_scene_npz(path: str) -> Dict[str, object]:
    """Load a scene NPZ; ``meta`` is decoded back into a dict."""
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    if "meta_json" in out:
        raw = out["meta_json"]
        out["meta"] = json.loads(str(raw.item() if raw.ndim == 0 else raw))
    return out


def path_polyline(npz: Dict[str, object], path_id: int, which: str = "q") -> np.ndarray:
    """Retrieve one path polyline by ``path_id`` from a loaded NPZ."""
    off = npz[f"path_{which}_offsets"]
    xy = npz[f"path_{which}_xy"]
    ids = [int(v) for v in npz["path_path_id"]]
    k = ids.index(int(path_id))
    return xy[off[k]:off[k + 1]]


def labels_of_path(npz: Dict[str, object], path_id: int) -> Dict[str, np.ndarray]:
    """All label rows of one path (retrievable by scene_id + path_id)."""
    m = npz["path_id"] == int(path_id)
    return {k: npz[k][m] for k in LABEL_FIELDS if k in npz}


def write_index(out_dir: str, entries: List[Dict[str, object]], name: str = "index.json") -> str:
    dst = os.path.join(out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump({"n_files": len(entries), "entries": entries}, fh, indent=2)
    return dst
