"""Repair the scenes whose path polylines were lost by the stacking bug.

A rejected path 0 used to empty the whole ``path_p_xy`` / ``path_q_xy`` array.
This script finds such files, regenerates just those scenes (same map, crop and
field of view, deterministic seed from the scene id), then rebuilds index.json and
recomputes the dataset card directly from the NPZ files.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import zlib
from typing import Dict, List

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

import yaml  # noqa: E402

from pathsafe_ellipse.map_io import build_map_catalog, load_moving_ai_map, map_group_id  # noqa: E402
from pathsafe_ellipse.v2.pilot import load_scene_npz, save_scene_npz, scene_metadata  # noqa: E402
from pathsafe_ellipse.v2.pipeline import process_scene  # noqa: E402
from pathsafe_ellipse.v2.scene import build_scene, free_components  # noqa: E402


def load_cfg(path: str) -> Dict:
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for k, v in list(cfg["paths"].items()):
        if isinstance(v, str) and v.startswith("./"):
            cfg["paths"][k] = os.path.normpath(os.path.join(PROJECT_ROOT, v))
    return cfg


def parse_scene_id(scene_id: str):
    """``<map stem>_c<x0>_<y0>_f<field>_k<crop>`` -> parts."""
    body, k = scene_id.rsplit("_k", 1)
    body, f = body.rsplit("_f", 1)
    stem, c = body.rsplit("_c", 1)
    x0, y0 = c.split("_")
    return stem, int(x0), int(y0), float(f), int(k)


def broken_files(root: str) -> List[str]:
    out = []
    for p in sorted(glob.glob(os.path.join(root, "*.npz"))):
        d = load_scene_npz(p)
        n_labels = int(d["path_id"].shape[0])
        n_poly = int(d["path_q_xy"].shape[0]) if "path_q_xy" in d else 0
        n_paths = int(len(d["path_path_id"])) if "path_path_id" in d else 0
        if n_paths and n_labels and n_poly < 2:
            out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dataset_v21_xl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg(os.path.join(PROJECT_ROOT, "configs", "data_generation_v2_1.yaml"))
    root = os.path.join(cfg["paths"]["v2_root"], f"pilot_{args.tag}", "v2_1")
    todo = broken_files(root)
    print(f"broken scene files: {len(todo)}")
    if args.dry_run:
        return 0
    records = {os.path.splitext(r.file_name)[0]: r
               for r in build_map_catalog(cfg["paths"]["map_dir"], 0.25) if r.parsed_ok}
    for p in todo:
        scene_id = os.path.splitext(os.path.basename(p))[0]
        stem, x0, y0, field_m, k = parse_scene_id(scene_id)
        rec = records.get(stem)
        if rec is None:
            print(f"  !! map not found for {scene_id}")
            continue
        src, smeta = load_moving_ai_map(os.path.join(cfg["paths"]["map_dir"], rec.rel_path))
        r0 = float(smeta["resolution"] or 0.25)
        crop_px = int(round(field_m / r0))
        seed = zlib.crc32(scene_id.encode()) % 100000
        scene = build_scene(src, r0, (x0, y0), crop_px, int(cfg["scene"]["size"]), field_m,
                            float(cfg["robot"]["radius_m"]),
                            float(cfg["robot"]["safety_margin_m"]),
                            int(cfg["robot"]["edt_upsample"]), scene_id, rec.file_name)
        res = process_scene(scene, cfg, np.random.default_rng(seed))
        _lab, n_comp, largest = free_components(scene.free)
        meta = scene_metadata(scene, cfg, seed, rec.rel_path, map_group_id(rec.file_name), n_comp)
        meta.update({"variant": "v21", "largest_component_cells": int(largest),
                     "repaired": True})
        save_scene_npz(res, scene, cfg, root, meta, prefix="")
        print(f"  repaired {scene_id}: labels {res.stats['n_anchors']}, "
              f"paths {res.stats['n_paths_accepted']}/{res.stats['n_paths']}")

    # ---- rebuild index.json from the directory
    entries = []
    for p in sorted(glob.glob(os.path.join(root, "*.npz"))):
        d = load_scene_npz(p)
        entries.append({"scene_id": str(d["meta"]["scene_id"]), "variant": "v21",
                        "dir": "v2_1", "npz": os.path.basename(p),
                        "n_paths": int(len(d["path_path_id"])) if "path_path_id" in d else 0,
                        "n_anchors": int(d["path_id"].shape[0])})
    with open(os.path.join(os.path.dirname(root), "index.json"), "w", encoding="utf-8") as fh:
        json.dump({"n_files": len(entries), "entries": entries}, fh, indent=2)
    print(f"index rebuilt: {len(entries)} entries")

    # ---- dataset card straight from the NPZ files
    tot = {k: 0 for k in ("scenes", "paths", "acc", "fail", "labels", "safe", "cont", "both",
                          "conv", "fnc", "fb", "failed")}
    areas, dens, fields, groups, cl_ms, el_ms = [], [], {}, set(), [], []
    for p in sorted(glob.glob(os.path.join(root, "*.npz"))):
        d = load_scene_npz(p)
        meta = d["meta"]
        tot["scenes"] += 1
        groups.add(meta["map_group"])
        fields[int(meta["physical_field_m"])] = fields.get(int(meta["physical_field_m"]), 0) + 1
        dens.append(meta["density_inflated"])
        n = int(d["path_id"].shape[0])
        tot["labels"] += n
        tot["safe"] += int(d["safety_verified"].sum())
        tot["cont"] += int(d["continuity_verified"].sum())
        tot["both"] += int(d["label_valid"].sum())
        if n:
            areas.append(d["ellipse_area"])
        if "path_path_id" in d:
            tot["paths"] += int(len(d["path_path_id"]))
            tot["acc"] += int(d["path_accepted"].sum()) if "path_accepted" in d else 0
            tot["fail"] += int(len(d["path_path_id"]) - d["path_accepted"].sum())
            for st in d["path_label_status"]:
                s = str(st)
                if s == "optimal_safe":
                    tot["conv"] += 1
                elif s == "feasible_not_converged":
                    tot["fnc"] += 1
                elif s == "fallback_astar":
                    tot["fb"] += 1
                elif s in ("failed", "infeasible"):
                    tot["failed"] += 1
            if "path_centerline_ms" in d:
                cl_ms.extend([v for v in np.asarray(d["path_centerline_ms"]).ravel().tolist()
                              if np.isfinite(v)])
            if "path_ellipse_ms_median" in d:
                el_ms.extend([v for v in np.asarray(d["path_ellipse_ms_median"]).ravel().tolist()
                              if np.isfinite(v)])
    areas = np.concatenate(areas) if areas else np.zeros(0)
    card = {
        "tag": args.tag, "n_scenes": tot["scenes"], "n_map_groups": len(groups),
        "field_histogram": fields, "paths": tot["paths"], "paths_accepted": tot["acc"],
        "paths_failed": tot["fail"], "labels": tot["labels"],
        "labels_safe": tot["safe"], "labels_continuity": tot["cont"],
        "labels_both": tot["both"], "labels_invalid": tot["labels"] - tot["both"],
        "solver_converged": tot["conv"], "feasible_not_converged": tot["fnc"],
        "fallbacks": tot["fb"], "failed_paths": tot["failed"],
        "density_p50": float(np.median(dens)), "density_max": float(max(dens)),
        "area_p50": float(np.median(areas)) if areas.size else None,
        "area_min": float(areas.min()) if areas.size else None,
        "area_max": float(areas.max()) if areas.size else None,
        "valid_label_rate": tot["both"] / max(tot["labels"], 1),
        "acceptance_rate": tot["acc"] / max(tot["paths"], 1),
        "centreline_ms_p50": float(np.median(cl_ms)) if cl_ms else None,
        "ellipse_ms_p50": float(np.median(el_ms)) if el_ms else None,
        "repaired_from_shards": True,
    }
    dst = os.path.join(PROJECT_ROOT, "reports_v2_1", f"dataset_card_{args.tag}.json")
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(card, fh, indent=2)
    for k, v in card.items():
        print(f"  {k:24s} {v}")
    print(f"-> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
