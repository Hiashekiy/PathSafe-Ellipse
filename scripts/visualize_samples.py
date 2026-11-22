"""Step S6 visualisation: map overviews, sample-level ellipses, path-level coverage."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pathsafe_ellipse.candidate_centers import generate_candidates  # noqa: E402
from pathsafe_ellipse.config import load_config, project_paths, setup_logging  # noqa: E402
from pathsafe_ellipse.local_features import PatchFeatures, patch_clearance, ray_directions  # noqa: E402
from pathsafe_ellipse.map_io import build_map_catalog, map_key  # noqa: E402
from pathsafe_ellipse.map_preprocess import preprocess_map  # noqa: E402
from pathsafe_ellipse.visualization import (  # noqa: E402
    plot_map_overview, plot_patch_with_rays, plot_path_ellipses, plot_sample_ellipse,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--n-maps", type=int, default=4)
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--n-paths", type=int, default=3)
    p.add_argument("--tag", default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg, "visualize_samples")
    pp = project_paths(cfg)
    res = float(cfg["map"]["resolution"])
    ps = int(cfg["local_features"]["patch_size"])

    records = {map_key(r): r for r in build_map_catalog(pp.map_dir, cfg["map"]["resolution"])
               if r.parsed_ok}
    available = sorted(k for k in records
                       if os.path.exists(os.path.join(pp.samples_dir, f"{k}.npz")))
    if not available:
        print("no per-map sample files found")
        return 2
    rng = np.random.default_rng(int(cfg["project"]["random_seed"]))

    picks = [available[i] for i in rng.permutation(len(available))[: args.n_maps]]
    print(f"maps for overview: {picks}")
    for key in picks:
        inflated = preprocess_map(records[key], pp.map_dir, pp.inflated_maps_dir,
                                  robot_radius=float(cfg["robot"]["radius"]),
                                  kernel_radius=int(cfg["robot"]["inflation_kernel"]),
                                  iterations=int(cfg["robot"]["inflation_iterations"]))
        paths_npz = np.load(os.path.join(pp.paths_dir, f"{key}.npz"))
        flat, offs = paths_npz["flat_paths"], paths_npz["path_offsets"]
        paths = [flat[offs[i]:offs[i + 1]] for i in range(len(offs) - 1)]
        plot_map_overview(
            inflated.occupancy_raw, inflated.occupancy_inflated, paths,
            os.path.join(pp.previews_dir, f"map_overview_{key}.png"),
            title=f"{key}  ({len(paths)} A* paths)", clearance=inflated.clearance,
        )

    # ---------------------------------------------------------------- samples
    print("sample-level figures")
    shown = 0
    for key in picks:
        data = np.load(os.path.join(pp.samples_dir, f"{key}.npz"))
        n = len(data["label_valid"])
        if n == 0:
            continue
        order = rng.permutation(n)[: max(1, args.n_samples // max(len(picks), 1))]
        for idx in order:
            patch = data["map_patch"][idx]
            raw_patch = data["map_patch_raw"][idx]
            feats = PatchFeatures(
                patch=patch,
                clearance=patch_clearance(patch),
                ray_directions=ray_directions(int(cfg["local_features"]["ray_count"])),
                ray_distances=data["ray_distances"][idx],
                ray_hits=data["ray_hits"][idx],
                ray_distances_raw=data["ray_distances_raw"][idx],
                local_xy=(64.0, 64.0),
            )
            cc = cfg["candidate_centers"]
            cand_set = generate_candidates(
                feats, offset_ratios=[float(r) for r in cc["offset_ratios"]],
                ray_margin=float(cc["ray_margin"]), dedupe_distance=float(cc["dedupe_distance"]),
                min_center_clearance=float(cc["min_center_clearance"]),
                segment_clearance=float(cc["segment_clearance"]), patch_size=ps,
                ray_max_range=float(cfg["local_features"]["ray_max_range"]))
            ab = data["ellipse_ab"][idx]
            ang = data["ellipse_angle"][idx]
            angle_rad = float(np.arctan2(ang[0], ang[1]))
            offset_m = data["ellipse_center_offset"][idx]
            chosen_center = (64.0 + offset_m[0] / res, 64.0 + offset_m[1] / res)
            label = {
                "offset_direction": int(data["offset_direction"][idx]),
                "offset_ratio": float(data["offset_ratio"][idx]),
                "point_inside_metric": float(data["point_inside_metric"][idx]),
            }
            out = os.path.join(pp.ellipse_viz_dir, f"sample_{key}_{idx:05d}.png")
            plot_sample_ellipse(
                patch, feats.ray_directions, feats.ray_distances_raw, feats.ray_hits,
                [(c.center_xy, c.offset_direction, c.offset_ratio) for c in cand_set.candidates],
                chosen_center, (float(ab[0]), float(ab[1]), angle_rad), label, res, out,
                title=f"{key} sample {idx}",
            )
            plot_patch_with_rays(
                patch, feats.ray_directions, feats.ray_distances_raw, feats.ray_hits,
                (64.0, 64.0), os.path.join(pp.previews_dir, f"patch_rays_{key}_{idx:05d}.png"),
                title=f"{key} sample {idx} rays", raw_patch=raw_patch,
            )
            shown += 1
    print(f"  wrote {shown} sample figures -> {pp.ellipse_viz_dir}")

    # ------------------------------------------------------------- path level
    print("path-level figures")
    for key in picks[: max(1, len(picks) // 2)]:
        data = np.load(os.path.join(pp.samples_dir, f"{key}.npz"))
        if len(data["label_valid"]) == 0:
            continue
        paths_npz = np.load(os.path.join(pp.paths_dir, f"{key}.npz"))
        flat, offs = paths_npz["flat_paths"], paths_npz["path_offsets"]
        inflated = preprocess_map(records[key], pp.map_dir, pp.inflated_maps_dir,
                                  robot_radius=float(cfg["robot"]["radius"]),
                                  kernel_radius=int(cfg["robot"]["inflation_kernel"]),
                                  iterations=int(cfg["robot"]["inflation_iterations"]))
        by_path: Dict[int, List[int]] = {}
        for i, pid in enumerate(data["path_id"]):
            by_path.setdefault(int(pid), []).append(i)
        chosen_paths = sorted(by_path, key=lambda p: -len(by_path[p]))[: args.n_paths]
        for pid in chosen_paths:
            idxs = by_path[pid]
            path = flat[offs[pid]:offs[pid + 1]].astype(float)
            ellipses = []
            for i in idxs:
                ab = data["ellipse_ab"][i]
                ang = data["ellipse_angle"][i]
                pt = data["point_xy"][i]
                off = data["ellipse_center_offset"][i]
                ellipses.append(((pt[0] + off[0] / res, pt[1] + off[1] / res),
                                 float(ab[0]), float(ab[1]),
                                 float(np.arctan2(ang[0], ang[1]))))
            plot_path_ellipses(
                inflated.occupancy_inflated, path, ellipses,
                os.path.join(pp.path_viz_dir, f"path_ellipses_{key}_{pid:03d}.png"),
                title=f"{key} path {pid} ({len(idxs)} labelled points)",
                resolution=res,
            )
    print(f"  wrote path figures -> {pp.path_viz_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
