"""V2 pilot: sample whole scenes, run the V2 label pipeline and render figures.

Figures (per scene, plus per-path detail):
  * scene overview: inflated occupancy + distance field + A* path + optimised
    centreline (p -> q displacement) + start/goal
  * path detail: ellipses along one path with the anchor points p_i and the
    label centres q_i, plus the active half-space lines of one ellipse
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from pathsafe_ellipse.map_io import build_map_catalog, load_moving_ai_map, map_group_id  # noqa: E402
from pathsafe_ellipse.v2.pipeline import process_scene  # noqa: E402
from pathsafe_ellipse.v2.scene import Scene, build_scene, free_components  # noqa: E402


def load_cfg(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for key in ("map_dir", "v2_root", "scene_dir", "label_dir", "report_dir", "viz_dir"):
        p = cfg["paths"][key]
        if not os.path.isabs(p):
            cfg["paths"][key] = os.path.normpath(os.path.join(PROJECT_ROOT, p))
    return cfg


def plot_scene(res, out_path: str, title: str) -> None:
    sc = res.scene
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
    ext = [0, sc.size * sc.r_scene, sc.size * sc.r_scene, 0]

    axes[0].imshow(sc.occupancy, cmap="gray_r", interpolation="nearest", extent=ext)
    axes[0].set_title(f"raw occupancy  (density {sc.density_raw:.3f})")
    im = axes[1].imshow(sc.dist_m, cmap="viridis", interpolation="nearest", extent=ext)
    axes[1].imshow(sc.blocked, cmap="gray_r", alpha=0.35, interpolation="nearest", extent=ext)
    axes[1].set_title("distance field D(x) [m] + inflated blocked cells")
    fig.colorbar(im, ax=axes[1], fraction=0.046, label="m")

    ax = axes[2]
    ax.imshow(sc.blocked, cmap="gray_r", alpha=0.55, interpolation="nearest", extent=ext)
    for rec in res.paths:
        p, q = rec["p_xy"], rec["q_xy"]
        ax.plot(p[:, 0], p[:, 1], "-", lw=1.0, color="tab:orange", alpha=0.9,
                label="A* (resampled)" if rec is res.paths[0] else None)
        ax.plot(q[:, 0], q[:, 1], "-", lw=1.3, color="tab:blue", alpha=0.9,
                label="distance-field centreline" if rec is res.paths[0] else None)
        ax.plot(p[0, 0], p[0, 1], "o", ms=5, color="lime", mec="k")
        ax.plot(p[-1, 0], p[-1, 1], "X", ms=7, color="red", mec="k")
    ax.set_title(f"A* vs optimised centreline  ({len(res.paths)} paths, "
                 f"FOV {sc.physical_field_m:.0f} m, r_scene {sc.r_scene:.4f} m)")
    ax.legend(loc="upper right", fontsize=8)
    for a in axes:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=115)
    plt.close(fig)


def plot_path(res, path_id: int, out_path: str, title: str) -> None:
    sc = res.scene
    rec = next((r for r in res.paths if r["path_id"] == path_id), None)
    if rec is None:
        return
    lab = res.labels
    sel = np.flatnonzero(lab["path_id"] == path_id)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.0))

    ext = [0, sc.size * sc.r_scene, sc.size * sc.r_scene, 0]
    ax = axes[0]
    ax.imshow(sc.blocked, cmap="gray_r", alpha=0.55, interpolation="nearest", extent=ext)
    p, q = rec["p_xy"], rec["q_xy"]
    ax.plot(p[:, 0], p[:, 1], "-", lw=1.2, color="tab:orange", label="A* p_i")
    ax.plot(q[:, 0], q[:, 1], "-", lw=1.0, color="tab:blue", alpha=0.8, label="centreline q_i")
    for k in sel:
        S = lab["ellipse_S"][k].reshape(2, 2)
        ab = lab["ellipse_ab"][k]
        ang = float(lab["ellipse_angle"][k])
        c = lab["q_xy"][k]
        ax.add_patch(Ellipse((c[0], c[1]), 2 * ab[0], 2 * ab[1], angle=np.degrees(ang),
                             fill=False, ec="tab:green" if lab["safety_verified"][k] else "red",
                             lw=1.0, alpha=0.9))
        ax.plot(lab["p_xy"][k][0], lab["p_xy"][k][1], ".", ms=2.4, color="tab:red")
        ax.plot(c[0], c[1], ".", ms=2.0, color="tab:blue")
    ax.set_title(f"path {path_id}: {len(sel)} anchors  "
                 f"(green = verified safe ellipse, blue dot = q_i, red dot = p_i)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])

    # zoom on one anchor with its active half-spaces
    ax = axes[1]
    if len(sel):
        k = int(sel[len(sel) // 2])
        c = lab["q_xy"][k]
        S = lab["ellipse_S"][k].reshape(2, 2)
        ab = lab["ellipse_ab"][k]
        ang = float(lab["ellipse_angle"][k])
        win = max(2.0, 2.2 * float(ab[0]))
        x0, x1 = max(0, c[0] - win), min(sc.size * sc.r_scene, c[0] + win)
        y0, y1 = max(0, c[1] - win), min(sc.size * sc.r_scene, c[1] + win)
        ax.imshow(sc.blocked, cmap="gray_r", alpha=0.6, interpolation="nearest", extent=ext)
        ax.add_patch(Ellipse((c[0], c[1]), 2 * ab[0], 2 * ab[1], angle=np.degrees(ang),
                             fill=False, ec="tab:blue", lw=1.8))
        ax.plot(c[0], c[1], "*", ms=10, color="tab:blue")
        ax.plot(lab["p_xy"][k][0], lab["p_xy"][k][1], "+", ms=10, color="tab:red", mew=2)
        hs = getattr(res, "halfspaces", {}).get(int(k), [])
        for h in hs[:40]:
            n, d = h["n"], h["d"]
            # line n.(x-c) = d
            if abs(n[1]) > 1e-9:
                xs = np.array([c[0] - win, c[0] + win])
                ys = c[1] + (d - n[0] * (xs - c[0])) / n[1]
            else:
                xs = np.array([c[0] + d / n[0], c[0] + d / n[0]])
                ys = np.array([c[1] - win, c[1] + win])
            ax.plot(xs, ys, "-", lw=0.6, color="tab:purple", alpha=0.55)
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
        ax.set_title(f"anchor {k}: a={ab[0]:.2f} m b={ab[1]:.2f} m, area={lab['ellipse_area'][k]:.2f} m²\n"
                     f"active half-spaces shown; D(q)={lab['clearance'][k]:.2f} m, "
                     f"status={lab['solver_status'][k]}, {lab['generation_time_ms'][k]:.1f} ms")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=115)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "data_generation_v2.yaml"))
    ap.add_argument("--n-scenes", type=int, default=4)
    ap.add_argument("--paths-per-scene", type=int, default=2)
    ap.add_argument("--n-plot-paths", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    cfg["path"]["paths_per_scene"] = args.paths_per_scene
    for key in ("v2_root", "scene_dir", "label_dir", "report_dir", "viz_dir"):
        os.makedirs(cfg["paths"][key], exist_ok=True)

    records: List = build_map_catalog(cfg["paths"]["map_dir"], cfg["scene"]["source_resolution"]
                                      if cfg["scene"]["source_resolution"] != "auto" else 0.25)
    records = [r for r in records if r.parsed_ok]
    rng = np.random.default_rng(args.seed)

    fields = cfg["scene"]["physical_fields_m"]
    scene_size = int(cfg["scene"]["size"])
    results = []
    t_start = time.perf_counter()

    for i in range(args.n_scenes):
        rec = records[int(rng.integers(0, len(records)))]
        field_m = float(fields[i % len(fields)])
        src, meta = load_moving_ai_map(os.path.join(cfg["paths"]["map_dir"], rec.rel_path))
        r0 = float(meta["resolution"] or 0.25)
        crop_px = int(round(field_m / r0))
        h, w = src.shape
        if crop_px >= h or crop_px >= w:
            print(f"  skip {rec.file_name}: map {w}x{h} smaller than crop {crop_px}")
            continue
        x0 = int(rng.integers(0, w - crop_px))
        y0 = int(rng.integers(0, h - crop_px))
        scene_id = f"{os.path.splitext(rec.file_name)[0]}_crop{x0}_{y0}_f{int(field_m)}"
        sc = build_scene(src, r0, (x0, y0), crop_px, scene_size, field_m,
                         float(cfg["robot"]["radius_m"]), float(cfg["robot"]["safety_margin_m"]),
                         int(cfg["robot"]["edt_upsample"]), scene_id, rec.file_name)
        lab_c, n_c, largest = free_components(sc.free)
        res = process_scene(sc, cfg, rng)
        res.stats.update({"source_rel_path": rec.rel_path, "map_group": map_group_id(rec.file_name),
                          "free_components": n_c, "largest_component_cells": largest,
                          "label_version": "v2"})
        results.append(res)
        print(f"  [{i+1}/{args.n_scenes}] {scene_id}: FOV {field_m:.0f} m, r_scene {sc.r_scene:.4f}, "
              f"occ {sc.density_inflated:.3f}, paths {len(res.paths)}, anchors {res.stats['n_anchors']}, "
              f"inserted {res.stats['n_inserted']}, statuses {res.stats['statuses']}, "
              f"ellipse ms p50 {res.stats['solve_ms_median']}")
        plot_scene(res, os.path.join(cfg["paths"]["viz_dir"], f"scene_{i:02d}.png"),
                   f"V2 scene {i}: {scene_id}")
        for j, prec in enumerate(res.paths[: args.n_plot_paths]):
            plot_path(res, prec["path_id"],
                      os.path.join(cfg["paths"]["viz_dir"], f"scene_{i:02d}_path{j}.png"),
                      f"V2 {scene_id} path {prec['path_id']}")

    # ---- summary
    tot_anchor = sum(r.stats["n_anchors"] for r in results)
    ok = sum(1 for r in results for s in [r.labels["safety_verified"]] for v in s if v)
    ms = [r.stats["solve_ms_median"] for r in results if r.stats["solve_ms_median"]]
    summary = {
        "n_scenes": len(results),
        "n_anchors": tot_anchor,
        "n_verified_safe": int(ok),
        "ellipse_ms_median_of_medians": float(np.median(ms)) if ms else None,
        "total_s": round(time.perf_counter() - t_start, 2),
        "scenes": [r.stats for r in results],
    }
    out = os.path.join(cfg["paths"]["report_dir"], "v2_pilot_summary.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(json.dumps({k: v for k, v in summary.items() if k != "scenes"}, indent=2))
    print(f"summary -> {out}")
    print(f"figures -> {cfg['paths']['viz_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
