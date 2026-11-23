"""Visualise a V3 trajectory dataset: overview, per-trajectory detail, dashboard.

    python scripts/viz_trajectory_v3.py --dir data/processed_v3/trajectories/pilot10 \
        --tag pilot10

Writes into ``outputs_v3/viz/<tag>/``:

    overview_<tag>.png        one panel per trajectory (map, path, centre line,
                              query points, LiDAR rays, ellipse labels)
    detail_<file>.png         full view + zoom + LiDAR heatmap + per-point curves
    dashboard_<tag>.png       dataset-level distributions and verification rates
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from pathsafe_ellipse.v2.dataset_v3 import load_trajectory, summarise_files  # noqa: E402

OBSTACLE_BG = "#7a7a7a"


# --------------------------------------------------------------------------- #
# Shared drawing
# --------------------------------------------------------------------------- #
def draw_map(ax, sample: Dict[str, object], lidar: bool = True) -> Dict[str, float]:
    occ = np.asarray(sample["occupancy"])
    rs = float(sample["meta"]["r_scene"])
    ext = [0.0, occ.shape[1] * rs, occ.shape[0] * rs, 0.0]      # y grows downwards
    ax.imshow(occ, cmap="gray_r", alpha=0.55, interpolation="nearest", extent=ext)
    p = np.asarray(sample["path_xy"], float)
    rng_ = np.asarray(sample["lidar_ranges"], float)
    if lidar:
        angles = 2.0 * np.pi * np.arange(rng_.shape[1]) / rng_.shape[1]
        ends = p[:, None, :] + rng_[:, :, None] * np.stack(
            [np.cos(angles), np.sin(angles)], axis=1)[None, :, :]
        segs = np.stack([np.repeat(p[:, None, :], rng_.shape[1], axis=1), ends], axis=2)
        ax.add_collection(LineCollection(segs.reshape(-1, 2, 2), colors="tab:cyan",
                                         linewidths=0.25, alpha=0.35, zorder=1))
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    return {"x0": ext[0], "x1": ext[1], "y0": ext[3], "y1": ext[2], "rs": rs}


def draw_labels(ax, sample: Dict[str, object]) -> None:
    q = np.asarray(sample["centerline_xy"], float)
    ab = np.asarray(sample["ellipse_ab"], float)
    ang = np.asarray(sample["ellipse_angle"], float)
    valid = np.asarray(sample["label_valid"]).astype(bool)
    for i in range(len(q)):
        ax.add_patch(Ellipse((q[i, 0], q[i, 1]), 2 * ab[i, 0], 2 * ab[i, 1],
                             angle=np.degrees(float(ang[i])), fill=False,
                             ec="tab:green" if valid[i] else "tab:red",
                             lw=0.65, alpha=0.85, zorder=5))
    p = np.asarray(sample["path_xy"], float)
    ax.plot(p[:, 0], p[:, 1], "-", lw=1.1, color="tab:orange", zorder=3)
    ax.plot(p[:, 0], p[:, 1], "o", ms=2.4, color="tab:orange", zorder=4)
    ax.plot(q[:, 0], q[:, 1], "-", lw=1.2, color="tab:blue", alpha=0.9, zorder=4)


def short(name: str, n: int = 44) -> str:
    return name if len(name) <= n else name[: n - 1] + "…"


# --------------------------------------------------------------------------- #
# Overview: one panel per trajectory
# --------------------------------------------------------------------------- #
def plot_overview(files: List[str], dst: str, n_show: int = 10) -> None:
    files = files[:n_show]
    ncol = 4 if len(files) > 6 else 3
    nrow = int(np.ceil(len(files) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 4.3 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, path in zip(axes, files):
        s = load_trajectory(path)
        draw_map(ax, s)
        draw_labels(ax, s)
        n = len(s["path_xy"])
        nv = int(np.asarray(s["label_valid"]).sum())
        ax.set_title(f"{s['meta']['scene_id'].split('_c')[0]}  p{s['meta']['path_id']}  "
                     f"{s['meta']['physical_field_m']:.0f} m  N={n} (valid {nv})", fontsize=7.5)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("V3 trajectory samples — inflated occupancy, A* path + query points "
                 "(orange), optimised centre line (blue), 16-ray LiDAR (cyan), "
                 "ellipse labels (green)", fontsize=11)
    fig.tight_layout()
    fig.savefig(dst, dpi=115)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Per-trajectory detail
# --------------------------------------------------------------------------- #
def plot_detail(sample: Dict[str, object], dst: str, name: str, zoom_m: float = 7.0) -> None:
    meta = sample["meta"]
    p = np.asarray(sample["path_xy"], float)
    q = np.asarray(sample["centerline_xy"], float)
    off = np.asarray(sample["center_offset"], float)
    ab = np.asarray(sample["ellipse_ab"], float)
    ang = np.asarray(sample["ellipse_angle"], float)
    area = np.asarray(sample["ellipse_area"], float)
    S = np.asarray(sample["ellipse_S"], float)
    lid = np.asarray(sample["lidar_ranges"], float)
    safety = np.asarray(sample["safety_verified"]).astype(bool)
    cont = np.asarray(sample["continuity_verified"]).astype(bool)
    valid = np.asarray(sample["label_valid"]).astype(bool)
    n = len(p)
    idx = np.arange(n)

    fig = plt.figure(figsize=(17.0, 10.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], hspace=0.30, wspace=0.40)

    # --- (0,0) whole scene
    ax = fig.add_subplot(gs[0, 0])
    draw_map(ax, sample)
    draw_labels(ax, sample)
    ax.plot(*np.asarray(sample["start_xy"], float), "s", ms=6, mfc="lime", mec="k", zorder=6)
    ax.plot(*np.asarray(sample["goal_xy"], float), "^", ms=7, mfc="magenta", mec="k", zorder=6)
    ax.set_title(f"{name}\n(map {meta['physical_field_m']:.0f} m, r_cell {meta['r_scene']:.4f} m, "
                 f"N={n})", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("whole 256x256 scene", fontsize=8)

    # --- (0,1) zoom on the largest ellipse (most informative sample point)
    k = int(np.argmax(area))
    ax = fig.add_subplot(gs[0, 1])
    draw_map(ax, sample)
    draw_labels(ax, sample)
    ax.set_xlim(q[k, 0] - zoom_m, q[k, 0] + zoom_m)
    ax.set_ylim(q[k, 1] + zoom_m, q[k, 1] - zoom_m)
    for i in range(n):
        ax.annotate(str(i), (p[i, 0], p[i, 1]), fontsize=5.5, color="k",
                    xytext=(2, 2), textcoords="offset points", zorder=7)
    ax.plot(q[k, 0], q[k, 1], "+", ms=10, color="k", zorder=8)
    ax.set_title(f"zoom on query point {k}: a={ab[k, 0]:.2f} m, b={ab[k, 1]:.2f} m, "
                 f"area={area[k]:.1f} m², |offset|={np.linalg.norm(off[k]):.2f} m", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])

    # --- (0,2) LiDAR heatmap (ray x query point)
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(lid.T, aspect="auto", origin="lower", cmap="viridis",
                   extent=[-0.5, n - 0.5, -0.5, lid.shape[1] - 0.5], vmin=0.0,
                   vmax=float(meta["lidar_max_range_m"]))
    ax.set_xlabel("query point index (aligned with path_xy)")
    ax.set_ylabel("LiDAR ray (16, 360°)")
    ax.set_title(f"lidar_ranges (N={n} x 16)  from path_xy, max {meta['lidar_max_range_m']:.1f} m",
                 fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="range [m]")

    # --- (1,0) offset + area
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(idx, np.linalg.norm(off, axis=1), "-o", ms=2.5, color="tab:purple",
            label="|centerline - path| [m]")
    ax.set_xlabel("query point index"); ax.set_ylabel("[m]", color="tab:purple")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(idx, area, "-", lw=1.0, color="tab:green", alpha=0.7, label="ellipse area [m²]")
    ax2.set_ylabel("area [m²]", color="tab:green")
    ax.set_title("centre offset and ellipse area per query point", fontsize=8)
    bad = idx[~valid]
    for i in bad:
        ax.axvspan(i - 0.4, i + 0.4, color="red", alpha=0.15)

    # --- (1,1) semi-axes + angle
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(idx, ab[:, 0], "-o", ms=2.2, color="tab:red", label="semi-major a")
    ax.plot(idx, ab[:, 1], "-o", ms=2.2, color="tab:blue", label="semi-minor b")
    ax.set_xlabel("query point index"); ax.set_ylabel("semi-axis [m]")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(idx, np.degrees(ang), "--", lw=0.9, color="k", alpha=0.55, label="angle")
    ax2.set_ylabel("angle [deg]", fontsize=8)
    ax.set_title("ellipse_ab and ellipse_angle", fontsize=8)

    # --- (1,2) shape matrix + verification flags
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(idx, S[:, 0, 0], "-", lw=1.1, color="tab:orange", label="S_xx")
    ax.plot(idx, S[:, 1, 1], "-", lw=1.1, color="tab:blue", label="S_yy")
    ax.plot(idx, S[:, 0, 1], "-", lw=1.0, color="tab:gray", label="S_xy")
    ax.set_xlabel("query point index"); ax.set_ylabel("S [m²]")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="upper left")
    ax.set_title("ellipse shape matrix; flags: "
                 f"safety {safety.mean():.2f} / continuity {cont.mean():.2f} / "
                 f"valid {valid.mean():.2f}", fontsize=8)
    for i in idx:
        if not safety[i]:
            ax.axvspan(i - 0.4, i + 0.4, color="tab:red", alpha=0.12)
        elif not cont[i]:
            ax.axvspan(i - 0.4, i + 0.4, color="tab:orange", alpha=0.12)
    fig.suptitle(f"{name} — one NPZ = one complete trajectory "
                 f"(scene {meta['scene_id']}, path {meta['path_id']}, "
                 f"{meta['label_version']})", fontsize=12)
    fig.savefig(dst, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Dataset dashboard
# --------------------------------------------------------------------------- #
def plot_dashboard(files: List[str], dst: str, tag: str) -> Dict[str, object]:
    stats = summarise_files(files)
    ns, areas, a_axis, b_axis, offsets, lidars = [], [], [], [], [], []
    safety = cont = valid = 0
    total = 0
    for path in files:
        s = load_trajectory(path)
        p = np.asarray(s["path_xy"], float)
        q = np.asarray(s["centerline_xy"], float)
        ab = np.asarray(s["ellipse_ab"], float)
        ns.append(len(p))
        areas.append(np.asarray(s["ellipse_area"], float))
        a_axis.append(ab[:, 0]); b_axis.append(ab[:, 1])
        offsets.append(np.linalg.norm(q - p, axis=1))
        lidars.append(np.asarray(s["lidar_ranges"], float))
        safety += int(np.asarray(s["safety_verified"]).sum())
        cont += int(np.asarray(s["continuity_verified"]).sum())
        valid += int(np.asarray(s["label_valid"]).sum())
        total += len(p)
    area = np.concatenate(areas); aa = np.concatenate(a_axis); bb = np.concatenate(b_axis)
    off = np.concatenate(offsets); lid = np.concatenate(lidars)

    fig, axes = plt.subplots(2, 4, figsize=(19.0, 8.4))
    ax = axes[0, 0]
    ax.hist(np.asarray(ns), bins=min(10, len(ns)), color="tab:blue", alpha=0.8)
    ax.set_xlabel("query points N per trajectory"); ax.set_ylabel("trajectories")
    ax.set_title(f"N distribution (mean {np.mean(ns):.1f}, "
                 f"min {min(ns)}, max {max(ns)})", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.hist(area, bins=30, color="tab:green", alpha=0.8)
    ax.set_xlabel("ellipse area [m²]"); ax.set_ylabel("query points")
    ax.set_title(f"ellipse_area (p50 {np.median(area):.1f} m²)", fontsize=9)
    ax.set_xscale("log"); ax.grid(alpha=0.3)

    ax = axes[0, 2]
    lim = max(float(aa.max()), float(bb.max())) * 1.05
    sc = ax.scatter(aa, bb, c=area, s=10, cmap="viridis", alpha=0.85, norm=LogNorm())
    ax.plot([0, lim], [0, lim], "--", lw=0.8, color="k", alpha=0.5, label="circle (a=b)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("semi-major a [m]"); ax.set_ylabel("semi-minor b [m]")
    ax.set_title("semi-axes (colour = area)", fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)

    ax = axes[0, 3]
    ax.hist(off, bins=30, color="tab:purple", alpha=0.8)
    ax.set_xlabel("|centerline - path| [m]"); ax.set_ylabel("query points")
    ax.set_title(f"centre offset (p50 {np.median(off):.2f} m, "
                 f"p95 {np.percentile(off, 95):.2f} m)", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.imshow(lid.T, aspect="auto", origin="lower", cmap="viridis",
              extent=[-0.5, len(lid) - 0.5, -0.5, lid.shape[1] - 0.5],
              vmax=float(np.max(lid)))
    ax.set_xlabel("all query points (concatenated)"); ax.set_ylabel("ray")
    ax.set_title(f"LiDAR ranges (mean {lid.mean():.2f} m, "
                 f"p05 {np.percentile(lid, 5):.2f} m)", fontsize=9)

    ax = axes[1, 1]
    ax.hist(lid.ravel(), bins=40, color="tab:cyan", alpha=0.85)
    ax.set_xlabel("range [m]"); ax.set_ylabel("rays")
    ax.set_title("LiDAR range distribution", fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1, 2]
    rays = np.arange(lid.shape[1])
    ray_mean = lid.mean(axis=0)
    ray_p05 = np.percentile(lid, 5, axis=0)
    ray_p95 = np.percentile(lid, 95, axis=0)
    ax.bar(rays, ray_mean, color="tab:blue", alpha=0.75, label="mean")
    ax.errorbar(rays, ray_mean, yerr=[ray_mean - ray_p05, ray_p95 - ray_mean],
                fmt="none", ecolor="k", elinewidth=0.8, capsize=2, label="p05–p95")
    ax.set_xlabel("LiDAR ray (0° = +x, 22.5° steps)")
    ax.set_ylabel("range [m]")
    ax.set_title("mean range per ray", fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    ax = axes[1, 3]
    ax.axis("off")
    rows = [
        ("files (trajectories)", stats["n_files"]),
        ("scenes / map groups", f"{stats['n_scenes']} / {stats['n_map_groups']}"),
        ("query points", stats["n_query_points"]),
        ("points per trajectory", f"{stats['points_per_trajectory']['mean']:.1f}"
                                  f" ({stats['points_per_trajectory']['min']:.0f}"
                                  f"–{stats['points_per_trajectory']['max']:.0f})"),
        ("safety_verified", f"{safety / total:.3f}"),
        ("continuity_verified", f"{cont / total:.3f}"),
        ("label_valid", f"{valid / total:.3f}"),
        ("semi-major a p50", f"{np.median(aa):.2f} m"),
        ("semi-minor b p50", f"{np.median(bb):.2f} m"),
        ("solver status", json.dumps(stats["solver_status_counts"])),
    ]
    ax.text(0.0, 1.0, "dataset summary", fontsize=11, weight="bold", va="top")
    for r, (key, val) in enumerate(rows):
        ax.text(0.0, 0.90 - 0.085 * r, f"{key}", fontsize=9, va="top")
        ax.text(0.58, 0.90 - 0.085 * r, f"{val}", fontsize=9, va="top", color="tab:blue")
    fig.suptitle(f"V3 trajectory dataset dashboard — {tag} "
                 f"({stats['n_files']} trajectories, {stats['n_query_points']} query points)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(dst, dpi=110)
    plt.close(fig)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(PROJECT_ROOT, "data", "processed_v3",
                                                  "trajectories", "pilot10"))
    ap.add_argument("--tag", default="pilot10")
    ap.add_argument("--out", default="")
    ap.add_argument("--max-overview", type=int, default=10)
    ap.add_argument("--zoom-m", type=float, default=7.0)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.npz")))
    if not files:
        print(f"no NPZ under {args.dir}")
        return 1
    out = args.out or os.path.join(PROJECT_ROOT, "outputs_v3", "viz", args.tag)
    os.makedirs(out, exist_ok=True)

    plot_overview(files, os.path.join(out, f"overview_{args.tag}.png"), args.max_overview)
    print(f"overview  -> {os.path.join(out, f'overview_{args.tag}.png')}")
    for path in files:
        s = load_trajectory(path)
        stem = os.path.splitext(os.path.basename(path))[0]
        dst = os.path.join(out, f"detail_{stem}.png")
        plot_detail(s, dst, stem, zoom_m=args.zoom_m)
        print(f"detail    -> {os.path.basename(dst)}")
    stats = plot_dashboard(files, os.path.join(out, f"dashboard_{args.tag}.png"), args.tag)
    print(f"dashboard -> {os.path.join(out, f'dashboard_{args.tag}.png')}")
    print(json.dumps({k: stats[k] for k in ("n_files", "n_scenes", "n_query_points",
                                            "label_valid_rate", "safety_verified_rate",
                                            "continuity_verified_rate")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
