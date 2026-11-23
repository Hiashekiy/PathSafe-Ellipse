"""Visualise raw V3 trajectory samples (data, not predictions).

    python scripts/viz_trajectory_v3.py --dir data/processed_v3/trajectories/train_v3 --n 3

Three panels per trajectory:

    1) scene      : inflated occupancy + A* path + query points + 16-ray LiDAR fan
    2) labels     : GT centre line (q_i), GT ellipses, and the centre offset p_i -> q_i
    3) stations   : zoom on three stations with every stored value printed
                    (offset, a, b, angle, area, label_valid / safety / continuity)

Everything is read straight from the NPZ, so this is exactly what the network sees.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

PATH_COLOR = "tab:orange"
LIDAR_COLOR = "tab:cyan"
LABEL_COLOR = "#1f77b4"
OFFSET_COLOR = "#2ca02c"


def load(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    out["meta"] = json.loads(str(out["meta_json"].item()))
    return out


def scene_base(ax, sample, title, rays=True):
    occ = np.asarray(sample["occupancy"])
    rs = float(sample["meta"]["r_scene"])
    ext = [0, occ.shape[1] * rs, occ.shape[0] * rs, 0]
    ax.imshow(occ, cmap="gray_r", alpha=0.5, interpolation="nearest", extent=ext)
    p = np.asarray(sample["path_xy"], float)
    if rays:
        lidar = np.asarray(sample["lidar_ranges"], float)
        ang = 2 * np.pi * np.arange(lidar.shape[-1]) / lidar.shape[-1]
        for i in range(len(p)):
            ax.plot(np.stack([np.full(len(ang), p[i, 0]),
                              p[i, 0] + lidar[i] * np.cos(ang)], 0),
                    np.stack([np.full(len(ang), p[i, 1]),
                              p[i, 1] + lidar[i] * np.sin(ang)], 0),
                    "-", color=LIDAR_COLOR, lw=0.25, alpha=0.35, zorder=1)
    ax.plot(p[:, 0], p[:, 1], "-", lw=1.0, color=PATH_COLOR, zorder=3)
    ax.plot(p[:, 0], p[:, 1], "o", ms=2.6, color=PATH_COLOR, zorder=4)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    return ext


def draw_labels(ax, sample, n_ellipses=10, arrows=True):
    p = np.asarray(sample["path_xy"], float)
    q = np.asarray(sample["centerline_xy"], float)
    ab = np.asarray(sample["ellipse_ab"], float)
    ang = np.asarray(sample["ellipse_angle"], float)
    valid = np.asarray(sample["label_valid"]).astype(bool)
    ax.plot(q[:, 0], q[:, 1], "-", lw=1.4, color=LABEL_COLOR, alpha=0.8, zorder=5)
    keep = [i for i in np.linspace(0, len(p) - 1, min(n_ellipses, len(p))).astype(int)
            if valid[i]]
    for i in keep:
        ax.add_patch(Ellipse(q[i], 2 * ab[i, 0], 2 * ab[i, 1], angle=np.degrees(ang[i]),
                             fill=False, ec=LABEL_COLOR, lw=1.4, zorder=6))
        if arrows:
            ax.annotate("", xy=q[i], xytext=p[i],
                        arrowprops=dict(arrowstyle="->", color=OFFSET_COLOR, lw=1.0),
                        zorder=5)
    bad = np.flatnonzero(~valid)
    if bad.size:
        ax.plot(p[bad, 0], p[bad, 1], "x", ms=5, color="tab:red", zorder=7)


def draw_stations(fig, ax, sample, ids):
    p = np.asarray(sample["path_xy"], float)
    q = np.asarray(sample["centerline_xy"], float)
    ab = np.asarray(sample["ellipse_ab"], float)
    ang = np.asarray(sample["ellipse_angle"], float)
    area = np.asarray(sample["ellipse_area"], float)
    off = np.asarray(sample["center_offset"], float)
    lidar = np.asarray(sample["lidar_ranges"], float)
    valid = np.asarray(sample["label_valid"]).astype(bool)
    safe = np.asarray(sample["safety_verified"]).astype(bool)
    cont = np.asarray(sample["continuity_verified"]).astype(bool)

    ang_r = 2 * np.pi * np.arange(lidar.shape[-1]) / lidar.shape[-1]
    for k, i in enumerate(ids):
        ax.plot(np.stack([np.full(len(ang_r), p[i, 0]),
                          p[i, 0] + lidar[i] * np.cos(ang_r)], 0),
                np.stack([np.full(len(ang_r), p[i, 1]),
                          p[i, 1] + lidar[i] * np.sin(ang_r)], 0),
                "-", color=LIDAR_COLOR, lw=0.8, alpha=0.7, zorder=2)
        ax.add_patch(Ellipse(q[i], 2 * ab[i, 0], 2 * ab[i, 1], angle=np.degrees(ang[i]),
                             fill=False, ec=LABEL_COLOR, lw=1.6, zorder=5))
        ax.plot(*p[i], "o", ms=5, color=PATH_COLOR, zorder=6)
        ax.plot(*q[i], "+", ms=8, color=LABEL_COLOR, zorder=6)
        ax.annotate("", xy=q[i], xytext=p[i],
                    arrowprops=dict(arrowstyle="->", color=OFFSET_COLOR, lw=1.2), zorder=4)
        dx, dy = float(off[i, 0]), float(off[i, 1])
        offsets = ((20, 24), (-210, 70), (20, -90))
        ax.annotate(
            f"station {i}\n offset {np.hypot(dx, dy):.2f} m ({dx:+.2f},{dy:+.2f})\n"
            f" a={ab[i, 0]:.2f} b={ab[i, 1]:.2f}\n θ={np.degrees(ang[i]):.0f}° "
            f"area={area[i]:.1f}\n safety={int(safe[i])} cont={int(cont[i])} "
            f"valid={int(valid[i])}",
            xy=(float(q[i, 0]), float(q[i, 1])), xytext=offsets[k % len(offsets)],
            textcoords="offset points", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", alpha=0.9),
            arrowprops=dict(arrowstyle="-", color="0.5", lw=0.6), zorder=9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(PROJECT_ROOT, "data", "processed_v3",
                                                  "trajectories", "train_v3"))
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--every", type=int, default=0,
                    help="0 = pick evenly spaced files, else take every k-th file")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    files = sorted(p for p in glob.glob(os.path.join(args.dir, "*.npz"))
                   if ".tmp." not in os.path.basename(p))
    if not files:
        print(f"no npz under {args.dir}")
        return 1
    if args.every:
        picked = files[:: args.every][: args.n]
    else:
        picked = [files[i] for i in np.linspace(0, len(files) - 1, args.n).astype(int)]
    out_dir = args.out or os.path.join(PROJECT_ROOT, "outputs_v3", "data_samples")
    os.makedirs(out_dir, exist_ok=True)

    print(f"{len(files)} files in {args.dir}; drawing {len(picked)}")
    for path in picked:
        s = load(path)
        meta = s["meta"]
        n = len(np.asarray(s["path_xy"]))
        fig, axes = plt.subplots(1, 3, figsize=(19.0, 6.4))
        scene_base(axes[0], s, "1) scene: map + A* path/query points + 16-ray LiDAR")
        ext = scene_base(axes[1], s, "2) labels: GT centre line (blue), ellipses, "
                                    "offset arrows, red x = invalid", rays=False)
        draw_labels(axes[1], s)
        ids = [0, n // 2, n - 1] if n >= 3 else list(range(n))
        scene_base(axes[2], s, "3) three stations with every stored value", rays=False)
        draw_stations(fig, axes[2], s, ids)
        xs = np.asarray(s["centerline_xy"], float)[:, 0]
        ys = np.asarray(s["centerline_xy"], float)[:, 1]
        pad = 6.0
        axes[2].set_xlim(xs.min() - pad, xs.max() + pad)
        axes[2].set_ylim(ys.max() + pad, ys.min() - pad)

        handles = [Line2D([], [], color=PATH_COLOR, marker="o", ms=3, lw=1.0,
                          label="A* path / query point p_i"),
                   Line2D([], [], color=LIDAR_COLOR, lw=1.0, label="LiDAR (16 rays, 9 m)"),
                   Line2D([], [], color=LABEL_COLOR, lw=1.4, label="centre line q_i / ellipse"),
                   Line2D([], [], color=OFFSET_COLOR, lw=1.2, label="offset q_i - p_i"),
                   Line2D([], [], color="tab:red", marker="x", ls="none",
                          label="label_valid = 0")]
        fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9, frameon=False)
        fig.suptitle(f"{meta['scene_id']}  path {meta['path_id']}   N={n}   "
                     f"r_cell {meta['r_scene']:.4f} m   field {meta['physical_field_m']:.0f} m   "
                     f"valid {int(np.asarray(s['label_valid']).sum())}/{n}",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0.05, 1, 0.94))
        stem = os.path.splitext(os.path.basename(path))[0]
        fig.savefig(os.path.join(out_dir, f"sample_{stem}.png"), dpi=110)
        plt.close(fig)
        print(f"  {stem[:56]:56s} N={n:3d}  "
              f"valid {int(np.asarray(s['label_valid']).sum()):3d}  "
              f"median area {np.median(np.asarray(s['ellipse_area'])):6.2f} m2  "
              f"median |offset| {np.median(np.linalg.norm(s['center_offset'], axis=1)):.2f} m")
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
