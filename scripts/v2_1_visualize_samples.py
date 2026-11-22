"""Visualise the generated pilot samples straight from the saved NPZ files.

Three figures per run:
  gallery_<tag>.png       scenes x {V2, V2.1}: inflated occupancy, A* path P,
                          optimised centre line Q, every valid ellipse
  sample_card_<scene>.png one scene: all labels + offsets, LiDAR rays from p_i
                          for two labels, and the clearance/area/offset curves
  label_grid_<scene>.png  4x4 individual training samples (what the network sees)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from pathsafe_ellipse.v2.pilot import load_scene_npz  # noqa: E402


def entry_dir(e):
    """Index entries may or may not carry an explicit sub-directory."""
    return e.get("dir") or e["variant"].replace("v21", "v2_1")


def path_poly(npz, path_id, which="q"):
    off = npz[f"path_{which}_offsets"]
    xy = npz[f"path_{which}_xy"]
    ids = [int(v) for v in npz["path_path_id"]]
    k = ids.index(int(path_id))
    return xy[off[k]:off[k + 1]]


def draw_scene(ax, npz, title, ellipses=True, offsets=False):
    meta = npz["meta"]
    rs = float(meta["r_scene_m"])
    n = npz["blocked"].shape[0]
    ext = [0, n * rs, n * rs, 0]
    ax.imshow(npz["blocked"], cmap="gray_r", alpha=0.55, interpolation="nearest", extent=ext)
    for pid in [int(v) for v in npz["path_path_id"]]:
        p = path_poly(npz, pid, "p")
        q = path_poly(npz, pid, "q")
        ax.plot(p[:, 0], p[:, 1], "-", lw=1.0, color="tab:orange", alpha=0.9)
        ax.plot(q[:, 0], q[:, 1], "-", lw=1.2, color="tab:blue", alpha=0.9)
        if offsets:
            step = max(1, len(p) // 14)
            for k in range(0, min(len(p), len(q)), step):
                ax.annotate("", xy=(q[k, 0], q[k, 1]), xytext=(p[k, 0], p[k, 1]),
                            arrowprops=dict(arrowstyle="-", color="tab:purple", lw=0.6,
                                            alpha=0.85))
        ax.plot(p[0, 0], p[0, 1], "o", ms=5, color="lime", mec="k")
        ax.plot(p[-1, 0], p[-1, 1], "X", ms=7, color="red", mec="k")
    if ellipses:
        for k in range(len(npz["q_xy"])):
            if not int(npz["label_valid"][k]):
                continue
            ab = npz["ellipse_ab"][k]
            ang = float(npz["ellipse_angle"][k])
            c = npz["q_xy"][k]
            ax.add_patch(Ellipse((c[0], c[1]), 2 * ab[0], 2 * ab[1],
                                 angle=math.degrees(ang), fill=False, ec="tab:green",
                                 lw=0.7, alpha=0.6))
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[3], ext[2])


def draw_label_patch(ax, npz, k, window_m=6.0):
    meta = npz["meta"]
    rs = float(meta["r_scene_m"])
    n = npz["blocked"].shape[0]
    q = npz["q_xy"][k]
    p = npz["p_xy"][k]
    ext = [0, n * rs, n * rs, 0]
    ax.imshow(npz["blocked"], cmap="gray_r", alpha=0.55, interpolation="nearest", extent=ext)
    ranges = npz["lidar_ranges"][k].astype(float)
    max_r = float(meta["lidar_max_range_m"])
    for j, r in enumerate(ranges):
        a = 2.0 * math.pi * j / len(ranges)
        ax.plot([p[0], p[0] + r * math.cos(a)], [p[1], p[1] + r * math.sin(a)],
                "-", lw=0.5, color="tab:red", alpha=0.55)
    t = npz["path_tangent"][k]
    ax.annotate("", xy=(p[0] + 1.5 * t[0], p[1] + 1.5 * t[1]), xytext=(p[0], p[1]),
                arrowprops=dict(arrowstyle="->", color="tab:olive", lw=1.0))
    ab = npz["ellipse_ab"][k]
    ax.add_patch(Ellipse((q[0], q[1]), 2 * ab[0], 2 * ab[1],
                         angle=math.degrees(float(npz["ellipse_angle"][k])),
                         fill=False, ec="tab:blue", lw=1.4))
    ax.annotate("", xy=(q[0], q[1]), xytext=(p[0], p[1]),
                arrowprops=dict(arrowstyle="->", color="tab:purple", lw=1.1))
    ax.plot(p[0], p[1], "+", ms=9, color="tab:red", mew=1.6)
    ax.plot(q[0], q[1], "*", ms=9, color="tab:blue")
    ax.set_aspect("equal")
    ax.set_xlim(q[0] - window_m, q[0] + window_m)
    ax.set_ylim(q[1] - window_m, q[1] + window_m)
    ax.set_title(f"pt {int(npz['point_id'][k])}  a={ab[0]:.2f} b={ab[1]:.2f} m  "
                 f"A={npz['ellipse_area'][k]:.1f} m²\n"
                 f"safe={int(npz['safety_verified'][k])} cont="
                 f"{int(npz['continuity_verified'][k])} valid={int(npz['label_valid'][k])} "
                 f"{npz['generation_time_ms'][k]:.0f} ms", fontsize=6.5)
    ax.set_xticks([]); ax.set_yticks([])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="stage4_20")
    ap.add_argument("--n-gallery", type=int, default=4)
    ap.add_argument("--scene", default=None, help="scene_id for the detail figures")
    ap.add_argument("--root", default=os.path.join(PROJECT_ROOT, "data", "processed_v2_1"))
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "outputs_v2_1", "samples"))
    args = ap.parse_args()

    base = os.path.join(args.root, f"pilot_{args.tag}")
    with open(os.path.join(base, "index.json"), encoding="utf-8") as fh:
        index = json.load(fh)["entries"]
    os.makedirs(args.out, exist_ok=True)

    v21 = [e for e in index if e["variant"] == "v21"][: args.n_gallery]
    v2 = [e for e in index if e["variant"] == "v2"]

    # ---------------- 1) gallery: same scene, V2 above V2.1 below
    fig, axes = plt.subplots(2, len(v21), figsize=(4.6 * len(v21), 9.4))
    if len(v21) == 1:
        axes = axes.reshape(2, 1)
    for j, e in enumerate(v21):
        d21 = load_scene_npz(os.path.join(base, entry_dir(e), e["npz"]))
        draw_scene(axes[0][j], d21, f"V2.1  {e['scene_id'][:30]}\n"
                                    f"{int(d21['label_valid'].sum())} valid labels, "
                                    f"FOV {d21['meta']['physical_field_m']:.0f} m",
                   offsets=True)
        match = next((x for x in v2 if x["scene_id"] == e["scene_id"]), None)
        if match:
            d2 = load_scene_npz(os.path.join(base, entry_dir(match), match["npz"]))
            draw_scene(axes[1][j], d2, f"V2  {e['scene_id'][:30]}\n"
                                       f"{int(d2['label_valid'].sum())} valid labels",
                       offsets=True)
        else:
            axes[1][j].axis("off")
    fig.suptitle(f"Generated samples - inflated occupancy, A* path (orange), "
                 f"V2.1/V2 centre line (blue), offset vectors (purple), "
                 f"verified ellipses (green)")
    fig.tight_layout()
    p1 = os.path.join(args.out, f"gallery_{args.tag}.png")
    fig.savefig(p1, dpi=110)
    plt.close(fig)
    print("->", p1)

    # ---------------- 2) sample card for one scene
    sid = args.scene or v21[0]["scene_id"]
    entry = next(e for e in index if e["scene_id"] == sid and e["variant"] == "v21")
    npz = load_scene_npz(os.path.join(base, entry_dir(entry), entry["npz"]))
    meta = npz["meta"]

    fig = plt.figure(figsize=(17, 9.6))
    ax = fig.add_subplot(2, 3, 1)
    draw_scene(ax, npz, f"{sid}\n{int(npz['label_valid'].sum())}/"
                        f"{len(npz['q_xy'])} labels valid, FOV "
                        f"{meta['physical_field_m']:.0f} m, r_scene {meta['r_scene_m']:.4f} m",
               offsets=True)
    valid = np.flatnonzero(npz["label_valid"]).astype(int)
    for panel, k in zip((fig.add_subplot(2, 3, 2), fig.add_subplot(2, 3, 3)),
                        [valid[0], valid[len(valid) // 2]] if len(valid) > 1 else [valid[0]] * 2):
        ax2 = fig.add_subplot(2, 3, 2) if k == valid[0] else fig.add_subplot(2, 3, 3)
        fig.axes.remove(panel)
        draw_label_patch(ax2, npz, int(k))
    ax = fig.add_subplot(2, 3, 4)
    for pid in [int(v) for v in npz["path_path_id"]]:
        m = npz["path_id"] == pid
        order = np.argsort(npz["point_id"][m])
        arc = np.arange(order.size) / max(order.size - 1, 1)
        ax.plot(arc, npz["clearance"][m][order], "-o", ms=2.5, lw=1.0, label=f"path {pid} D(q)")
        ax.plot(arc, np.linalg.norm(npz["center_offset"][m][order], axis=1), "-s", ms=2.5,
                lw=1.0, alpha=0.8, label=f"path {pid} |q-p|")
    ax.axhline(float(meta["d_safe_m"]), color="red", ls=":", lw=1.0, label="d_safe")
    ax.set_xlabel("normalised arc length"); ax.set_ylabel("m")
    ax.set_title("clearance and offset along the labelled path")
    ax.legend(fontsize=6); ax.grid(alpha=0.3)
    ax = fig.add_subplot(2, 3, 5)
    for pid in [int(v) for v in npz["path_path_id"]]:
        m = npz["path_id"] == pid
        order = np.argsort(npz["point_id"][m])
        ax.plot(np.arange(order.size) / max(order.size - 1, 1), npz["ellipse_area"][m][order],
                "-o", ms=2.5, lw=1.0, label=f"path {pid}")
    ax.set_xlabel("normalised arc length"); ax.set_ylabel("ellipse area [m²]")
    ax.set_title("labelled ellipse area"); ax.legend(fontsize=6); ax.grid(alpha=0.3)
    ax = fig.add_subplot(2, 3, 6); ax.axis("off")
    rows = [("field", "value")] + [
        ("label_version", meta["label_version"]), ("git_commit", meta["git_commit"]),
        ("map", meta["source_map_id"]), ("crop", str(meta["crop_origin"])),
        ("FOV / r_scene", f"{meta['physical_field_m']:.0f} m / {meta['r_scene_m']:.4f} m"),
        ("d_safe", f"{meta['d_safe_m']:.2f} m"),
        ("R_max", f"{meta['r_max_m']:.1f} m"),
        ("seed", str(meta["seed"])),
        ("labels total", str(len(npz["q_xy"]))),
        ("safe / cont / valid", f"{int(npz['safety_verified'].sum())} / "
                                f"{int(npz['continuity_verified'].sum())} / "
                                f"{int(npz['label_valid'].sum())}"),
        ("area p50 / min", f"{np.median(npz['ellipse_area']):.1f} / "
                           f"{npz['ellipse_area'].min():.2f} m²"),
        ("ellipse ms p50", f"{np.median(npz['generation_time_ms']):.0f}"),
        ("paths (status)", "; ".join(f"{int(i)}:{s}" for i, s in
                                     zip(npz["path_path_id"], npz["path_label_status"]))),
    ]
    t = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="left")
    t.auto_set_font_size(False); t.set_fontsize(6.6); t.scale(1, 1.25)
    ax.set_title("sample metadata (from the NPZ)", fontsize=9)
    fig.suptitle(f"Sample card - {sid}")
    fig.tight_layout()
    p2 = os.path.join(args.out, f"sample_card_{sid}.png")
    fig.savefig(p2, dpi=110)
    plt.close(fig)
    print("->", p2)

    # ---------------- 3) 4x4 grid of individual training samples
    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    pick = valid[np.linspace(0, len(valid) - 1, min(16, len(valid))).astype(int)]
    for ax, k in zip(axes.ravel(), pick):
        draw_label_patch(ax, npz, int(k), window_m=5.0)
    for ax in axes.ravel()[len(pick):]:
        ax.axis("off")
    fig.suptitle(f"Individual training samples - {sid}\n"
                 f"red rays = 16 LiDAR ranges cast from the A* point p_i, olive arrow = "
                 f"path tangent, purple arrow = centre offset, blue ellipse = label")
    fig.tight_layout()
    p3 = os.path.join(args.out, f"label_grid_{sid}.png")
    fig.savefig(p3, dpi=115)
    plt.close(fig)
    print("->", p3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
