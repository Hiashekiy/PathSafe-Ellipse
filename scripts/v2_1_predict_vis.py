"""Visualise RAW test-set predictions against ground truth (no repair, no scaling).

Figure 1 (scene overlays): held-out scenes with the occupancy grid, A* stations,
ground-truth labels (blue) and predicted ellipses (orange dashed).
Figure 2 (per-sample grid): individual test labels, GT vs prediction with IoU.
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
import torch  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from pathsafe_ellipse.v2.dataset_v21 import (  # noqa: E402
    Standardiser, build_table, load_split_files, split_by_group,
)
from pathsafe_ellipse.v2.pilot import load_scene_npz  # noqa: E402
from v2_1_train_full import Net, ellipse_iou, to_ellipse  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dataset_v21")
    ap.add_argument("--run", default="runs_v21_full")
    ap.add_argument("--n-scenes", type=int, default=6)
    ap.add_argument("--grid", type=int, default=16)
    args = ap.parse_args()

    ckpt = torch.load(os.path.join(PROJECT_ROOT, args.run, "model_seed0.pt"),
                      map_location="cpu", weights_only=False)
    model = Net(dropout=0.0)
    model.load_state_dict(ckpt["model"])
    model.eval()

    files = load_split_files(os.path.join(PROJECT_ROOT, "data", "processed_v2_1",
                                         f"pilot_{args.tag}", "v2_1"))
    table = build_table(files, with_patch=True, only_valid=True)
    mask_tr, mask_va, mask_te = split_by_group(table["group"], seed=0)
    lidar_std = Standardiser.fit(table["lidar"][mask_tr])
    extra_std = Standardiser.fit(table["clearance_p"][mask_tr])
    off_std = Standardiser.fit(table["offset"][mask_tr])

    with torch.no_grad():
        pred_all = model(torch.from_numpy(table["patch"][mask_te]).float(),
                         torch.from_numpy(lidar_std(table["lidar"][mask_te])).float(),
                         torch.from_numpy(extra_std(table["clearance_p"][mask_te])).float()
                         ).numpy()
    off_p_all, ab_p_all, ang_p_all = to_ellipse(pred_all)
    off_p_all = off_p_all * off_std.std + off_std.mean
    off_g_all = table["offset"][mask_te]
    ab_g_all = np.exp(table["log_axes"][mask_te])
    a2 = table["angle2"][mask_te]
    ang_g_all = 0.5 * np.arctan2(a2[:, 1], a2[:, 0])
    iou_all = ellipse_iou(off_p_all, ab_p_all, ang_p_all, off_g_all, ab_g_all, ang_g_all)

    te_idx = np.flatnonzero(mask_te)
    local_of = {int(g): i for i, g in enumerate(te_idx)}
    scenes = sorted(set(table["scene_id"][mask_te].tolist()))[: args.n_scenes]

    # ---------------- figure 1: scene overlays
    ncol = 3
    nrow = int(math.ceil(len(scenes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.4 * ncol, 6.2 * nrow))
    axes = np.atleast_2d(axes)
    scene_stats = []
    for ax, sid in zip(axes.ravel(), scenes):
        npz = load_scene_npz(os.path.join(PROJECT_ROOT, "data", "processed_v2_1",
                                         f"pilot_{args.tag}", "v2_1", f"{sid}.npz"))
        rs = float(npz["meta"]["r_scene_m"])
        n = npz["blocked"].shape[0]
        ax.imshow(npz["blocked"], cmap="gray_r", alpha=0.5, interpolation="nearest",
                  extent=[0, n * rs, n * rs, 0])
        rows = np.flatnonzero((table["scene_id"] == sid) & mask_te)
        ious = []
        for g in rows:
            i_loc = local_of[int(g)]
            sel = (npz["path_id"] == table["path_id"][g]) & \
                  (npz["point_id"] == table["point_id"][g])
            if not np.any(sel):
                continue
            j = int(np.flatnonzero(sel)[0])
            p_xy = npz["p_xy"][j].astype(float)
            cg = npz["q_xy"][j].astype(float)
            cp = p_xy + off_p_all[i_loc]
            ax.plot(p_xy[0], p_xy[1], ".", ms=2.0, color="tab:red", alpha=0.8)
            ax.add_patch(Ellipse((cg[0], cg[1]), 2 * ab_g_all[i_loc, 0], 2 * ab_g_all[i_loc, 1],
                                 angle=math.degrees(ang_g_all[i_loc]), fill=False,
                                 ec="tab:blue", lw=1.0))
            ax.add_patch(Ellipse((cp[0], cp[1]), 2 * ab_p_all[i_loc, 0], 2 * ab_p_all[i_loc, 1],
                                 angle=math.degrees(ang_p_all[i_loc]), fill=False,
                                 ec="tab:orange", lw=1.0, ls="--"))
            ious.append(float(iou_all[i_loc]))
        ious = np.asarray(ious)
        ax.set_title(f"{sid[:36]}\n{len(ious)} test labels, IoU mean "
                     f"{ious.mean():.3f} (p50 {np.median(ious):.3f})", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        scene_stats.append({"scene": sid, "n": int(len(ious)),
                            "iou_mean": float(ious.mean()),
                            "iou_p50": float(np.median(ious))})
    for ax in axes.ravel()[len(scenes):]:
        ax.axis("off")
    fig.suptitle("V2.1 RAW predictions on held-out scenes   "
                 "blue = GT label, orange dashed = prediction, red dots = A* stations p_i")
    fig.tight_layout()
    p1 = os.path.join(PROJECT_ROOT, args.run, "test_predictions_scenes.png")
    fig.savefig(p1, dpi=112)
    plt.close(fig)

    # ---------------- figure 2: per-sample grid
    pick = np.linspace(0, len(iou_all) - 1, min(args.grid, len(iou_all))).astype(int)
    ncol = 4
    nrow = int(math.ceil(len(pick) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.2 * nrow))
    axes = np.atleast_2d(axes)
    for ax, i_loc in zip(axes.ravel(), pick):
        g = te_idx[i_loc]
        sid = str(table["scene_id"][g])
        npz = load_scene_npz(os.path.join(PROJECT_ROOT, "data", "processed_v2_1",
                                         f"pilot_{args.tag}", "v2_1", f"{sid}.npz"))
        rs = float(npz["meta"]["r_scene_m"])
        patch = table["patch"][g, 0]
        px = patch.shape[0]
        p_xy = npz["p_xy"][int(table["point_id"][g])].astype(float) if False else None
        sel = (npz["path_id"] == table["path_id"][g]) & \
              (npz["point_id"] == table["point_id"][g])
        j = int(np.flatnonzero(sel)[0])
        p_xy = npz["p_xy"][j].astype(float)
        # the patch is centred on p_i; draw the ellipses in the same local frame
        half = px // 2
        cx = p_xy[0] / rs - half
        cy = p_xy[1] / rs - half
        ax.imshow(patch, cmap="gray_r", alpha=0.6, interpolation="nearest",
                  extent=[cx * rs, (cx + px) * rs, (cy + px) * rs, cy * rs])
        cg = npz["q_xy"][j].astype(float)
        cp = p_xy + off_p_all[i_loc]
        ax.add_patch(Ellipse((cg[0], cg[1]), 2 * ab_g_all[i_loc, 0], 2 * ab_g_all[i_loc, 1],
                             angle=math.degrees(ang_g_all[i_loc]), fill=False,
                             ec="tab:blue", lw=1.5))
        ax.add_patch(Ellipse((cp[0], cp[1]), 2 * ab_p_all[i_loc, 0], 2 * ab_p_all[i_loc, 1],
                             angle=math.degrees(ang_p_all[i_loc]), fill=False,
                             ec="tab:orange", lw=1.5, ls="--"))
        ax.plot(p_xy[0], p_xy[1], "+", ms=9, color="tab:red", mew=1.6)
        ax.plot(cg[0], cg[1], "*", ms=9, color="tab:blue")
        ax.plot(cp[0], cp[1], "x", ms=7, color="tab:orange", mew=1.4)
        ax.set_title(f"IoU {iou_all[i_loc]:.3f} | GT a={ab_g_all[i_loc,0]:.2f} "
                     f"b={ab_g_all[i_loc,1]:.2f}\npred a={ab_p_all[i_loc,0]:.2f} "
                     f"b={ab_p_all[i_loc,1]:.2f} | centre err "
                     f"{np.linalg.norm(off_p_all[i_loc]-off_g_all[i_loc]):.2f} m", fontsize=6.5)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[len(pick):]:
        ax.axis("off")
    fig.suptitle("Individual test predictions (64x64 patch around p_i)   "
                 "blue = GT, orange dashed = prediction")
    fig.tight_layout()
    p2 = os.path.join(PROJECT_ROOT, args.run, "test_predictions_grid.png")
    fig.savefig(p2, dpi=115)
    plt.close(fig)

    summary = {
        "n_test": int(mask_te.sum()),
        "iou_mean": float(iou_all.mean()), "iou_p50": float(np.median(iou_all)),
        "iou_gt_0_5": float(np.mean(iou_all > 0.5)),
        "centre_err_mean_m": float(np.linalg.norm(off_p_all - off_g_all, axis=1).mean()),
        "scenes": scene_stats,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "scenes"}, indent=2))
    for s in scene_stats:
        print(f"  {s['scene'][:44]:46s} n={s['n']:4d} IoU {s['iou_mean']:.3f}")
    dst = os.path.join(PROJECT_ROOT, args.run, "test_predictions.json")
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"-> {p1}\n-> {p2}\n-> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
