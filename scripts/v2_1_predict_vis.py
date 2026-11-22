"""Test-set prediction visualisation, one trajectory per figure.

For every path: three panels side by side
    1. ground truth only
    2. prediction only (filled, so the shape reads at a glance)
    3. overlay (GT outline + translucent prediction), plus the paths and stations
Plus a best/median/worst comparison sheet and a large per-sample GT|pred grid.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Ellipse, Patch  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from pathsafe_ellipse.v2.dataset_v21 import (  # noqa: E402
    Standardiser, build_table, load_split_files, split_by_group,
)
from pathsafe_ellipse.v2.pilot import load_scene_npz  # noqa: E402
from v2_1_train_full import Net, ellipse_iou, to_ellipse  # noqa: E402


def load_predictions(args):
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
        pred = model(torch.from_numpy(table["patch"][mask_te]).float(),
                     torch.from_numpy(lidar_std(table["lidar"][mask_te])).float(),
                     torch.from_numpy(extra_std(table["clearance_p"][mask_te])).float()
                     ).numpy()
    off_p, ab_p, ang_p = to_ellipse(pred)
    off_p = off_p * off_std.std + off_std.mean
    off_g = table["offset"][mask_te]
    ab_g = np.exp(table["log_axes"][mask_te])
    a2 = table["angle2"][mask_te]
    ang_g = 0.5 * np.arctan2(a2[:, 1], a2[:, 0])
    iou = ellipse_iou(off_p, ab_p, ang_p, off_g, ab_g, ang_g)
    te_idx = np.flatnonzero(mask_te)
    local = {int(g): i for i, g in enumerate(te_idx)}
    return table, mask_te, local, off_p, ab_p, ang_p, off_g, ab_g, ang_g, iou


def group_paths(table, mask_te) -> Dict[tuple, List[int]]:
    out: Dict[tuple, List[int]] = {}
    for g in np.flatnonzero(mask_te):
        key = (str(table["scene_id"][g]), int(table["path_id"][g]))
        out.setdefault(key, []).append(int(g))
    for k in out:
        out[k].sort(key=lambda g: int(table["point_id"][g]))
    return out


def draw_path(ax, npz, rows, local, off_p, ab_p, ang_p, off_g, ab_g, ang_g, mode: str,
              fill_pred: bool = False, path_ids=None, point_ids=None):
    rs = float(npz["meta"]["r_scene_m"])
    n = npz["blocked"].shape[0]
    ax.imshow(npz["blocked"], cmap="gray_r", alpha=0.45, interpolation="nearest",
              extent=[0, n * rs, n * rs, 0])
    pts_g, pts_p, pts_a = [], [], []
    for g in rows:
        i = local[g]
        sel = (npz["path_id"] == path_ids[g]) & (npz["point_id"] == point_ids[g])
        if not np.any(sel):
            continue
        j = int(np.flatnonzero(sel)[0])
        p_xy = npz["p_xy"][j].astype(float)
        cg = p_xy + off_g[i]
        cp = p_xy + off_p[i]
        pts_g.append(cg)
        pts_p.append(cp)
        pts_a.append(p_xy)
        if mode in ("gt", "both"):
            ax.add_patch(Ellipse((cg[0], cg[1]), 2 * ab_g[i, 0], 2 * ab_g[i, 1],
                                 angle=math.degrees(ang_g[i]), fill=False,
                                 ec="tab:blue", lw=1.3, zorder=4))
        if mode in ("pred", "both"):
            ax.add_patch(Ellipse((cp[0], cp[1]), 2 * ab_p[i, 0], 2 * ab_p[i, 1],
                                 angle=math.degrees(ang_p[i]),
                                 facecolor="tab:orange" if fill_pred else "none",
                                 alpha=0.28 if fill_pred else 1.0, fill=fill_pred,
                                 ec="tab:orange", lw=1.3, ls="--", zorder=3))
        if mode == "both":
            ax.annotate("", xy=(cp[0], cp[1]), xytext=(cg[0], cg[1]),
                        arrowprops=dict(arrowstyle="->", color="0.35", lw=0.6, alpha=0.8))
    # connect the stations: the whole point is to see the trajectory, not loose rings
    if pts_a:
        a = np.asarray(pts_a)
        ax.plot(a[:, 0], a[:, 1], "-", color="tab:red", lw=1.5, alpha=0.75, zorder=6,
                label="A* path p_i")
    if pts_g and mode in ("gt", "both"):
        c = np.asarray(pts_g)
        ax.plot(c[:, 0], c[:, 1], "-", color="tab:blue", lw=1.8, zorder=7,
                label="GT centre line q_i")
        ax.plot(c[:, 0], c[:, 1], ".", ms=3.0, color="tab:blue", zorder=8)
    if pts_p and mode in ("pred", "both"):
        c = np.asarray(pts_p)
        ax.plot(c[:, 0], c[:, 1], "-", color="tab:orange", lw=1.8, zorder=7,
                label="predicted centres q̂_i")
        ax.plot(c[:, 0], c[:, 1], "x", ms=3.4, color="tab:orange", zorder=8)
    # keep the view on the labelled corridor
    allc = np.vstack([np.asarray(pts_g + pts_p + pts_a)]) if pts_g else np.zeros((1, 2))
    pad = 6.0
    ax.set_xlim(max(0, allc[:, 0].min() - pad), allc[:, 0].max() + pad)
    ax.set_ylim(allc[:, 1].max() + pad, max(0, allc[:, 1].min() - pad))
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dataset_v21")
    ap.add_argument("--run", default="runs_v21_full")
    ap.add_argument("--n-paths", type=int, default=12)
    ap.add_argument("--grid", type=int, default=12)
    args = ap.parse_args()
    out = os.path.join(PROJECT_ROOT, args.run, "test_vis")
    os.makedirs(out, exist_ok=True)

    (table, mask_te, local, off_p, ab_p, ang_p, off_g, ab_g, ang_g, iou) = load_predictions(args)
    paths = group_paths(table, mask_te)
    # rank paths by their mean IoU, keep long ones so a figure is informative
    scored = []
    for key, rows in paths.items():
        if len(rows) < 6:
            continue
        idx = [local[g] for g in rows]
        scored.append((float(np.mean(iou[idx])), key, rows))
    scored.sort(key=lambda t: t[0])
    print(f"test paths: {len(scored)} (>=6 labels), IoU range "
          f"{scored[0][0]:.3f} .. {scored[-1][0]:.3f}")

    # ---- 1) one figure per path, three panels
    picked = (scored[: args.n_paths // 3]                      # worst
              + scored[len(scored) // 2 - args.n_paths // 6: len(scored) // 2 + args.n_paths // 6]
              + scored[-args.n_paths // 3:])                   # best
    seen, index = set(), []
    for s, key, rows in picked:
        if key in seen:
            continue
        seen.add(key)
        index.append({"scene": key[0], "path_id": key[1], "n": len(rows), "iou_mean": s})
        npz = load_scene_npz(os.path.join(PROJECT_ROOT, "data", "processed_v2_1",
                                         f"pilot_{args.tag}", "v2_1", f"{key[0]}.npz"))
        idx = [local[g] for g in rows]
        iou_p = iou[idx]
        fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.4))
        pids, ptids = table["path_id"], table["point_id"]
        draw_path(axes[0], npz, rows, local, off_p, ab_p, ang_p, off_g, ab_g, ang_g, "gt",
                  path_ids=pids, point_ids=ptids)
        axes[0].set_title(f"ground truth  ({len(rows)} labels)", fontsize=11)
        axes[0].legend(loc="upper right", fontsize=8)
        draw_path(axes[1], npz, rows, local, off_p, ab_p, ang_p, off_g, ab_g, ang_g, "pred",
                  fill_pred=True, path_ids=pids, point_ids=ptids)
        axes[1].set_title(f"prediction  (IoU mean {iou_p.mean():.3f}, "
                          f"p50 {np.median(iou_p):.3f}, min {iou_p.min():.3f})", fontsize=11)
        axes[1].legend(loc="upper right", fontsize=8)
        draw_path(axes[2], npz, rows, local, off_p, ab_p, ang_p, off_g, ab_g, ang_g, "both",
                  fill_pred=True, path_ids=pids, point_ids=ptids)
        axes[2].set_title("overlay: blue = GT, orange = prediction, grey arrow = centre error",
                          fontsize=11)
        axes[2].legend(handles=[Patch(facecolor="none", edgecolor="tab:blue", label="GT"),
                                Patch(facecolor="tab:orange", alpha=0.3,
                                      edgecolor="tab:orange", ls="--", label="prediction")],
                       loc="upper right", fontsize=8)
        fig.suptitle(f"{key[0]}   path {key[1]}   -   hold-out map groups", fontsize=13)
        fig.tight_layout()
        name = f"{key[0][:44]}__p{key[1]}.png"
        fig.savefig(os.path.join(out, name), dpi=118)
        plt.close(fig)
    print(f"per-path figures: {len(index)} -> {out}")

    # ---- 2) per-sample GT | pred grid (large)
    pick = np.linspace(0, len(iou) - 1, min(args.grid, len(iou))).astype(int)
    ncol = 4
    nrow = int(math.ceil(len(pick) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.6 * nrow))
    axes = np.atleast_2d(axes)
    te_idx = np.flatnonzero(mask_te)
    for ax, i in zip(axes.ravel(), pick):
        g = te_idx[i]
        sid = str(table["scene_id"][g])
        npz = load_scene_npz(os.path.join(PROJECT_ROOT, "data", "processed_v2_1",
                                         f"pilot_{args.tag}", "v2_1", f"{sid}.npz"))
        sel = (npz["path_id"] == table["path_id"][g]) & \
              (npz["point_id"] == table["point_id"][g])
        j = int(np.flatnonzero(sel)[0])
        p_xy = npz["p_xy"][j].astype(float)
        rs = float(npz["meta"]["r_scene_m"])
        cg = p_xy + off_g[i]
        cp = p_xy + off_p[i]
        win = max(2.0, 1.35 * max(ab_g[i, 0], ab_p[i, 0]))
        bl = npz["blocked"]
        n = bl.shape[0]
        ax.imshow(bl, cmap="gray_r", alpha=0.45, interpolation="nearest",
                  extent=[0, n * rs, n * rs, 0])
        ax.add_patch(Ellipse((cg[0], cg[1]), 2 * ab_g[i, 0], 2 * ab_g[i, 1],
                             angle=math.degrees(ang_g[i]), fill=False, ec="tab:blue", lw=2.0))
        ax.add_patch(Ellipse((cp[0], cp[1]), 2 * ab_p[i, 0], 2 * ab_p[i, 1],
                             angle=math.degrees(ang_p[i]), facecolor="tab:orange",
                             alpha=0.25, ec="tab:orange", lw=2.0, ls="--", fill=True))
        ax.plot(p_xy[0], p_xy[1], "+", ms=11, color="tab:red", mew=2.0)
        ax.set_xlim(cg[0] - win, cg[0] + win)
        ax.set_ylim(cg[1] - win, cg[1] + win)
        ax.set_aspect("equal")
        ax.set_title(f"IoU {iou[i]:.3f}   GT {ab_g[i,0]:.2f}x{ab_g[i,1]:.2f}   "
                     f"pred {ab_p[i,0]:.2f}x{ab_p[i,1]:.2f}\n"
                     f"centre err {np.linalg.norm(off_p[i]-off_g[i]):.2f} m",
                     fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[len(pick):]:
        ax.axis("off")
    fig.suptitle("Test samples: blue = ground truth, orange = prediction (filled)",
                 fontsize=13)
    fig.tight_layout()
    p2 = os.path.join(out, "samples_gt_vs_pred.png")
    fig.savefig(p2, dpi=118)
    plt.close(fig)

    summary = {"n_test": int(mask_te.sum()), "n_test_paths": len(scored),
               "iou_mean": float(iou.mean()), "iou_p50": float(np.median(iou)),
               "iou_min": float(iou.min()), "iou_gt_0_5": float(np.mean(iou > 0.5)),
               "centre_err_mean_m": float(np.linalg.norm(off_p - off_g, axis=1).mean()),
               "paths": index}
    with open(os.path.join(out, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "paths"}, indent=2))
    print(f"-> {p2}\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
