"""Prediction vs ground truth — minimal, unambiguous colouring.

    python pathsafe_ellipse_net/visualize.py --checkpoint runs_v3/full_50k_e100/best.pt --n 4

Two panels per trajectory, nothing else:

    left   GT   : blue ellipses, blue dashed centre polyline
    right  pred : orange ellipses, orange solid centre polyline

No collision encoding, no GT/pred overlay in one axes, at most 10 ellipses per panel
(uniformly subsampled) so the lines never pile up.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from pathsafe_ellipse_net.datasets import (TrajectoryDataset, collate_paths,  # noqa: E402
                                          list_trajectories, split_by_group)
from pathsafe_ellipse_net.evaluation.metrics import ellipse_iou_grid  # noqa: E402
from pathsafe_ellipse_net.models import EllipseNet  # noqa: E402
from pathsafe_ellipse_net.train import groups_for, resolve_device  # noqa: E402

GT_COLOR = "#1f77b4"        # blue
PRED_COLOR = "#ff7f0e"      # orange
PATH_COLOR = "0.45"
MAX_ELLIPSES = 10


def _pick(n: int, valid: np.ndarray, k: int = MAX_ELLIPSES):
    idx = [i for i in np.linspace(0, n - 1, min(k, n)).astype(int) if valid[i]]
    return idx


def _scene(ax, item, title):
    occ = np.asarray(item["occupancy"])[0]
    rs = float(item["r_scene"])
    ext = [0, occ.shape[1] * rs, occ.shape[0] * rs, 0]
    ax.imshow(occ, cmap="gray_r", alpha=0.5, interpolation="nearest", extent=ext)
    p = np.asarray(item["path_xy"])
    ax.plot(p[:, 0], p[:, 1], "-", lw=1.2, color=PATH_COLOR, zorder=3)
    ax.plot(p[:, 0], p[:, 1], "o", ms=2.4, color=PATH_COLOR, zorder=3)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--trajectory-dir", default="")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = blob["cfg"]
    if args.trajectory_dir:
        cfg["data"]["trajectory_dir"] = os.path.abspath(args.trajectory_dir)
    files = list_trajectories(cfg["data"]["trajectory_dir"])
    groups = groups_for(files)
    tr, va, te = split_by_group(groups, frac=tuple(cfg["data"]["split_frac"]),
                                seed=int(cfg["data"]["split_seed"]))
    idx = {"train": tr, "val": va, "test": te}[args.split][: args.n]
    device = resolve_device(cfg["train"]["device"])
    model = EllipseNet(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    ds = TrajectoryDataset([files[i] for i in idx], cfg)
    tag = os.path.basename(os.path.dirname(args.checkpoint))
    out_dir = args.out or os.path.join(PROJECT_ROOT, "outputs_v3", f"net_viz_{tag}")
    os.makedirs(out_dir, exist_ok=True)

    records = []
    for k in range(len(idx)):
        item = ds[k]
        b = {j: (v.to(device) if torch.is_tensor(v) else v)
             for j, v in collate_paths([item]).items()}
        with torch.no_grad():
            out = model(b["occupancy"], b["path_xy"], b["lidar"] / model.max_range_m,
                        b["geometry"], b["r_scene"], b["mask"], lidar_m=b["lidar"],
                        map_index=b.get("map_index"))
        m = int(np.asarray(item["path_xy"]).shape[0])
        p = np.asarray(item["path_xy"])
        gt_c = p + np.asarray(item["center_offset"])
        gt_ab, gt_ang = np.asarray(item["ab"]), np.asarray(item["angle"])
        pr_c = out["center"][0, :m].cpu().numpy()
        pr_ab, pr_ang = out["axes"][0, :m].cpu().numpy(), out["angle"][0, :m].cpu().numpy()
        valid = np.asarray(item["valid"])
        ce = np.linalg.norm(pr_c - gt_c, axis=1)
        ious = [ellipse_iou_grid(pr_c[i], out["shape_matrix"][0, i].cpu().numpy(), gt_c[i],
                                 np.asarray(item["S"])[i]) for i in np.flatnonzero(valid)]
        keep = _pick(m, valid)

        fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.4))
        _scene(axes[0], item, "GT (blue)")
        for i in keep:
            axes[0].add_patch(Ellipse(gt_c[i], 2 * gt_ab[i, 0], 2 * gt_ab[i, 1],
                                      angle=np.degrees(gt_ang[i]), fill=False,
                                      ec=GT_COLOR, lw=1.6, zorder=5))
        axes[0].plot(gt_c[:, 0], gt_c[:, 1], "--", lw=2.0, color=GT_COLOR, zorder=6)

        _scene(axes[1], item, "prediction (orange)")
        for i in keep:
            axes[1].add_patch(Ellipse(pr_c[i], 2 * pr_ab[i, 0], 2 * pr_ab[i, 1],
                                      angle=np.degrees(pr_ang[i]), fill=False,
                                      ec=PRED_COLOR, lw=1.6, zorder=5))
        axes[1].plot(pr_c[:, 0], pr_c[:, 1], "-", lw=2.0, color=PRED_COLOR, zorder=6)

        handles = [Line2D([], [], color=PATH_COLOR, lw=1.2, label="A* path + query points"),
                   Line2D([], [], color=GT_COLOR, lw=1.6, ls="--", label="GT ellipse / centre"),
                   Line2D([], [], color=PRED_COLOR, lw=1.6, ls="-", label="prediction ellipse / centre")]
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10, frameon=False)
        fig.suptitle(f"{item['scene_id'][:34]} p{item['path_id']}   N={m}   "
                     f"median centre err {np.median(ce[valid]):.2f} m   "
                     f"mean IoU {np.mean(ious) if ious else float('nan'):.2f}",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0.06, 1, 0.94))
        stem = os.path.splitext(item["file"])[0]
        fig.savefig(os.path.join(out_dir, f"gt_pred_{stem}.png"), dpi=115)
        plt.close(fig)
        records.append({"file": item["file"], "n_points": m, "valid": int(valid.sum()),
                        "median_center_error": float(np.median(ce[valid])),
                        "mean_iou": float(np.mean(ious)) if ious else None})
        print(f"  {item['file'][:52]:52s} ctr {np.median(ce[valid]):.2f} m  "
              f"IoU {np.mean(ious) if ious else float('nan'):.2f}")
    with open(os.path.join(out_dir, f"summary_{args.split}.json"), "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
