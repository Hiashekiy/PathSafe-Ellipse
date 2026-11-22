"""Visualise model predictions on the full map along A* search paths.

For a few (map, path) pairs of the chosen split this draws:
  * the inflated occupancy map,
  * the A* search path,
  * the ground-truth safe ellipses (green dashed) and
  * the network's predicted ellipses (blue solid),
plus per-path geometric statistics in the title.
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
from matplotlib.patches import Ellipse  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pathsafe_ellipse.config import load_config, project_paths, setup_logging  # noqa: E402
from pathsafe_ellipse.ellipse_validation import ellipse_obstacle_intersections  # noqa: E402
from pathsafe_ellipse.evaluation import ellipse_iou_grid, rebuild_L_cells  # noqa: E402
from pathsafe_ellipse.map_io import build_map_catalog, map_key  # noqa: E402
from pathsafe_ellipse.map_preprocess import preprocess_map  # noqa: E402
from pathsafe_ellipse.model import EllipseNet  # noqa: E402
from pathsafe_ellipse.torch_dataset import SplitData  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--n-paths", type=int, default=4)
    p.add_argument("--margin", type=float, default=60.0, help="zoom margin in cells")
    p.add_argument("--every", type=int, default=0,
                   help="draw every k-th ellipse (0 = auto, keeps ~20 per path)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg, "visualize_predictions")
    pp = project_paths(cfg)
    res = float(cfg["map"]["resolution"])
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    ckpt_path = args.ckpt or os.path.join(pp.outputs_dir, "checkpoints", "ellipse_net_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = EllipseNet(width=int(ckpt.get("width", 32))).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    data = SplitData(os.path.join(pp.splits_dir, f"{args.split}.npz"))
    with open(os.path.join(pp.labels_dir, "map_index.json"), encoding="utf-8") as fh:
        map_index = {int(k): v for k, v in json.load(fh).items()}
    catalog = {map_key(r): r for r in build_map_catalog(pp.map_dir, cfg["map"]["resolution"]) if r.parsed_ok}

    keys = np.stack([data.arrays["map_id"], data.arrays["path_id"]], axis=1)
    uniq, counts = np.unique(keys, axis=0, return_counts=True)
    rng = np.random.default_rng(args.seed)
    # prefer well-populated paths, but keep maps distinct
    order = np.argsort(-counts)
    picked, used_maps = [], set()
    for i in order:
        m = int(uniq[i][0])
        if m in used_maps and len(used_maps) < args.n_paths:
            continue
        picked.append((m, int(uniq[i][1])))
        used_maps.add(m)
        if len(picked) >= args.n_paths:
            break
    if len(picked) < args.n_paths:
        for i in order:
            cand = (int(uniq[i][0]), int(uniq[i][1]))
            if cand not in picked:
                picked.append(cand)
            if len(picked) >= args.n_paths:
                break

    out_dir = os.path.join(pp.outputs_dir, "prediction_visualizations")
    os.makedirs(out_dir, exist_ok=True)

    for map_id, path_id in picked:
        key = map_index[map_id]
        record = catalog[key]
        inflated = preprocess_map(record, pp.map_dir, pp.inflated_maps_dir,
                                  robot_radius=float(cfg["robot"]["radius"]),
                                  kernel_radius=int(cfg["robot"]["inflation_kernel"]),
                                  iterations=int(cfg["robot"]["inflation_iterations"]))
        paths_npz = np.load(os.path.join(pp.paths_dir, f"{key}.npz"))
        flat, offs = paths_npz["flat_paths"], paths_npz["path_offsets"]
        path = flat[offs[path_id]:offs[path_id + 1]].astype(float)

        rows = np.flatnonzero((keys[:, 0] == map_id) & (keys[:, 1] == path_id))
        rows = rows[np.argsort(data.arrays["point_id"][rows])]
        batch = data.batch(rows)
        with torch.no_grad():
            tensors = {k: torch.from_numpy(batch[k]).to(device)
                       for k in ("patch", "rays", "hits", "tangent")}
            center, ab, angle, _, _ = model.predict_ellipse(tensors)
        center = center.cpu().numpy()
        ab = np.sort(ab.cpu().numpy(), axis=1)[:, ::-1]
        angle = angle.cpu().numpy()

        gt_center = batch["center_offset"]
        gt_ab = np.sort(batch["ab"], axis=1)[:, ::-1]
        gt_angle = np.arctan2(batch["sincos"][:, 0], batch["sincos"][:, 1])
        pts = data.arrays["point_xy"][rows].astype(float)

        ious = ellipse_iou_grid((center, ab, angle), (gt_center, gt_ab, gt_angle), res)
        L_pred = rebuild_L_cells(ab[:, 0], ab[:, 1], angle, res)
        centers_cells = np.stack([64.0 + center[:, 0] / res, 64.0 + center[:, 1] / res], axis=1)
        safe = 0
        for bi in range(len(rows)):
            patch = batch["patch"][bi, 0].astype(np.uint8)   # already unpacked (n,1,128,128)
            collides, _, _ = ellipse_obstacle_intersections(L_pred[bi], centers_cells[bi], patch)
            safe += int(not collides)

        every = args.every if args.every > 0 else max(1, int(np.ceil(len(rows) / 20)))
        fig, ax = plt.subplots(figsize=(9.5, 9.0))
        ax.imshow(inflated.occupancy_inflated, cmap="gray_r", interpolation="nearest")
        ax.plot(path[:, 0], path[:, 1], "-", lw=2.6, color="tab:purple",
                label="A* search path", zorder=8, solid_capstyle="round")
        ax.plot(path[0, 0], path[0, 1], "o", ms=9, color="lime", mec="k", label="start", zorder=9)
        ax.plot(path[-1, 0], path[-1, 1], "X", ms=10, color="red", mec="k", label="goal", zorder=9)
        for bi in range(0, len(rows), every):
            # ground truth
            c_gt = pts[bi] + gt_center[bi] / res
            ax.add_patch(Ellipse((c_gt[0], c_gt[1]), width=2 * gt_ab[bi, 0] / res,
                                 height=2 * gt_ab[bi, 1] / res, angle=np.degrees(gt_angle[bi]),
                                 fill=False, ec="tab:green", lw=1.0, ls="--", alpha=0.85, zorder=2))
            # prediction
            c_pr = pts[bi] + center[bi] / res
            ax.add_patch(Ellipse((c_pr[0], c_pr[1]), width=2 * ab[bi, 0] / res,
                                 height=2 * ab[bi, 1] / res, angle=np.degrees(angle[bi]),
                                 fill=False, ec="tab:blue", lw=1.3, alpha=0.95, zorder=5))
            ax.plot(pts[bi, 0], pts[bi, 1], ".", ms=2.2, color="tab:red", zorder=7)
            ax.plot(c_pr[0], c_pr[1], ".", ms=1.8, color="tab:blue", zorder=6)
        ax.plot([], [], color="tab:green", ls="--", lw=1.2,
                label=f"ground-truth ellipse (every {every}th)")
        ax.plot([], [], color="tab:blue", lw=1.3,
                label=f"predicted ellipse (every {every}th)")
        ax.plot([], [], ".", color="tab:red", label="path points")

        margin = args.margin
        ax.set_xlim(max(0, path[:, 0].min() - margin), min(inflated.occupancy_inflated.shape[1],
                                                          path[:, 0].max() + margin))
        ax.set_ylim(min(inflated.occupancy_inflated.shape[0], path[:, 1].max() + margin),
                    max(0, path[:, 1].min() - margin))
        ax.set_title(f"{key}   path {path_id}   ({len(rows)} labelled path points; "
                     f"every {every}th ellipse drawn)\n"
                     f"mean IoU={ious.mean():.3f}    predicted ellipses collision-free = "
                     f"{safe}/{len(rows)} ({100.0 * safe / max(len(rows), 1):.1f}%)", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        out = os.path.join(out_dir, f"pred_{args.split}_{key}_path{path_id:03d}.png")
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"wrote {out}  (points={len(rows)}, IoU={ious.mean():.3f}, safe={safe}/{len(rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
