"""Visualise trained predictions against ground-truth labels on held-out scenes."""
from __future__ import annotations

import argparse
import glob
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
from v2_1_train_full import Net, S_from, ellipse_iou, safe_shrink, to_ellipse  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dataset_v21")
    ap.add_argument("--run", default="runs_v21_full")
    ap.add_argument("--n-scenes", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "runs_v21_full",
                                                  "predictions_vs_gt.png"))
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
    # refit the same standardisers used in training
    lidar_std = Standardiser.fit(table["lidar"][mask_tr])
    extra_std = Standardiser.fit(table["clearance_p"][mask_tr])
    off_std = Standardiser.fit(table["offset"][mask_tr])

    test_scenes = [s for s in sorted(set(table["scene_id"][mask_te].tolist()))][: args.n_scenes]
    fig, axes = plt.subplots(2, len(test_scenes) // 2 if len(test_scenes) > 1 else 1,
                             figsize=(6.2 * max(1, len(test_scenes) // 2), 11.0))
    axes = np.atleast_2d(axes)
    stats = []
    for ax, sid in zip(axes.ravel(), test_scenes):
        m = (table["scene_id"] == sid) & mask_te
        with torch.no_grad():
            pred = model(torch.from_numpy(table["patch"][m]).float(),
                         torch.from_numpy(lidar_std(table["lidar"][m])).float(),
                         torch.from_numpy(extra_std(table["clearance_p"][m])).float()
                         ).numpy()
        off_p, ab_p, ang_p = to_ellipse(pred)
        off_p = off_p * off_std.std + off_std.mean
        ab_g = np.exp(table["log_axes"][m])
        a2 = table["angle2"][m]
        ang_g = 0.5 * np.arctan2(a2[:, 1], a2[:, 0])
        iou = ellipse_iou(off_p, ab_p, ang_p, table["offset"][m], ab_g, ang_g)

        npz = load_scene_npz(os.path.join(PROJECT_ROOT, "data", "processed_v2_1",
                                         f"pilot_{args.tag}", "v2_1", f"{sid}.npz"))
        rs = float(npz["meta"]["r_scene_m"])
        bl = npz["blocked"]
        n = bl.shape[0]
        ext = [0, n * rs, n * rs, 0]
        ax.imshow(bl, cmap="gray_r", alpha=0.5, interpolation="nearest", extent=ext)
        pids = table["path_id"][m]
        pts = table["point_id"][m]
        safe_raw = safe_proj = 0
        for k in range(len(off_p)):
            sel = (npz["path_id"] == pids[k]) & (npz["point_id"] == pts[k])
            if not np.any(sel):
                continue
            j = int(np.flatnonzero(sel)[0])
            p_xy = npz["p_xy"][j].astype(float)
            c = p_xy + off_p[k]                       # predicted centre
            cg = npz["q_xy"][j].astype(float)
            ax.plot([p_xy[0], cg[0]], [p_xy[1], cg[1]], "-", lw=0.5, color="tab:gray", alpha=0.6)
            ax.add_patch(Ellipse((cg[0], cg[1]), 2 * ab_g[k, 0], 2 * ab_g[k, 1],
                                 angle=math.degrees(ang_g[k]), fill=False,
                                 ec="tab:blue", lw=1.0))
            ax.add_patch(Ellipse((c[0], c[1]), 2 * ab_p[k, 0], 2 * ab_p[k, 1],
                                 angle=math.degrees(ang_p[k]), fill=False,
                                 ec="tab:orange", lw=1.0, ls="--"))
            ok_raw = safe_shrink(ab_p[k], ang_p[k], c, bl.astype(np.uint8), rs)[0] >= 1.0 - 1e-9
            g = safe_shrink(ab_p[k], ang_p[k], c, bl.astype(np.uint8), rs)[0]
            safe_raw += int(ok_raw)
            safe_proj += int(g > 0)
            if g < 1.0 - 1e-9:
                ax.add_patch(Ellipse((c[0], c[1]), 2 * ab_p[k, 0] * g, 2 * ab_p[k, 1] * g,
                                     angle=math.degrees(ang_p[k]), fill=False,
                                     ec="tab:green", lw=0.9, ls=":"))
            ax.plot(p_xy[0], p_xy[1], ".", ms=2, color="tab:red")
        nn_ = len(off_p)
        ax.set_title(f"{sid[:34]}\nlabels {nn_}, IoU mean {iou.mean():.3f}, "
                     f"raw safe {safe_raw / max(nn_, 1):.2f} -> projected "
                     f"{safe_proj / max(nn_, 1):.2f}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        stats.append({"scene": sid, "n": int(nn_), "iou_mean": float(iou.mean()),
                      "safe_raw": safe_raw / max(nn_, 1),
                      "safe_proj": safe_proj / max(nn_, 1)})
    for ax in axes.ravel()[len(test_scenes):]:
        ax.axis("off")
    fig.suptitle("V2.1 predictions on held-out scenes  -  blue = GT label, "
                 "orange dashed = prediction, green dotted = projected-safe prediction, "
                 "red dots = A* stations p_i")
    fig.tight_layout()
    fig.savefig(args.out, dpi=115)
    plt.close(fig)
    print(json.dumps(stats, indent=2))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
