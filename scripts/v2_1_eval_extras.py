"""Extra evaluation: stratified metrics, constant baseline, and safety of predictions.

Safety uses the SAME exact geometric test as label generation (blocked cells taken
straight from the scene NPZ), so a predicted ellipse is only counted safe if it is
provably collision free at the predicted centre.
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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from pathsafe_ellipse.v2.dataset_v21 import (  # noqa: E402
    Standardiser, build_table, load_split_files, split_by_group,
)
from pathsafe_ellipse.v2.ellipse import exact_clearance  # noqa: E402
from pathsafe_ellipse.v2.pilot import load_scene_npz  # noqa: E402
from v2_1_train import V21Net, ellipse_iou, to_ellipse  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dataset_v21")
    ap.add_argument("--run", default="runs_v21")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--safety-samples", type=int, default=400)
    args = ap.parse_args()

    ckpt = torch.load(os.path.join(PROJECT_ROOT, args.run, "model.pt"),
                      map_location="cpu", weights_only=False)
    model = V21Net()
    model.load_state_dict(ckpt["model"])
    model.eval()
    lidar_std = Standardiser(np.array(ckpt["lidar_std"]["mean"]),
                             np.array(ckpt["lidar_std"]["std"]))
    extra_std = Standardiser(np.array(ckpt["extra_std"]["mean"]),
                            np.array(ckpt["extra_std"]["std"]))
    off_std = Standardiser(np.array(ckpt["off_std"]["mean"]),
                          np.array(ckpt["off_std"]["std"]))

    files = load_split_files(os.path.join(PROJECT_ROOT, "data", "processed_v2_1",
                                         f"pilot_{args.tag}", "v2_1"))
    table = build_table(files, with_patch=True, only_valid=True)
    mask_tr, mask_va, mask_te = split_by_group(table["group"], seed=args.seed)

    def predict(mask):
        with torch.no_grad():
            pred = model(torch.from_numpy(table["patch"][mask]).float(),
                         torch.from_numpy(lidar_std(table["lidar"][mask])).float(),
                         torch.from_numpy(extra_std(table["clearance_p"][mask])).float()
                         ).numpy()
        off, ab, ang = to_ellipse(pred)
        return off * off_std.std + off_std.mean, ab, ang

    off_p, ab_p, ang_p = predict(mask_te)
    off_g = table["offset"][mask_te]
    ab_g = np.exp(table["log_axes"][mask_te])
    ang2 = table["angle2"][mask_te]
    ang_g = 0.5 * np.arctan2(ang2[:, 1], ang2[:, 0])
    iou = ellipse_iou(off_p, ab_p, ang_p, off_g, ab_g, ang_g)
    area_g = np.pi * ab_g[:, 0] * ab_g[:, 1]
    area_p = np.pi * ab_p[:, 0] * ab_p[:, 1]
    aspect_g = ab_g[:, 0] / np.maximum(ab_g[:, 1], 1e-6)
    ang_err = np.degrees(np.abs(np.arctan2(np.sin(ang_p - ang_g), np.cos(ang_p - ang_g))))

    # ---- constant baseline (predict the training means)
    tr_area = np.pi * np.exp(table["log_axes"][mask_tr, 0]) * np.exp(table["log_axes"][mask_tr, 1])
    base_iou = ellipse_iou(np.tile(table["offset"][mask_tr].mean(0), (len(off_g), 1)),
                           np.tile(np.exp(table["log_axes"][mask_tr].mean(0)), (len(off_g), 1)),
                           np.full(len(off_g), 0.0), off_g, ab_g, ang_g)

    def strat(name, bounds, key):
        rows = []
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            m = (key >= lo) & (key < hi)
            if m.sum() == 0:
                continue
            rows.append({"range": [float(lo), float(hi)], "n": int(m.sum()),
                         "iou_mean": float(iou[m].mean()),
                         "centre_err_m": float(np.linalg.norm(off_p[m] - off_g[m], axis=1).mean()),
                         "area_ratio": float(np.median(area_p[m] / np.maximum(area_g[m], 1e-6))),
                         "angle_err_deg": float(ang_err[m].mean())})
        return {"by": name, "rows": rows}

    strata = [strat("gt_area_m2", [0, 5, 20, 60, 200], area_g),
              strat("gt_aspect", [1.0, 1.2, 1.6, 2.5, 10.0], aspect_g)]
    elongated = aspect_g > 1.6
    angle_elongated = float(ang_err[elongated].mean()) if elongated.any() else None

    # ---- safety of the predicted ellipses (exact test on the stored blocked grid)
    rng = np.random.default_rng(0)
    files_te = [f for f, g in zip(files, [True] * len(files))]
    by_scene = {}
    for f in files:
        npz = load_scene_npz(f)
        by_scene[str(npz["meta"]["scene_id"])] = npz
    scenes_te = sorted(set(table["scene_id"][mask_te].tolist()))
    test_idx = np.flatnonzero(mask_te)
    idx = rng.choice(test_idx, size=min(args.safety_samples, int(mask_te.sum())),
                     replace=False)
    safe_p = safe_g = 0
    n_checked = 0
    pos_of = {int(g): i for i, g in enumerate(test_idx)}
    for k in idx:
        i_te = pos_of[int(k)]
        sid = str(table["scene_id"][k])
        npz = by_scene.get(sid)
        if npz is None:
            continue
        rs = float(npz["meta"]["r_scene_m"])
        # locate the row inside that scene by path_id/point_id
        sel = (npz["path_id"] == table["path_id"][k]) & (npz["point_id"] == table["point_id"][k])
        if not np.any(sel):
            continue
        j = int(np.flatnonzero(sel)[0])
        p_xy = npz["p_xy"][j].astype(float)
        blocked = npz["blocked"].astype(np.uint8)
        # predicted ellipse
        c, s_ = math.cos(ang_p[i_te]), math.sin(ang_p[i_te])
        R = np.array([[c, -s_], [s_, c]])
        S = R @ np.diag(ab_p[i_te] ** 2) @ R.T
        ok_p, _, _, _ = exact_clearance(S, p_xy + off_p[i_te], blocked, rs)
        Sg = npz["ellipse_S"][j].reshape(2, 2)
        ok_g, _, _, _ = exact_clearance(Sg, npz["q_xy"][j].astype(float), blocked, rs)
        safe_p += int(ok_p)
        safe_g += int(ok_g)
        n_checked += 1
    safety = {"n_checked": n_checked,
              "pred_safe_rate": safe_p / max(n_checked, 1),
              "gt_safe_rate": safe_g / max(n_checked, 1)}

    out = {"test": {"n": int(mask_te.sum()),
                    "iou_mean": float(iou.mean()), "iou_p50": float(np.median(iou)),
                    "iou_gt_0_5": float(np.mean(iou > 0.5)),
                    "centre_err_mean_m": float(np.linalg.norm(off_p - off_g, axis=1).mean()),
                    "area_ratio_p50": float(np.median(area_p / np.maximum(area_g, 1e-6))),
                    "angle_err_mean_deg": float(ang_err.mean()),
                    "angle_err_median_deg": float(np.median(ang_err)),
                    "angle_err_mean_deg_aspect_gt_1_6": angle_elongated},
           "constant_baseline": {"iou_mean": float(base_iou.mean()),
                                 "iou_p50": float(np.median(base_iou)),
                                 "train_area_p50": float(np.median(tr_area))},
           "strata": strata, "safety": safety}
    print(f"test IoU {out['test']['iou_mean']:.4f} vs constant baseline "
          f"{out['constant_baseline']['iou_mean']:.4f}")
    print(f"angle error: mean {out['test']['angle_err_mean_deg']:.1f}° "
          f"median {out['test']['angle_err_median_deg']:.1f}° | elongated ellipses "
          f"(aspect>1.6, n={int(elongated.sum())}) {angle_elongated:.1f}°")
    for s in strata:
        print(f"  by {s['by']}:")
        for r in s["rows"]:
            print(f"    {r['range']} n={r['n']:5d} IoU {r['iou_mean']:.3f} "
                  f"centre {r['centre_err_m']:.2f} m area ratio {r['area_ratio']:.2f} "
                  f"angle {r['angle_err_deg']:.1f}°")
    print(f"safety of predictions: {safety['pred_safe_rate']:.3f} "
          f"({safety['n_checked']} checked) vs GT labels {safety['gt_safe_rate']:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    rows = strata[0]["rows"]
    axes[0].bar([f"{r['range'][0]:.0f}-{r['range'][1]:.0f}" for r in rows],
                [r["iou_mean"] for r in rows])
    axes[0].axhline(out["constant_baseline"]["iou_mean"], color="r", ls="--", lw=1,
                    label="constant baseline")
    axes[0].set_xlabel("GT area [m2]"); axes[0].set_ylabel("IoU"); axes[0].legend()
    axes[0].set_title("IoU by area stratum"); axes[0].grid(alpha=0.3)
    rows2 = strata[1]["rows"]
    axes[1].bar([f"{r['range'][0]:.1f}-{r['range'][1]:.1f}" for r in rows2],
                [r["angle_err_deg"] for r in rows2])
    axes[1].set_xlabel("GT aspect ratio a/b"); axes[1].set_ylabel("mean angle error [deg]")
    axes[1].set_title("angle error by elongation"); axes[1].grid(alpha=0.3)
    axes[2].bar(["prediction", "GT label"], [safety["pred_safe_rate"], safety["gt_safe_rate"]])
    axes[2].set_ylim(0, 1.05); axes[2].set_ylabel("exact-verified safe rate")
    axes[2].set_title(f"predicted ellipse safety (n={n_checked})"); axes[2].grid(alpha=0.3)
    fig.suptitle(f"V2.1 baseline evaluation - {args.tag}")
    fig.tight_layout()
    dst_fig = os.path.join(PROJECT_ROOT, args.run, "evaluation_extras.png")
    fig.savefig(dst_fig, dpi=115)
    plt.close(fig)
    dst = os.path.join(PROJECT_ROOT, args.run, "metrics_extras.json")
    json.dump(out, open(dst, "w", encoding="utf-8"), indent=2)
    print(f"-> {dst}\n-> {dst_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
