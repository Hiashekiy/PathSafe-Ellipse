"""Evaluate the trained ellipse network on the held-out test split."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pathsafe_ellipse.config import load_config, project_paths, setup_logging  # noqa: E402
from pathsafe_ellipse.evaluation import (  # noqa: E402
    EllipseCriterion, angle_error_deg, ellipse_iou_grid, evaluate_model,
)
from pathsafe_ellipse.model import EllipseNet  # noqa: E402
from pathsafe_ellipse.torch_dataset import SplitData  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--tag", default="")
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--safety-samples", type=int, default=3000)
    p.add_argument("--n-vis", type=int, default=12)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def plot_predictions(model, data, device, resolution, out_path, n=12, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    rng = np.random.default_rng(seed)
    idx = rng.permutation(data.n)[:n]
    batch = data.batch(idx)
    with torch.no_grad():
        tensors = {k: torch.from_numpy(batch[k]).to(device)
                   for k in ("patch", "rays", "hits", "tangent")}
        center, ab, angle, _, _ = model.predict_ellipse(tensors)
    center, ab, angle = center.cpu().numpy(), ab.cpu().numpy(), angle.cpu().numpy()
    gt_center, gt_angle = batch["center_offset"], np.arctan2(batch["sincos"][:, 0], batch["sincos"][:, 1])
    rows = int(np.ceil(n / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(4 * 4, 4 * rows), squeeze=False)
    for k, i in enumerate(idx):
        ax = axes[k // 4][k % 4]
        patch = np.unpackbits(data.packed[i], axis=1).reshape(128, 128)
        ax.imshow(patch, cmap="gray_r", interpolation="nearest")
        gt_ab_k = np.sort(batch["ab"][k])[::-1]
        iou = float(ellipse_iou_grid((center[k:k + 1], ab[k:k + 1], angle[k:k + 1]),
                                     (gt_center[k:k + 1], gt_ab_k[None, :],
                                      gt_angle[k:k + 1]), resolution)[0])
        for (c, a, b, t, color, ls) in (
            (gt_center[k], gt_ab_k[0], gt_ab_k[1], gt_angle[k], "tab:green", "--"),
            (center[k], ab[k, 0], ab[k, 1], angle[k], "tab:blue", "-"),
        ):
            ax.add_patch(Ellipse((64 + c[0] / resolution, 64 + c[1] / resolution),
                                 width=2 * a / resolution, height=2 * b / resolution,
                                 angle=np.degrees(t), fill=False, ec=color, lw=1.6, ls=ls))
        ax.plot(64, 64, "+", color="red", ms=9)
        ax.set_title(f"IoU={iou:.3f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    for k in range(len(idx), rows * 4):
        axes[k // 4][k % 4].axis("off")
    fig.suptitle("green dashed = ground-truth ellipse, blue = prediction")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg, "evaluate")
    pp = project_paths(cfg)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    res = float(cfg["map"]["resolution"])
    eta = float(cfg["label"]["point_inside_eta"])

    ckpt_path = args.ckpt or os.path.join(pp.outputs_dir, "checkpoints", f"ellipse_net_best{args.tag}.pt")
    if not os.path.exists(ckpt_path):
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        return 2
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = EllipseNet(width=int(ckpt.get("width", 32))).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"loaded {ckpt_path} (epoch {ckpt.get('epoch')})")

    results = {}
    for split in (args.split, "val"):
        path = os.path.join(pp.splits_dir, f"{split}.npz")
        if not os.path.exists(path):
            continue
        data = SplitData(path)
        metrics = evaluate_model(
            model, data, device, res, batch_size=args.batch_size,
            max_samples=args.max_samples, eta=eta, safety_check=True,
            max_safety_samples=args.safety_samples, criterion=EllipseCriterion())
        results[split] = metrics
        print(f"[{split}] n={metrics['n_samples']} loss={metrics['loss']:.4f} "
              f"center={metrics['center_err_m']:.3f}m axes=({metrics['a_err_m']:.3f},"
              f"{metrics['b_err_m']:.3f})m angle={metrics['angle_err_deg']:.1f}deg "
              f"IoU={metrics['iou']:.4f} rho_ok={metrics['rho_ok_rate']:.4f} "
              f"safe={metrics['safety_rate']:.4f} safe+inside={metrics['safe_and_contains_rate']:.4f}")
        if split == args.split:
            plot_predictions(model, data, device, res,
                             os.path.join(pp.reports_dir, f"test_predictions{args.tag}.png"),
                             n=args.n_vis)

    out_json = os.path.join(pp.reports_dir, f"test_metrics{args.tag}.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"metrics -> {out_json}")

    if args.split in results:
        m = results[args.split]
        md = [
            "# PathSafe-Ellipse :: network training & test report\n",
            f"checkpoint: `{os.path.basename(ckpt_path)}` (epoch {ckpt.get('epoch')})\n",
            "## Held-out test split (unseen map groups)\n",
            "| metric | value |", "|---|---|",
            f"| test samples | {m['n_samples']} |",
            f"| composite loss | {m['loss']:.4f} |",
            f"| centre offset error | {m['center_err_m']:.3f} m |",
            f"| semi-axis a error | {m['a_err_m']:.3f} m |",
            f"| semi-axis b error | {m['b_err_m']:.3f} m |",
            f"| orientation error (angle_valid samples) | {m['angle_err_deg']:.2f} deg |",
            f"| ellipse IoU (patch raster) | {m['iou']:.4f} |",
            f"| path point inside prediction (rho <= {eta}) | {m['rho_ok_rate']:.4f} |",
            f"| predicted ellipse collision-free | {m['safety_rate']:.4f} ({m['safety_checked']} checked) |",
            f"| **safe AND contains path point** | **{m['safe_and_contains_rate']:.4f}** |",
        ]
        if "val" in results:
            v = results["val"]
            md += ["\n## Validation split\n", "| metric | value |", "|---|---|",
                   f"| val samples | {v['n_samples']} |",
                   f"| composite loss | {v['loss']:.4f} |",
                   f"| centre offset error | {v['center_err_m']:.3f} m |",
                   f"| ellipse IoU | {v['iou']:.4f} |",
                   f"| safe AND contains path point | {v['safe_and_contains_rate']:.4f} |"]
        md.append("\n预测可视化：`reports/test_predictions.png`；训练曲线：`reports/training_curves.png`。\n")
        md_path = os.path.join(pp.reports_dir, f"training_report{args.tag}.md")
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(md) + "\n")
        print(f"report -> {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
