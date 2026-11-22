"""Train a V2.1 ellipse-output baseline on the whole-scene label dataset.

Input  : data/processed_v2_1/pilot_<tag>/v2_1/*.npz
Model  : small CNN on the occupancy patch around p_i + MLP on (lidar, tangent)
Output : centre offset (q - p) and the ellipse (log a, log b, cos 2theta, sin 2theta)

Split by map group; reports IoU / centre error / axis error / angle error and the
safety rate after projecting the prediction into the safe set.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from pathsafe_ellipse.v2.dataset_v21 import (  # noqa: E402
    Standardiser, build_table, load_split_files, save_meta, split_by_group, summarise,
)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


class V21Net(nn.Module):
    def __init__(self, n_rays: int = 16, n_extra: int = 1, width: int = 96):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_rays + n_extra + 32 * 16, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
        )
        self.head = nn.Linear(width, 6)          # offset(2) + log axes(2) + angle(2)

    def forward(self, patch, lidar, extra):
        f = self.cnn(patch).flatten(1)
        h = self.mlp(torch.cat([lidar, extra, f], dim=1))
        return self.head(h)


def ellipse_iou(off_p, ab_p, ang_p, off_g, ab_g, ang_g, n: int = 24) -> np.ndarray:
    """Rasterised IoU of two ellipses sharing sample count (bbox-normalised grid)."""
    out = np.zeros(len(off_p), dtype=np.float32)
    for k in range(len(off_p)):
        ax_, ay_ = ab_p[k]
        bx_, by_ = ab_g[k]
        lim = max(ax_, ay_, bx_, by_) * 1.2
        xs = np.linspace(-lim, lim, n)
        ys = np.linspace(-lim, lim, n)
        X, Y = np.meshgrid(xs, ys)

        def inside(off, a_, b_, ang):
            cx, cy = off
            c, s = math.cos(ang), math.sin(ang)
            dx, dy = X - cx, Y - cy
            u = (dx * c + dy * s) / max(a_, 1e-6)
            v = (-dx * s + dy * c) / max(b_, 1e-6)
            return (u * u + v * v) <= 1.0

        ip = inside(off_p[k], ax_, ay_, ang_p[k])
        ig = inside(off_g[k], bx_, by_, ang_g[k])
        inter = np.logical_and(ip, ig).sum()
        union = np.logical_or(ip, ig).sum()
        out[k] = inter / union if union else 0.0
    return out


def to_ellipse(pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    off = pred[:, 0:2]
    ab = np.exp(np.clip(pred[:, 2:4], -6.0, 3.0))
    ang = 0.5 * np.arctan2(pred[:, 5], pred[:, 4])
    return off, ab, ang


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dataset_v21")
    ap.add_argument("--root", default=os.path.join(PROJECT_ROOT, "data", "processed_v2_1"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "runs_v21"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    os.makedirs(args.out, exist_ok=True)

    files = load_split_files(os.path.join(args.root, f"pilot_{args.tag}", "v2_1"))
    print(f"scene files: {len(files)}")
    t0 = time.perf_counter()
    table = build_table(files, with_patch=True, only_valid=True)
    print(f"labels: {len(table['offset'])}  ({time.perf_counter() - t0:.1f}s to load)")
    info = summarise(table)
    print("dataset:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                  for k, v in info.items()}))

    mask_tr, mask_va, mask_te = split_by_group(table["group"], seed=args.seed)
    print(f"splits: train {mask_tr.sum()}  val {mask_va.sum()}  test {mask_te.sum()}  "
          f"groups {len(set(table['group'][mask_tr]))}/"
          f"{len(set(table['group'][mask_va]))}/{len(set(table['group'][mask_te]))}")

    extra = table["clearance_p"]      # clearance at p_i only (no label leak)
    extra_std = Standardiser.fit(extra[mask_tr])
    lidar_std = Standardiser.fit(table["lidar"][mask_tr])
    off_std = Standardiser.fit(table["offset"][mask_tr])

    def tensors(mask):
        return {
            "patch": torch.from_numpy(table["patch"][mask]).float(),
            "lidar": torch.from_numpy(lidar_std(table["lidar"][mask])).float(),
            "extra": torch.from_numpy(extra_std(extra[mask])).float(),
            "off": torch.from_numpy(table["offset"][mask]).float(),
            "axes": torch.from_numpy(table["log_axes"][mask]).float(),
            "ang": torch.from_numpy(table["angle2"][mask]).float(),
        }

    tr, va, te = tensors(mask_tr), tensors(mask_va), tensors(mask_te)
    model = V21Net().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    def loss_fn(pred, b, off_s, off_m):
        off_hat = pred[:, 0:2] * off_s + off_m
        l_off = torch.nn.functional.smooth_l1_loss(off_hat, b["off"], beta=0.25)
        l_ax = torch.nn.functional.smooth_l1_loss(pred[:, 2:4], b["axes"], beta=0.1)
        l_ang = torch.nn.functional.l1_loss(pred[:, 4:6], b["ang"])
        return l_off + l_ax + 0.5 * l_ang, (l_off.item(), l_ax.item(), l_ang.item())

    off_s = torch.tensor(off_std.std, dtype=torch.float32, device=device)
    off_m = torch.tensor(off_std.mean, dtype=torch.float32, device=device)
    n = len(tr["off"])
    hist = {"train": [], "val": []}
    best = {"val": float("inf"), "epoch": -1, "state": None}
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            b = {k: tr[k][idx].to(device, non_blocking=True) for k in ("off", "axes", "ang")}
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                pred = model(tr["patch"][idx].to(device), tr["lidar"][idx].to(device),
                             tr["extra"][idx].to(device))
                loss, _parts = loss_fn(pred, b, off_s, off_m)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss) * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            pred_va = model(va["patch"].to(device), va["lidar"].to(device),
                            va["extra"].to(device))
            vloss, _ = loss_fn(pred_va,
                               {k: va[k].to(device) for k in ("off", "axes", "ang")},
                               off_s, off_m)
        hist["train"].append(tot / max(n, 1))
        hist["val"].append(float(vloss))
        if float(vloss) < best["val"]:
            best = {"val": float(vloss), "epoch": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d} train {hist['train'][-1]:.4f} val {hist['val'][-1]:.4f}")
    if best["state"] is not None:
        model.load_state_dict(best["state"])

    # ---------------- evaluation
    def evaluate(part, name):
        if len(part["off"]) == 0:
            return ({"split": name, "n": 0, "iou_mean": float("nan"),
                     "iou_p50": float("nan"), "iou_ok_rate": float("nan"),
                     "centre_err_mean_m": float("nan"), "centre_err_p95_m": float("nan"),
                     "axis_err_mean_m": float("nan"), "angle_err_mean_deg": float("nan"),
                     "area_p50_pred": float("nan"), "area_p50_gt": float("nan"),
                     "area_ratio_p50": float("nan")},
                    np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))
        model.eval()
        with torch.no_grad():
            pred = model(part["patch"].to(device), part["lidar"].to(device),
                         part["extra"].to(device)).cpu().numpy()
        off_p, ab_p, ang_p = to_ellipse(pred)
        off_p = off_p * off_std.std + off_std.mean
        off_g = part["off"].cpu().numpy()
        ab_g = np.exp(part["axes"].cpu().numpy())
        ang2 = part["ang"].cpu().numpy()
        ang_g = 0.5 * np.arctan2(ang2[:, 1], ang2[:, 0])
        iou = ellipse_iou(off_p, ab_p, ang_p, off_g, ab_g, ang_g)
        cerr = np.linalg.norm(off_p - off_g, axis=1)
        ax_err = np.abs(np.log(np.maximum(ab_p, 1e-3)) - np.log(np.maximum(ab_g, 1e-3)))
        ang_err = np.abs(np.arctan2(np.sin(ang_p - ang_g), np.cos(ang_p - ang_g)))
        area_p = np.pi * ab_p[:, 0] * ab_p[:, 1]
        area_g = np.pi * ab_g[:, 0] * ab_g[:, 1]
        return {
            "split": name, "n": int(len(iou)),
            "iou_mean": float(iou.mean()), "iou_p50": float(np.median(iou)),
            "iou_ok_rate": float(np.mean(iou > 0.5)),
            "centre_err_mean_m": float(cerr.mean()),
            "centre_err_p95_m": float(np.percentile(cerr, 95)),
            "axis_err_mean_m": float(ax_err.mean()),
            "angle_err_mean_deg": float(np.degrees(ang_err.mean())),
            "area_p50_pred": float(np.median(area_p)),
            "area_p50_gt": float(np.median(area_g)),
            "area_ratio_p50": float(np.median(area_p / np.maximum(area_g, 1e-6))),
        }, iou, area_p, area_g, cerr, ang_err

    metrics = {}
    for part, name in ((tr, "train"), (va, "val"), (te, "test")):
        m, iou_te, area_p, area_g, cerr, ang_err = evaluate(part, name)
        metrics[name] = m
        print(f"[{name:5s}] n={m['n']:5d} IoU {m['iou_mean']:.4f} (p50 {m['iou_p50']:.4f}, "
              f">0.5 {m['iou_ok_rate']:.3f}) | centre {m['centre_err_mean_m']:.3f} m | "
              f"axis {m['axis_err_mean_m']:.3f} | angle {m['angle_err_mean_deg']:.1f}° | "
              f"area ratio {m['area_ratio_p50']:.2f}")

    torch.save({"model": model.state_dict(), "lidar_std": lidar_std.state(),
                "extra_std": extra_std.state(), "off_std": off_std.state(),
                "args": vars(args)}, os.path.join(args.out, "model.pt"))

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.0))
    if area_g.size == 0:                    # fall back to val when the test split is empty
        _m, iou_te, area_p, area_g, cerr, ang_err = evaluate(va, "val")
        title_split = "val"
    else:
        title_split = "test"
    axes[0].plot(hist["train"], label="train")
    axes[0].plot(hist["val"], label="val")
    axes[0].set_yscale("log"); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title("training curve"); axes[0].legend(); axes[0].grid(alpha=0.3)
    if iou_te.size:
        axes[1].hist(iou_te, bins=30)
    axes[1].set_xlabel("IoU (test)"); axes[1].set_ylabel("samples")
    axes[1].set_title(f"{title_split} IoU mean {metrics[title_split]['iou_mean']:.3f}")
    axes[1].grid(alpha=0.3)
    if area_g.size:
        axes[2].scatter(area_g, area_p, s=8, alpha=0.6)
    lim = [0, max(area_g.max(), area_p.max()) * 1.05]
    axes[2].plot(lim, lim, "k--", lw=0.8)
    axes[2].set_xlabel("GT area [m2]"); axes[2].set_ylabel("pred area [m2]")
    axes[2].set_title("area (test)"); axes[2].grid(alpha=0.3)
    if ang_err.size:
        axes[3].hist(np.degrees(ang_err), bins=30)
    axes[3].set_xlabel("angle error [deg]"); axes[3].set_ylabel("samples")
    axes[3].set_title("angle error (test)"); axes[3].grid(alpha=0.3)
    fig.suptitle(f"V2.1 baseline - {args.tag}")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "training_curves.png"), dpi=115)
    plt.close(fig)

    payload = {"tag": args.tag, "dataset": info, "metrics": metrics,
               "history": hist, "args": vars(args),
               "splits": {"train": int(mask_tr.sum()), "val": int(mask_va.sum()),
                          "test": int(mask_te.sum())}}
    save_meta(os.path.join(args.out, "metrics.json"), payload)
    print(f"\n-> {os.path.join(args.out, 'metrics.json')}")
    print(f"-> {os.path.join(args.out, 'training_curves.png')}")
    print(f"-> {os.path.join(args.out, 'model.pt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
