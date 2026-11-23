"""Multi-trajectory prediction dashboard for a ``ellipsenet`` checkpoint.

    python ellipsenet/visualize_grid.py --checkpoint checkpoints/train10k_e100/best.pt \
        --split test --n 24 --seed 11 --out-dir results/figures/grid

Writes
    grid_predictions.png   预测：blocked 图 + A* 路径 + 每条轨迹的预测椭圆（绿=精确无碰撞，红=碰撞）
    grid_gt.png            同一批轨迹的 GT 参考（仅诊断）
    per_trajectory.csv     逐轨迹指标（精确安全率、连续性、边界覆盖、面积、净空、越界）
    summary_metrics.png    逐轨迹指标分布 + 汇总
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from ellipsenet.data.dataset import build_loader, load_index, split_by_group  # noqa: E402
from ellipsenet.evaluation.metrics import exact_trajectory  # noqa: E402
from ellipsenet.geometry.polytopes import (build_polytopes,  # noqa: E402
                                           polygon_from_halfspaces)
from ellipsenet.models.psenet import PSENet  # noqa: E402


def ellipse_xy(c, a, b, theta, m=72):
    t = np.linspace(0.0, 2.0 * np.pi, m)
    x = c[0] + a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
    y = c[1] + a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)
    return x, y


@torch.no_grad()
def collect(model, loader, device, regions: bool = True,
            window_margin: float = 3.0) -> List[Dict[str, object]]:
    out_rows: List[Dict[str, object]] = []
    for batch in loader:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(b, tau=1.0)
        dec = model.decode_argmax(out)
        mask = b["mask"].cpu().numpy()
        center = dec["center"].cpu().numpy()
        a = dec["a"].cpu().numpy()
        bb = dec["b"].cpu().numpy()
        th = dec["theta"].cpu().numpy()
        prob = dec["prob"].cpu().numpy()
        occ = b["occupancy"].cpu().numpy()[:, 0]
        path = b["path_xy"].cpu().numpy()
        res = b["r_scene"].cpu().numpy()
        gt_c = b["diag_center"].cpu().numpy()
        gt_ab = b["diag_ab"].cpu().numpy()
        gt_th = b["diag_angle"].cpu().numpy()
        gt_v = b["diag_valid"].cpu().numpy() > 0.5
        for i in range(mask.shape[0]):
            n = int(mask[i].sum())
            if n < 2:
                continue
            ex = exact_trajectory(center[i, :n], a[i, :n], bb[i, :n], th[i, :n],
                                  occ[i].astype(np.uint8), float(res[i]),
                                  rho_max=0.85)
            # convex safe region per predicted ellipse (one batched call per trajectory)
            polys = None
            if regions:
                polys = build_polytopes(
                    torch.as_tensor(center[i, :n], device=device),
                    torch.as_tensor(a[i, :n], device=device),
                    torch.as_tensor(bb[i, :n], device=device),
                    torch.as_tensor(th[i, :n], device=device),
                    torch.as_tensor(occ[i], device=device), float(res[i]),
                    window_margin_m=window_margin)
            out_rows.append({
                "scene_id": batch["meta"]["scene_id"][i],
                "file": batch["meta"]["file"][i],
                "n_points": n,
                "resolution_m": float(res[i]),
                "occupancy": occ[i],
                "path": path[i, :n],
                "center": center[i, :n],
                "a": a[i, :n], "b": bb[i, :n], "theta": th[i, :n],
                "prob": prob[i, :n],
                "gt_center": gt_c[i, :n], "gt_ab": gt_ab[i, :n],
                "gt_theta": gt_th[i, :n], "gt_valid": gt_v[i, :n],
                "regions": ([polygon_from_halfspaces(*polys.planes(k)) for k in range(n)]
                            if polys is not None else None),
                # a region is drawable when it is obstacle free and still anchored on the
                # ellipse centre; "regions_full" only says it also contains the whole ellipse
                "regions_ok": (np.logical_and(polys.converged.cpu().numpy(),
                                               polys.centre_ok.cpu().numpy())
                                if polys is not None else None),
                "regions_full": (np.logical_and(polys.converged.cpu().numpy(),
                                                 polys.feasible.cpu().numpy())
                                  if polys is not None else None),
                "safe": ex["safe"],
                "pair_ok": ex["pair_ok"],
                "pair_cover": ex["pair_cover"],
                "border": ex["border"],
                "corridor_ok": bool(ex["safe"].all() and (ex["pair_ok"].all()
                                                         if len(ex["pair_ok"]) else True)),
            })
    return out_rows


def draw_panel(ax, row: Dict[str, object], mode: str, idx: int,
               plain: bool = False, titles: bool = True) -> None:
    occ = row["occupancy"]
    res = float(row["resolution_m"])
    H, W = occ.shape
    extent = [0.0, (W - 1) * res, (H - 1) * res, 0.0]
    if plain:
        ax.set_facecolor("white")
    else:
        ax.imshow(1.0 - occ, cmap="gray", extent=extent, vmin=0, vmax=1)
    path = row["path"]
    ax.plot(path[:, 0], path[:, 1], "-", color="tab:blue", lw=1.1, zorder=4)
    if mode == "pred":
        regs = row.get("regions")
        n_noreg = 0
        if regs:
            for i, poly in enumerate(regs):
                if not bool(row["regions_ok"][i]):
                    # the ellipse itself collides, so no obstacle-free convex region exists.
                    # Drawing the half-built polytope here would be a meaningless giant box;
                    # mark the ellipse instead and let the title count them.
                    n_noreg += 1
                    c = row["center"][i]
                    ax.plot([c[0]], [c[1]], "x", color="tab:red", ms=2.6, mew=0.6,
                            zorder=6)
                    continue
                if len(poly) < 3:
                    continue
                ax.fill(poly[:, 0], poly[:, 1], color="tab:orange", alpha=0.13,
                        lw=0, zorder=2)
                ax.plot(np.r_[poly[:, 0], poly[0, 0]], np.r_[poly[:, 1], poly[0, 1]],
                        "-", color="tab:orange", lw=0.45, alpha=0.8, zorder=2)
        for i in range(row["n_points"]):
            x, y = ellipse_xy(row["center"][i], row["a"][i], row["b"][i], row["theta"][i])
            ax.plot(x, y, "-", lw=0.7,
                    color="tab:green" if row["safe"][i] else "tab:red", zorder=3)
        ax.plot(row["center"][:, 0], row["center"][:, 1], "--", color="tab:purple",
                lw=0.8, zorder=5)
        if titles:
            safe = float(np.mean(row["safe"]))
            cover = np.mean(row["pair_cover"]) if len(row["pair_cover"]) else 0.0
            nreg = (row["n_points"] - n_noreg) if regs else 0
            nfull = int(np.sum(row["regions_full"])) if regs else 0
            ax.set_title(f"{idx:02d} {row['scene_id'][:18]}\n"
                         f"safe {safe:.2f} | cover {cover:.2f} | N={row['n_points']}"
                         + (f" | regions {nreg} (full {nfull})" if regs else ""),
                         fontsize=6.0, pad=2.0)
    else:
        v = row["gt_valid"]
        for i in range(row["n_points"]):
            if not v[i]:
                continue
            x, y = ellipse_xy(row["gt_center"][i], row["gt_ab"][i, 0],
                              row["gt_ab"][i, 1], row["gt_theta"][i])
            ax.plot(x, y, "-", lw=0.7, color="tab:green")
        ax.plot(row["gt_center"][:, 0], row["gt_center"][:, 1], "--", color="tab:purple",
                lw=0.8)
        if titles:
            ax.set_title(f"{idx:02d} GT ({int(v.sum())}/{row['n_points']} labelled)",
                         fontsize=6.5)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[3], extent[2])
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(not plain)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out-dir", default="results/figures/grid")
    ap.add_argument("--device", default="")
    ap.add_argument("--plain", action="store_true",
                    help="white background: only the trajectory and its ellipses")
    ap.add_argument("--no-titles", action="store_true", help="no text at all")
    ap.add_argument("--big", type=int, default=3,
                    help="how many large clean single-trajectory figures to write")
    ap.add_argument("--no-regions", action="store_true",
                    help="skip the convex safe regions (ellipses only)")
    ap.add_argument("--window-margin", type=float, default=3.0,
                    help="region cap beyond the ellipse semi-major axis [m]")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    model = PSENet(cfg).to(device).eval()
    model.load_state_dict(ck["model"])

    entries = load_index(cfg["data"]["sdf_dir"])
    tr, va, te = split_by_group(entries, frac=tuple(cfg["data"]["split_frac"]),
                                seed=int(cfg["data"]["split_seed"]))
    pick = {"train": tr, "val": va, "test": te}[args.split]
    pick = [i for i in pick if int(entries[i]["n_points"]) <= int(cfg["data"]["max_points"])]
    rng = np.random.default_rng(args.seed)
    pick = [pick[i] for i in sorted(rng.choice(len(pick), size=min(args.n, len(pick)),
                                               replace=False))]
    loader = build_loader([entries[i] for i in pick], cfg["data"]["trajectory_dir"],
                          cfg["data"]["sdf_dir"], args.batch_size, cfg,
                          augment=False, shuffle=False)
    rows = collect(model, loader, device, regions=not args.no_regions,
                   window_margin=args.window_margin)
    os.makedirs(args.out_dir, exist_ok=True)

    n = len(rows)
    cols = max(1, int(args.cols))
    rws = int(np.ceil(n / cols))
    out_sets = [("pred", "grid_predictions.png", False)]
    if args.plain:
        out_sets.append(("pred", "grid_predictions_plain.png", True))
    if not args.no_titles:
        out_sets.append(("gt", "grid_gt.png", False))
    for mode, name, plain in out_sets:
        fig, axes = plt.subplots(rws, cols, figsize=(3.0 * cols, 3.0 * rws))
        axes = np.atleast_2d(axes)
        for k in range(rws * cols):
            ax = axes[k // cols, k % cols]
            if k < n:
                draw_panel(ax, rows[k], mode, k, plain=plain,
                           titles=not args.no_titles)
            else:
                ax.axis("off")
        if not args.no_titles:
            extra = (f" | regions: window +{args.window_margin:g} m"
                     if (mode == "pred" and not args.no_regions) else "")
            fig.suptitle(f"{os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint)))}"
                         f" | split {args.split} | {n} trajectories | "
                         f"{'prediction + convex safe regions' if mode == 'pred' else 'GT reference (diagnostic only)'}"
                         + extra, fontsize=11, y=0.995)
        fig.subplots_adjust(left=0.005, right=0.995, top=0.955, bottom=0.005,
                            wspace=0.03, hspace=0.16)
        fig.savefig(os.path.join(args.out_dir, name), dpi=130)
        plt.close(fig)

    # large, clean single-trajectory figures (no text, no metrics)
    for k in range(min(args.big, n)):
        fig, ax = plt.subplots(figsize=(7.2, 7.2))
        draw_panel(ax, rows[k], "pred", k, plain=args.plain, titles=False)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        fig.savefig(os.path.join(args.out_dir, f"single_{k:02d}.png"), dpi=150)
        plt.close(fig)

    csv_path = os.path.join(args.out_dir, "per_trajectory.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["idx", "scene_id", "file", "n_points", "resolution_m",
                     "exact_safe_rate", "collision_rate", "midpoint_continuity",
                     "boundary_cover", "corridor_success", "border_violation",
                     "area_p50_m2", "p_max_mean", "regions", "regions_ok",
                     "regions_full"])
        for k, r in enumerate(rows):
            wr.writerow([k, r["scene_id"], r["file"], r["n_points"], f"{r['resolution_m']:.4f}",
                         f"{np.mean(r['safe']):.4f}", f"{1 - np.mean(r['safe']):.4f}",
                         f"{np.mean(r['pair_ok']) if len(r['pair_ok']) else float('nan'):.4f}",
                         f"{np.mean(r['pair_cover']) if len(r['pair_cover']) else float('nan'):.4f}",
                         int(r["corridor_ok"]), f"{np.mean(r['border']):.4f}",
                         f"{np.median(np.pi * r['a'] * r['b']):.1f}",
                         f"{np.mean(r['prob']):.3f}",
                         len(r["regions"]) if r["regions"] else 0,
                         int(np.sum(r["regions_ok"])) if r["regions"] else 0,
                         int(np.sum(r["regions_full"])) if r["regions"] else 0])

    # summary figure: per-trajectory distributions
    safe = np.array([np.mean(r["safe"]) for r in rows])
    cover = np.array([np.mean(r["pair_cover"]) if len(r["pair_cover"]) else np.nan
                      for r in rows])
    cont = np.array([np.mean(r["pair_ok"]) if len(r["pair_ok"]) else np.nan for r in rows])
    area = np.array([np.median(np.pi * r["a"] * r["b"]) for r in rows])
    npts = np.array([r["n_points"] for r in rows])
    corridor = np.array([r["corridor_ok"] for r in rows], dtype=float)

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    axes[0].hist(safe, bins=np.linspace(0, 1, 21), color="tab:green")
    axes[0].axvline(safe.mean(), color="k", ls="--", lw=1)
    axes[0].set_title(f"per-trajectory exact safe rate\nmean {safe.mean():.3f} "
                      f"| points {np.mean(safe):.3f}", fontsize=8)
    axes[0].set_xlabel("fraction of safe ellipses")
    axes[1].hist(cover, bins=20, color="tab:blue")
    axes[1].axvline(np.nanmean(cover), color="k", ls="--", lw=1)
    axes[1].set_title(f"boundary cover rate\nmean {np.nanmean(cover):.3f}", fontsize=8)
    axes[2].hist(cont, bins=20, color="tab:orange")
    axes[2].axvline(np.nanmean(cont), color="k", ls="--", lw=1)
    axes[2].set_title(f"midpoint continuity\nmean {np.nanmean(cont):.3f}", fontsize=8)
    axes[3].scatter(npts, safe, s=18, color="tab:red")
    axes[3].set_title(f"safe vs N  |  corridor success {corridor.mean():.3f}",
                      fontsize=8)
    axes[3].set_xlabel("path points N"); axes[3].set_ylabel("safe rate")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(f"multi-trajectory summary ({n} trajectories, split {args.split})",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "summary_metrics.png"), dpi=120)
    plt.close(fig)

    summary = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "split": args.split, "n_trajectories": n,
        "n_points": int(npts.sum()),
        "exact_safe_rate_per_point": float(np.mean(np.concatenate([r["safe"] for r in rows]))),
        "exact_safe_rate_per_trajectory": float(safe.mean()),
        "trajectories_fully_safe": float(np.mean(safe == 1.0)),
        "corridor_success_rate": float(corridor.mean()),
        "midpoint_continuity": float(np.nanmean(cont)),
        "boundary_cover": float(np.nanmean(cover)),
        "area_p50_m2": float(np.median(area)),
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"-> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
