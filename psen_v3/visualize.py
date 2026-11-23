"""Visualise predictions of a ``psen_v3`` checkpoint (one figure per trajectory).

    python psen_v3/visualize.py --checkpoint runs_v3sdf/full_s0/best.pt --n 6

Each figure shows, on the same map: the A* path, the 16 candidate centres of a few
sampled path points (thin), the argmax-selected centre and ellipse (thick), and - for
reference only - the stored GT ellipses.  GT is never a training label; it is drawn so
the reader can see what the geometric objective converged to.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from psen_v3.data.dataset import (TrajectoryDataset, collate, load_index,  # noqa: E402
                                  split_by_group)
from psen_v3.evaluation.metrics import exact_trajectory  # noqa: E402
from psen_v3.models.psenet import PSENet  # noqa: E402


def ellipse_xy(c, a, b, theta, n=96):
    t = np.linspace(0.0, 2.0 * np.pi, n)
    x = c[0] + a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
    y = c[1] + a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)
    return x, y


def plot_candidates(item, out, prob, n, res, extent, occ, path,
                    center, a, b, theta, tag, out_dir):
    """Zoom on one path point: the 16 ray-constrained candidates and their weights.

    Left: the 16 rays with their LiDAR ranges, the 16 candidate centres
    ``q_ij = p_i + alpha_i * max(d_ij - m, 0) * u_ij`` and one ellipse per candidate
    (opacity proportional to the direction probability).  Right: the chosen (argmax)
    candidate alone, so the selected ellipse can be read without clutter.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    i = int(np.argmax(prob)) if prob.size else 0        # most confident point
    p = path[i]
    q = out["q"][0, i].cpu().numpy()                    # (16,2)
    pi = out["pi"][0, i].cpu().numpy()
    lidar = np.asarray(item["lidar"])[i]
    max_range = float(item["max_range_m"])
    jbest = int(np.argmax(pi))
    span = float(max(lidar.max(), 3.0)) + 1.5
    lo, hi = p - span, p + span

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4))
    for ax, mode in zip(axes, ("all", "argmax")):
        ax.imshow(1.0 - occ, cmap="gray", extent=extent, vmin=0, vmax=1)
        ax.plot(path[:, 0], path[:, 1], "-o", color="tab:blue", ms=3, lw=1.1,
                label="A* path", zorder=5)
        ax.plot([p[0]], [p[1]], "s", color="k", ms=6, zorder=6, label="query point i")
        if mode == "all":
            for k in range(len(pi)):
                ang = 2.0 * np.pi * k / 16.0
                u = np.array([np.cos(ang), np.sin(ang)])
                ax.plot([p[0], p[0] + lidar[k] * u[0]], [p[1], p[1] + lidar[k] * u[1]],
                        "-", color="tab:red", lw=0.6, alpha=0.35)
                ax.plot([q[k, 0]], [q[k, 1]], "o", color="tab:orange",
                        ms=2.0 + 9.0 * float(pi[k]), zorder=7)
                x, y = ellipse_xy(q[k], a[i], b[i], theta[i])
                ax.plot(x, y, "-", color="tab:orange", lw=0.8,
                        alpha=0.15 + 0.85 * float(pi[k]))
            ax.set_title(f"16 candidates | alpha={float(out['alpha'][0, i]):.2f} | "
                         f"a={a[i]:.2f} b={b[i]:.2f} m | "
                         f"theta={np.degrees(theta[i]):.0f} deg")
        else:
            ang = 2.0 * np.pi * jbest / 16.0
            u = np.array([np.cos(ang), np.sin(ang)])
            ax.plot([p[0], p[0] + lidar[jbest] * u[0]], [p[1], p[1] + lidar[jbest] * u[1]],
                    "-", color="tab:red", lw=1.2,
                    label=f"ray {jbest} (d={lidar[jbest]:.2f} m)")
            x, y = ellipse_xy(center[i], a[i], b[i], theta[i])
            ax.plot(x, y, "-", color="tab:green", lw=1.8, label="selected ellipse")
            ax.plot([center[i, 0]], [center[i, 1]], "o", color="tab:purple", ms=7,
                    label="selected centre q_ij")
            tag_hit = "hit" if lidar[jbest] < max_range - 1e-3 else "range-clipped"
            ax.set_title(f"argmax ray {jbest}: pi = {pi[jbest]:.3f} ({tag_hit})")
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(hi[1], lo[1])
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(alpha=0.2)
    fig.suptitle(f"{item['scene_id']} path {item['path_id']} | point {i}/{n} | "
                 f"res {res} m | zoom +-{span:.1f} m")
    fig.tight_layout()
    png = os.path.join(out_dir, f"candidates_{tag:02d}_{item['file'][:-4]}.png")
    fig.savefig(png, dpi=110)
    plt.close(fig)
    return png


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--seed", type=int, default=-1,
                    help=">=0: random sample of trajectories instead of the first N")
    ap.add_argument("--out-dir", default="outputs_v3sdf/viz")
    ap.add_argument("--device", default="")
    ap.add_argument("--candidate-figs", type=int, default=2,
                    help="emit a 16-candidate zoom figure for the first K trajectories")
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
    if args.seed >= 0:
        rng = np.random.default_rng(args.seed)
        pick = [pick[i] for i in sorted(rng.choice(len(pick), size=min(args.n, len(pick)),
                                                  replace=False))]
    else:
        pick = pick[args.offset: args.offset + args.n]

    ds = TrajectoryDataset([entries[i] for i in pick], cfg["data"]["trajectory_dir"],
                           cfg["data"]["sdf_dir"], augment=False)
    os.makedirs(args.out_dir, exist_ok=True)
    summary = []
    grid = plt.figure(figsize=(4.2 * len(pick), 4.2))
    for j, idx in enumerate(pick):
        item = ds[j]
        batch = {}
        for k, v in collate([item]).items():
            batch[k] = v.to(device) if torch.is_tensor(v) else v
        with torch.no_grad():
            out = model(batch, tau=1.0)
            dec = model.decode_argmax(out)
        res = float(item["r_scene"])
        H, W = item["occupancy"].shape[-2:]
        extent = [0.0, (W - 1) * res, (H - 1) * res, 0.0]
        n = int(item["n_points"])
        center = dec["center"][0, :n].cpu().numpy()
        a = dec["a"][0, :n].cpu().numpy()
        b = dec["b"][0, :n].cpu().numpy()
        theta = dec["theta"][0, :n].cpu().numpy()
        prob = dec["prob"][0, :n].cpu().numpy()
        qall = out["q"][0, :n].cpu().numpy()
        occ = np.asarray(item["occupancy"])[0]
        path = np.asarray(item["path_xy"])[:n]
        gt_c = np.asarray(item["diag_center"])[:n]
        gt_ab = np.asarray(item["diag_ab"])[:n]
        gt_th = np.asarray(item["diag_angle"])[:n]
        gt_valid = np.asarray(item["diag_valid"])[:n] > 0.5

        exact = exact_trajectory(center, a, b, theta, occ.astype(np.uint8), res)
        safe_rate = float(np.mean(exact["safe"]))
        cover = float(np.mean(exact["pair_cover"])) if len(exact["pair_cover"]) else float("nan")

        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        for ax, mode in zip(axes, ("prediction", "GT reference")):
            ax.imshow(1.0 - occ, cmap="gray", extent=extent, vmin=0, vmax=1)
            ax.plot(path[:, 0], path[:, 1], "-o", color="tab:blue", ms=2.5, lw=1.0,
                    label="A* path")
            if mode == "prediction":
                step = max(1, n // 8)
                for i in range(0, n, step):
                    ax.plot(qall[i, :, 0], qall[i, :, 1], ".", color="tab:orange", ms=2)
                for i in range(n):
                    x, y = ellipse_xy(center[i], a[i], b[i], theta[i])
                    col = "tab:green" if exact["safe"][i] else "tab:red"
                    ax.plot(x, y, "-", color=col, lw=1.0)
                ax.plot(center[:, 0], center[:, 1], "-", color="tab:purple", lw=1.0,
                        label="selected centres")
                ax.set_title(f"{mode}: safe {safe_rate:.2f} | cover {cover:.2f} | "
                             f"p_max {prob.mean():.2f}")
            else:
                for i in range(n):
                    if not gt_valid[i]:
                        continue
                    x, y = ellipse_xy(gt_c[i], gt_ab[i, 0], gt_ab[i, 1], gt_th[i])
                    ax.plot(x, y, "-", color="tab:green", lw=1.0)
                ax.plot(gt_c[:, 0], gt_c[:, 1], "-", color="tab:purple", lw=1.0,
                        label="GT centre line")
                ax.set_title(f"{mode}: {int(gt_valid.sum())}/{n} labelled")
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[3], extent[2])
            ax.set_aspect("equal")
            ax.legend(loc="upper right", fontsize=7)
        fig.suptitle(f"{item['scene_id']} | path id {item['path_id']} | N={n} | "
                     f"res {res} m | exact safe {safe_rate:.3f}")
        fig.tight_layout()
        png = os.path.join(args.out_dir, f"pred_{j:02d}_{item['file'][:-4]}.png")
        fig.savefig(png, dpi=110)
        plt.close(fig)
        summary.append({"png": os.path.basename(png), "scene_id": item["scene_id"],
                        "n_points": n, "exact_safe_rate": safe_rate,
                        "boundary_cover_rate": cover})

        axg = grid.add_subplot(1, len(pick), j + 1)
        axg.imshow(1.0 - occ, cmap="gray", extent=extent, vmin=0, vmax=1)
        for i in range(n):
            x, y = ellipse_xy(center[i], a[i], b[i], theta[i])
            axg.plot(x, y, "-", color="tab:green" if exact["safe"][i] else "tab:red",
                     lw=0.7)
        axg.plot(path[:, 0], path[:, 1], "-", color="tab:blue", lw=0.8)
        axg.set_title(f"{os.path.basename(png)[:22]}\nsafe {safe_rate:.2f}", fontsize=7)
        axg.set_aspect("equal")
        axg.set_xticks([]); axg.set_yticks([])

        if j < args.candidate_figs and n >= 2:
            cand_png = plot_candidates(
                item, out, prob, n, res, extent, occ, path,
                center, a, b, theta, j, args.out_dir)
            summary[-1]["candidates_png"] = os.path.basename(cand_png)
    grid.tight_layout()
    grid.savefig(os.path.join(args.out_dir, "overview.png"), dpi=120)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=float)
    print(json.dumps(summary, indent=1, default=float))
    print(f"-> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


