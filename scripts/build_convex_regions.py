"""Predict ellipses -> inflate them into convex safe regions (polytopes).

    python scripts/build_convex_regions.py --split test --n 8

For every trajectory: the network predicts one ellipse per path point, then all ellipses
of that trajectory are inflated **in a single batched call** (the per-ellipse python loop
is only used for the serial reference timing).  Each region is verified exactly (no
blocked square overlaps it and the ellipse stays inside), the timings are reported and a
figure per trajectory is written to ``results/figures/polytopes``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from ellipsenet.data.dataset import TrajectoryDataset, collate, load_index, split_by_group  # noqa: E402
from ellipsenet.geometry.exact import exact_clearance  # noqa: E402
from ellipsenet.geometry.polytopes import (build_polytopes,  # noqa: E402
                                           build_polytopes_serial, ellipse_matrix,
                                           ellipse_support, polytope_area, verify_polytopes)
from ellipsenet.models.psenet import PSENet  # noqa: E402


def predict_ellipses(model, item, device):
    batch = {k: (v.to(device) if torch.is_tensor(v) else v)
             for k, v in collate([item]).items()}
    with torch.no_grad():
        out = model(batch, tau=1.0)
        dec = model.decode_argmax(out)
    n = int(item["n_points"])
    return (dec["center"][0, :n].cpu(), dec["a"][0, :n].cpu(), dec["b"][0, :n].cpu(),
            dec["theta"][0, :n].cpu(), dec["prob"][0, :n].cpu())


def region_polygon(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vertices of ``{A x <= b}`` (convex hull of the feasible plane intersections)."""
    from scipy.spatial import ConvexHull

    pts = []
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            M = np.stack([A[i], A[j]])
            if abs(np.linalg.det(M)) < 1e-9:
                continue
            p = np.linalg.solve(M, np.array([b[i], b[j]]))
            if np.all(A @ p <= b + 1e-6):
                pts.append(p)
    if len(pts) < 3:
        return np.empty((0, 2))
    pts = np.asarray(pts)
    try:
        hull = ConvexHull(pts)
        return pts[hull.vertices]
    except Exception:            # degenerate (collinear) vertex set: nothing to fill
        return np.empty((0, 2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/v3_best.pt")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--region-margin", type=float, default=0.02)
    ap.add_argument("--window-margin", type=float, default=1.5,
                    help="region cap beyond the ellipse semi-major axis [m]; larger = bigger region, slower")
    ap.add_argument("--serial-check", type=int, default=2,
                    help="how many trajectories to also time with the per-ellipse loop")
    ap.add_argument("--out-dir", default="results/figures/polytopes")
    ap.add_argument("--out-json", default="results/metrics/polytope_bench.json")
    ap.add_argument("--device", default="")
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
    _tr, _va, te = split_by_group(entries, frac=tuple(cfg["data"]["split_frac"]),
                                  seed=int(cfg["data"]["split_seed"]))
    pool = {"train": _tr, "val": _va, "test": te}[args.split]
    pool = [i for i in pool if int(entries[i]["n_points"]) <= int(cfg["data"]["max_points"])]
    rng = np.random.default_rng(args.seed)
    pick = [entries[pool[i]] for i in sorted(rng.choice(len(pool), size=min(args.n, len(pool)),
                                                       replace=False))]
    ds = TrajectoryDataset(pick, cfg["data"]["trajectory_dir"], cfg["data"]["sdf_dir"],
                           augment=False)
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    tot_pred = tot_region = 0.0
    tot_ellipses = 0
    serial_ms = []
    for j in range(len(ds)):
        item = ds[j]
        res = float(item["r_scene"])
        blocked = item["occupancy"][0]
        t0 = time.perf_counter()
        cen, a, bb, th, prob = predict_ellipses(model, item, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_pred = (time.perf_counter() - t0) * 1e3

        dev_blocked = torch.as_tensor(np.asarray(blocked), dtype=torch.float32, device=device)
        t0 = time.perf_counter()
        polys = build_polytopes(cen.to(device), a.to(device), bb.to(device), th.to(device),
                               dev_blocked, res, region_margin_m=args.region_margin,
                               window_margin_m=args.window_margin)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_region = (time.perf_counter() - t0) * 1e3

        rep = verify_polytopes(polys, dev_blocked, res)
        # split the violations: only ellipses that are themselves obstacle free can be
        # expected to sit inside an obstacle-free region
        viol_feasible_safe = 0
        areas = [polytope_area(*polys.planes(i)) for i in range(len(cen))]
        # an ellipse that itself collides cannot be contained in an obstacle-free region
        occ_np = np.asarray(blocked).astype(np.uint8)
        safe_mask = []
        for i in range(len(cen)):
            R = np.array([[np.cos(float(th[i])), -np.sin(float(th[i]))],
                          [np.sin(float(th[i])), np.cos(float(th[i]))]])
            S = R @ np.diag([float(a[i]) ** 2, float(bb[i]) ** 2]) @ R.T
            ok, _mq, _b, _i = exact_clearance(S, cen[i].numpy(), occ_np, res)
            safe_mask.append(bool(ok))
        # an ellipse that leaves the map also cannot be contained in an in-map region
        Hm, Wm = occ_np.shape
        margin = 0.0
        in_map = ((cen[:, 0] - np.asarray(a) > margin)
                  & (cen[:, 0] + np.asarray(a) < (Wm - 1) * res - margin)
                  & (cen[:, 1] - np.asarray(a) > margin)
                  & (cen[:, 1] + np.asarray(a) < (Hm - 1) * res - margin)).tolist()
        ell_inside = True
        max_overrun = 0.0
        n_checked = 0
        for i in range(len(cen)):
            if not (safe_mask[i] and in_map[i]):
                continue
            n_checked += 1
            k = int(polys.n_planes[i])
            Q = ellipse_matrix(a[i:i + 1].to(device), bb[i:i + 1].to(device),
                               th[i:i + 1].to(device))[0]
            sup = ellipse_support(Q, polys.center[i], polys.A[i, :k])
            overrun = float((sup - polys.b[i, :k]).max())
            single = type(polys)(A=polys.A[i:i + 1, :k].cpu(), b=polys.b[i:i + 1, :k].cpu(),
                                n_planes=torch.tensor([k]), converged=torch.tensor([True]),
                                n_rounds=torch.tensor([k]), center=polys.center[i:i + 1].cpu(),
                                radius=polys.radius[i:i + 1].cpu(),
                                feasible=polys.feasible[i:i + 1].cpu())
            if verify_polytopes(single, dev_blocked.cpu(), res)["violating_regions"] > 0:
                viol_feasible_safe += 1
            max_overrun = max(max_overrun, overrun)
            ell_inside &= overrun <= 2.0 * args.region_margin

        if j < args.serial_check:
            t0 = time.perf_counter()
            build_polytopes_serial(cen.numpy(), a.numpy(), bb.numpy(), th.numpy(),
                                   np.asarray(blocked), res,
                                   region_margin_m=args.region_margin,
                                   window_margin_m=args.window_margin)
            serial_ms.append((time.perf_counter() - t0) * 1e3)

        tot_pred += t_pred
        tot_region += t_region
        tot_ellipses += len(cen)
        rows.append({
            "file": item["file"], "n_points": int(len(cen)),
            "pred_ms": round(t_pred, 2), "regions_ms": round(t_region, 2),
            "ms_per_ellipse": round(t_region / max(len(cen), 1), 3),
            "planes_mean": float(np.mean([int(k) for k in polys.n_planes])),
            "area_mean_m2": float(np.mean(areas)) if areas else 0.0,
            "area_min_m2": float(np.min(areas)) if areas else 0.0,
            "violating_regions": rep["violating_regions"],
            "violating_regions_feasible_safe": viol_feasible_safe,
            "ellipse_inside_when_safe": bool(ell_inside),
            "ellipse_overrun_max_m": round(max_overrun, 4),
            "n_checked_ellipses": int(n_checked),
            "n_skipped_ellipses": int(len(cen) - n_checked),
            "converged_all": bool(polys.converged.all()),
        })

        # ---------------------------------------------------------------- figure
        occ = np.asarray(blocked)
        res_ = res
        H, W = occ.shape
        extent = [0.0, (W - 1) * res_, (H - 1) * res_, 0.0]
        fig, ax = plt.subplots(figsize=(7.6, 7.6))
        ax.imshow(1.0 - occ, cmap="gray", extent=extent, vmin=0, vmax=1)
        path = np.asarray(item["path_xy"])
        ax.plot(path[:, 0], path[:, 1], "-", color="tab:blue", lw=1.2, label="A* path")
        t = np.linspace(0, 2 * np.pi, 72)
        for i in range(len(cen)):
            A, b = polys.planes(i)
            poly = region_polygon(A, b)
            if len(poly) >= 3:
                ax.fill(poly[:, 0], poly[:, 1], color="tab:orange", alpha=0.18, lw=0)
                ax.plot(np.r_[poly[:, 0], poly[0, 0]], np.r_[poly[:, 1], poly[0, 1]],
                        "-", color="tab:orange", lw=0.7)
            c, aa, bbb, tt = (float(cen[i, 0]), float(cen[i, 1])), float(a[i]), float(bb[i]), float(th[i])
            x = c[0] + aa * np.cos(t) * np.cos(tt) - bbb * np.sin(t) * np.sin(tt)
            y = c[1] + aa * np.cos(t) * np.sin(tt) + bbb * np.sin(t) * np.cos(tt)
            ax.plot(x, y, "-", color="tab:green", lw=0.8)
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[3], extent[2])
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{item['file'][:38]}\n{len(cen)} regions | "
                     f"{t_region:.1f} ms | {t_region / max(len(cen), 1):.2f} ms/ellipse",
                     fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"regions_{j:02d}.png"), dpi=130)
        plt.close(fig)

    summary = {
        "device": str(device), "split": args.split, "n_trajectories": len(rows),
        "n_ellipses": tot_ellipses,
        "predict_total_ms": round(tot_pred, 2),
        "regions_total_ms": round(tot_region, 2),
        "ms_per_trajectory_regions": round(tot_region / len(rows), 2),
        "ms_per_ellipse_batched": round(tot_region / max(tot_ellipses, 1), 3),
        "ms_per_trajectory_serial": (round(float(np.mean(serial_ms)), 2) if serial_ms else None),
        "violating_regions": int(sum(r["violating_regions"] for r in rows)),
        "violating_regions_feasible_safe": int(sum(r["violating_regions_feasible_safe"] for r in rows)),
        "ellipse_inside_all_safe": all(r["ellipse_inside_when_safe"] for r in rows),
        "ellipse_overrun_max_m": round(max(r["ellipse_overrun_max_m"] for r in rows), 4),
        "ellipse_inside_tolerance": round(2 * args.region_margin, 4),
        "n_checked_ellipses": int(sum(r["n_checked_ellipses"] for r in rows)),
        "n_skipped_ellipses": int(sum(r["n_skipped_ellipses"] for r in rows)),
        "skipped_reason": "ellipse itself collides or leaves the map (no in-map obstacle-free region can contain it)",
        "converged_all": all(r["converged_all"] for r in rows),
        "trajectories": rows,
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "trajectories"}, indent=2,
                     ensure_ascii=False))
    print(f"-> {args.out_dir}\n-> {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
