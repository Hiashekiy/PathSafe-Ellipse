"""Stress test: guidance trajectories that hug a wall for a long distance.

Part A - mine the existing test split for "long wall-hugging runs" (consecutive
         path points with small clearance) and compare network metrics per
         clearance stratum against the overall test numbers.
Part B - purpose-build wall-hugging A* paths on *unseen* test maps (a large
         per-cell cost on high-clearance cells forces the search to hug
         obstacles), run them through the real labelling pipeline (this also
         checks that the IRIS labels are producible in that regime), and
         evaluate + visualise the network on them.
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
from pathsafe_ellipse.dataset_builder import build_samples_for_map, get_solver  # noqa: E402
from pathsafe_ellipse.ellipse_validation import ellipse_obstacle_intersections  # noqa: E402
from pathsafe_ellipse.evaluation import ellipse_iou_grid, rebuild_L_cells  # noqa: E402
from pathsafe_ellipse.map_io import build_map_catalog, map_key  # noqa: E402
from pathsafe_ellipse.map_preprocess import preprocess_map  # noqa: E402
from pathsafe_ellipse.model import EllipseNet  # noqa: E402
from pathsafe_ellipse.path_search import astar  # noqa: E402
from pathsafe_ellipse.path_sampling import sample_path, path_tangents  # noqa: E402
from pathsafe_ellipse.torch_dataset import SplitData  # noqa: E402

TIGHT = 3.0          # cells: "hugging the wall" clearance threshold
RUN_MIN_SAMPLES = 8  # consecutive tight samples that count as a long run


# --------------------------------------------------------------------------- #
def project_safe(L, center_cells, patch, iters=12):
    """Largest uniform shrink (about the patch centre) that removes all collisions.

    Scaling both the centre offset and the shape matrix about the path point
    keeps ``rho = ||L^-1 (p - c)||`` invariant, so "path point inside the
    ellipse" is preserved by construction; the shrink only trades area for
    safety.  Returns the scale in (0, 1].
    """
    c0 = np.array([64.0, 64.0])
    if not ellipse_obstacle_intersections(L, center_cells, patch)[0]:
        return 1.0
    lo, hi = 0.0, 1.0                     # lo safe (degenerate), hi unsafe
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if ellipse_obstacle_intersections(L * mid, c0 + mid * (center_cells - c0), patch)[0]:
            hi = mid
        else:
            lo = mid
    return lo


def model_metrics(model, device, res, patches, rays, hits, tangent, gt_center, gt_ab,
                  gt_sincos, eta, safety_max=None, projection=True):
    """Metrics for a batch of samples given as raw arrays."""
    gt_ab = np.sort(gt_ab, axis=1)[:, ::-1]
    gt_angle = np.arctan2(gt_sincos[:, 0], gt_sincos[:, 1])
    n = len(patches)
    order = np.arange(n)
    if safety_max is not None and n > safety_max:
        order = np.random.default_rng(0).permutation(n)[:safety_max]
    preds = {"center": [], "ab": [], "angle": []}
    with torch.no_grad():
        for s in range(0, n, 256):
            sl = slice(s, min(s + 256, n))
            t = {
                "patch": torch.from_numpy(patches[sl][:, None].astype(np.float32)).to(device),
                "rays": torch.from_numpy(rays[sl].astype(np.float32)).to(device),
                "hits": torch.from_numpy(hits[sl].astype(np.float32)).to(device),
                "tangent": torch.from_numpy(tangent[sl].astype(np.float32)).to(device),
            }
            c, ab, ang, _, _ = model.predict_ellipse(t)
            preds["center"].append(c.cpu().numpy())
            preds["ab"].append(ab.cpu().numpy())
            preds["angle"].append(ang.cpu().numpy())
    center = np.concatenate(preds["center"])
    ab = np.sort(np.concatenate(preds["ab"]), axis=1)[:, ::-1]
    angle = np.concatenate(preds["angle"])

    iou = ellipse_iou_grid((center, ab, angle), (gt_center, gt_ab, gt_angle), res)
    ang_err = np.degrees(np.minimum(np.mod(np.abs(angle - gt_angle), np.pi),
                                    np.pi - np.mod(np.abs(angle - gt_angle), np.pi)))
    L_pred = rebuild_L_cells(ab[:, 0], ab[:, 1], angle, res)
    centers = np.stack([64.0 + center[:, 0] / res, 64.0 + center[:, 1] / res], axis=1)
    rho = np.array([float(np.linalg.norm(np.linalg.inv(L_pred[i]) @ (np.array([64.0, 64.0]) - centers[i])))
                    for i in range(n)])
    safe = inside = both = 0
    collides_all = np.zeros(n, dtype=bool)
    for i in order:
        collides_all[i] = ellipse_obstacle_intersections(L_pred[i], centers[i], patches[i])[0]
        ok_in = rho[i] <= eta
        safe += int(not collides_all[i])
        inside += int(ok_in)
        both += int((not collides_all[i]) and ok_in)
    n_safety = int(len(order))
    out = {
        "n": int(n),
        "center_err_m": float(np.abs(center - gt_center).mean()),
        "a_err_m": float(np.abs(ab[:, 0] - gt_ab[:, 0]).mean()),
        "b_err_m": float(np.abs(ab[:, 1] - gt_ab[:, 1]).mean()),
        "angle_err_deg": float(ang_err.mean()),
        "iou": float(iou.mean()),
        "iou_p10": float(np.percentile(iou, 10)),
        "rho_ok_rate": float(np.mean(rho <= eta)),
        "safety_rate": safe / max(n_safety, 1),
        "safe_and_inside_rate": both / max(n_safety, 1),
        "n_safety_checked": n_safety,
    }
    if projection:
        scales = np.ones(n)
        for i in order:
            scales[i] = project_safe(L_pred[i], centers[i], patches[i])
        sel = order
        proj_ab = ab[sel] * scales[sel][:, None]
        proj_center = center[sel] * scales[sel][:, None]
        iou_proj = ellipse_iou_grid((proj_center, proj_ab, angle[sel]),
                                    (gt_center[sel], gt_ab[sel], gt_angle[sel]), res)
        out["projection_needed_rate"] = float(np.mean(scales[sel] < 0.999))
        out["projection_mean_scale"] = float(np.mean(scales[sel]))
        out["projection_mean_scale_when_needed"] = float(
            np.mean(scales[sel][scales[sel] < 0.999])) if np.any(scales[sel] < 0.999) else 1.0
        out["iou_after_projection"] = float(iou_proj.mean())
    return out


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = EllipseNet(width=int(ckpt.get("width", 32))).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def long_hugging_mask(point_ids, clearances, threshold=TIGHT, min_run=RUN_MIN_SAMPLES):
    """Samples that belong to a run of >= min_run consecutive tight points."""
    order = np.argsort(point_ids)
    tight = clearances[order] <= threshold
    mask = np.zeros(len(point_ids), dtype=bool)
    i = 0
    while i < len(tight):
        if not tight[i]:
            i += 1
            continue
        j = i
        while j < len(tight) and tight[j]:
            j += 1
        if (j - i) >= min_run:
            mask[order[i:j]] = True
        i = j
    return mask


# --------------------------------------------------------------------------- #
def part_a(model, device, data, res, eta, cfg, out_dir):
    keys = np.stack([data.arrays["map_id"], data.arrays["path_id"]], axis=1)
    clr = data.arrays["clearance"].astype(float)
    pid = data.arrays["point_id"].astype(int)

    hug = np.zeros(len(clr), dtype=bool)
    runs = 0
    for k in np.unique(keys, axis=0):
        sel = np.flatnonzero((keys[:, 0] == k[0]) & (keys[:, 1] == k[1]))
        m = long_hugging_mask(pid[sel], clr[sel])
        if m.any():
            runs += 1
        hug[sel[m]] = True

    bands = {
        "compact <=3 cells (贴墙)": clr <= 3.0,
        "3-5 cells": (clr > 3.0) & (clr <= 5.0),
        "5-8 cells": (clr > 5.0) & (clr <= 8.0),
        "open >8 cells": clr > 8.0,
        "long hugging runs only": hug,
        "ALL test samples": np.ones(len(clr), dtype=bool),
    }
    rows = []
    if "map_patch_packed" in data.arrays:
        packed = data.arrays["map_patch_packed"]
    else:
        packed = data.packed
    for name, sel in bands.items():
        idx = np.flatnonzero(sel)
        if len(idx) == 0:
            continue
        idx = idx[: min(len(idx), 1200)]
        patches = np.unpackbits(packed[idx], axis=2).reshape(len(idx), 128, 128)
        m = model_metrics(
            model, device, res, patches,
            data.arrays["ray_distances"][idx], data.arrays["ray_hits"][idx],
            data.arrays["path_tangent"][idx], data.arrays["ellipse_center_offset"][idx],
            data.arrays["ellipse_ab"][idx], data.arrays["ellipse_angle"][idx], eta,
            safety_max=400)
        m["subset"] = name
        m["subset_total"] = int(sel.sum())
        rows.append(m)
        print(f"  {name:28s} n={m['subset_total']:6d} (eval {m['n']:4d}) "
              f"IoU={m['iou']:.3f} centre={m['center_err_m']:.2f}m "
              f"axes=({m['a_err_m']:.2f},{m['b_err_m']:.2f})m ang={m['angle_err_deg']:.1f}° "
              f"rho_ok={m['rho_ok_rate']:.3f} safe={m['safety_rate']:.3f} "
              f"safe+in={m['safe_and_inside_rate']:.3f}"
              + (f" | +投影: shrink_needed={m['projection_needed_rate']:.2f} "
                 f"scale={m['projection_mean_scale']:.3f} IoU={m['iou_after_projection']:.3f} "
                 f"safe=1.000" if "projection_needed_rate" in m else ""))
    print(f"  (paths containing a long hugging run: {runs} / {len(np.unique(keys, axis=0))})")
    with open(os.path.join(out_dir, "wall_hugging_strata.json"), "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "tight_cells": TIGHT, "run_min_samples": RUN_MIN_SAMPLES,
                   "paths_with_long_runs": runs,
                   "paths_total": int(len(np.unique(keys, axis=0)))}, fh, indent=2)
    return rows


# --------------------------------------------------------------------------- #
def part_b(model, device, cfg, pp, res, eta, records, out_dir, n_maps=3, n_paths=4):
    """Purpose-built wall-hugging paths -> label generation -> network metrics."""
    occ_clear = None
    generated = []          # (key, path array, samples dict)
    label_stats = {"points": 0, "samples": 0, "reasons": {}}
    solver = get_solver(cfg)

    for rec in records[:n_maps]:
        key = map_key(rec)
        inflated = preprocess_map(rec, pp.map_dir, pp.inflated_maps_dir,
                                  robot_radius=float(cfg["robot"]["radius"]),
                                  kernel_radius=int(cfg["robot"]["inflation_kernel"]),
                                  iterations=int(cfg["robot"]["inflation_iterations"])
                                  if "inflated_iterations" in cfg["robot"] else int(cfg["robot"]["inflation_iterations"]))
        occ = inflated.occupancy_inflated
        clr = inflated.clearance
        # Wall-hugging trajectories are physically produced by *narrow corridors*.
        # A pure cost bias cannot create them on open street grids (detouring to
        # hug a wall never pays off), so detect genuinely elongated tight regions
        # (clearance <= TIGHT), take their two extremes as endpoints, and then
        # forbid leaving the corridor.
        from scipy import ndimage as ndi

        narrow = (occ == 0) & (clr <= TIGHT)
        labels, n_comp = ndi.label(narrow, structure=np.ones((3, 3), dtype=bool))
        corridors = []
        for cid in range(1, n_comp + 1):
            pts = np.argwhere(labels == cid)      # (row, col)
            if len(pts) < 200:
                continue
            xy = np.column_stack([pts[:, 1], pts[:, 0]]).astype(float)
            centred = xy - xy.mean(axis=0)
            cov = np.cov(centred.T)
            vals, vecs = np.linalg.eigh(cov)
            major = vecs[:, int(np.argmax(vals))]
            proj = centred @ major
            extent_major = float(proj.max() - proj.min())
            extent_minor = 2.0 * np.sqrt(max(vals.min(), 1e-9))
            if extent_major < 60.0 or extent_major < 2.5 * extent_minor:
                continue
            corridors.append((extent_major, xy[int(np.argmin(proj))], xy[int(np.argmax(proj))]))
        corridors.sort(key=lambda c: -c[0])

        exit_cost = np.where(clr > TIGHT, 8.0, 0.0)
        paths = []
        for extent, start_f, goal_f in corridors[:12]:
            start = (int(round(start_f[0])), int(round(start_f[1])))
            goal = (int(round(goal_f[0])), int(round(goal_f[1])))
            if occ[start[1], start[0]] or occ[goal[1], goal[0]]:
                continue
            path = astar(occ, start, goal, extra_cost=exit_cost)
            if path is None or len(path) < 60:
                continue
            arr = np.asarray(path)
            tight = float(np.mean(clr[arr[:, 1], arr[:, 0]] <= TIGHT))
            if tight < 0.5:
                continue
            paths.append((tight, arr, start, goal))
            if len(paths) >= n_paths:
                break
        if corridors:
            print(f"  {key}: {len(corridors)} tight corridors detected, "
                  f"longest extent {corridors[0][0]:.0f} cells")
        if not paths:
            print(f"  {key}: no wall-hugging path found")
            continue
        arrays, info = build_samples_for_map(
            inflated, inflated.occupancy_raw,
            {"paths": [c[1] for c in paths]}, cfg, solver, seed=7)
        label_stats["points"] += int(info["candidate_stats"].get("points", 0))
        label_stats["samples"] += int(len(arrays["label_valid"]))
        for k, v in info["reasons"].items():
            label_stats["reasons"][k] = label_stats["reasons"].get(k, 0) + int(v)
        tight_pcts = [round(c[0], 3) for c in paths]
        print(f"  {key}: {len(paths)} hugging paths (tight fraction {tight_pcts}), "
              f"{len(arrays['label_valid'])} labelled points "
              f"(of {info['candidate_stats'].get('points', 0)})")
        generated.append((key, inflated, paths, arrays))

    if not generated:
        print("  no wall-hugging samples generated")
        return label_stats, []

    # ----- network metrics on these freshly generated wall-hugging samples
    all_patches, all_rays, all_hits, all_tan = [], [], [], []
    all_c, all_ab, all_sc, all_safe_gt = [], [], [], []
    for key, inflated, paths, arrays in generated:
        n = len(arrays["label_valid"])
        all_patches.append(arrays["map_patch"].reshape(n, 128, 128))
        all_rays.append(arrays["ray_distances"])
        all_hits.append(arrays["ray_hits"])
        all_tan.append(arrays["path_tangent"])
        all_c.append(arrays["ellipse_center_offset"])
        all_ab.append(arrays["ellipse_ab"])
        all_sc.append(arrays["ellipse_angle"])
    patches = np.concatenate(all_patches)
    metrics = model_metrics(model, device, res, patches,
                            np.concatenate(all_rays), np.concatenate(all_hits),
                            np.concatenate(all_tan), np.concatenate(all_c),
                            np.concatenate(all_ab), np.concatenate(all_sc), eta,
                            safety_max=None)
    print(f"  network on wall-hugging samples: IoU={metrics['iou']:.3f} "
          f"centre={metrics['center_err_m']:.2f}m axes=({metrics['a_err_m']:.2f},"
          f"{metrics['b_err_m']:.2f})m ang={metrics['angle_err_deg']:.1f}° "
          f"rho_ok={metrics['rho_ok_rate']:.3f} safe={metrics['safety_rate']:.3f} "
          f"safe+in={metrics['safe_and_inside_rate']:.3f}")
    if "projection_needed_rate" in metrics:
        print(f"  after safety projection: shrink needed on "
              f"{metrics['projection_needed_rate']*100:.1f}% of samples, "
              f"mean scale {metrics['projection_mean_scale']:.3f} "
              f"(when needed {metrics['projection_mean_scale_when_needed']:.3f}), "
              f"IoU {metrics['iou']:.3f} -> {metrics['iou_after_projection']:.3f}, "
              f"collision-free 100% by construction")
    with open(os.path.join(out_dir, "wall_hugging_generated.json"), "w", encoding="utf-8") as fh:
        json.dump({"label_stats": label_stats, "network": metrics,
                   "maps": [g[0] for g in generated]}, fh, indent=2)

    # ----- figures: map + hugging path + GT vs predicted ellipses
    for key, inflated, paths, arrays in generated:
        n = len(arrays["label_valid"])
        pch = arrays["map_patch"].reshape(n, 128, 128)
        with torch.no_grad():
            t = {
                "patch": torch.from_numpy(pch[:, None].astype(np.float32)).to(device),
                "rays": torch.from_numpy(arrays["ray_distances"].astype(np.float32)).to(device),
                "hits": torch.from_numpy(arrays["ray_hits"].astype(np.float32)).to(device),
                "tangent": torch.from_numpy(arrays["path_tangent"].astype(np.float32)).to(device),
            }
            center, ab, ang, _, _ = model.predict_ellipse(t)
        center = center.cpu().numpy(); ab = np.sort(ab.cpu().numpy(), axis=1)[:, ::-1]
        ang = ang.cpu().numpy()
        gt_c = arrays["ellipse_center_offset"]
        gt_ab = np.sort(arrays["ellipse_ab"], axis=1)[:, ::-1]
        gt_ang = np.arctan2(arrays["ellipse_angle"][:, 0], arrays["ellipse_angle"][:, 1])
        pts = arrays["point_xy"].astype(float)
        ious = ellipse_iou_grid((center, ab, ang), (gt_c, gt_ab, gt_ang), res)
        L_pred = rebuild_L_cells(ab[:, 0], ab[:, 1], ang, res)
        cen = np.stack([64.0 + center[:, 0] / res, 64.0 + center[:, 1] / res], axis=1)
        safe = sum(int(not ellipse_obstacle_intersections(L_pred[i], cen[i], pch[i])[0])
                   for i in range(n))

        for pi, (tight, path, start, goal) in enumerate(paths):
            sel = np.flatnonzero(arrays["path_id"] == pi)
            if len(sel) < 5:
                continue
            every = max(1, int(np.ceil(len(sel) / 18)))
            fig, ax = plt.subplots(figsize=(9.5, 9.0))
            ax.imshow(inflated.occupancy_inflated, cmap="gray_r", interpolation="nearest")
            ax.plot(path[:, 0], path[:, 1], "-", lw=2.6, color="tab:purple",
                    label="wall-hugging A* path", zorder=8, solid_capstyle="round")
            ax.plot(path[0, 0], path[0, 1], "o", ms=9, color="lime", mec="k", label="start", zorder=9)
            ax.plot(path[-1, 0], path[-1, 1], "X", ms=10, color="red", mec="k", label="goal", zorder=9)
            for bi in sel[::every]:
                c_gt = pts[bi] + gt_c[bi] / res
                ax.add_patch(Ellipse((c_gt[0], c_gt[1]), 2 * gt_ab[bi, 0] / res,
                                     2 * gt_ab[bi, 1] / res, angle=np.degrees(gt_ang[bi]),
                                     fill=False, ec="tab:green", lw=1.1, ls="--", zorder=2))
                c_pr = pts[bi] + center[bi] / res
                ax.add_patch(Ellipse((c_pr[0], c_pr[1]), 2 * ab[bi, 0] / res,
                                     2 * ab[bi, 1] / res, angle=np.degrees(ang[bi]),
                                     fill=False, ec="tab:blue", lw=1.3, zorder=5))
                ax.plot(pts[bi, 0], pts[bi, 1], ".", ms=2.2, color="tab:red", zorder=7)
            ax.plot([], [], color="tab:green", ls="--", label="ground-truth ellipse")
            ax.plot([], [], color="tab:blue", label="predicted ellipse")
            ax.plot([], [], ".", color="tab:red", label="path points (labels)")
            m = 55
            ax.set_xlim(max(0, path[:, 0].min() - m), min(inflated.occupancy_inflated.shape[1], path[:, 0].max() + m))
            ax.set_ylim(min(inflated.occupancy_inflated.shape[0], path[:, 1].max() + m), max(0, path[:, 1].min() - m))
            ax.set_title(f"WALL-HUGGING  {key}  path {pi}  "
                         f"(tight fraction {tight:.0%}, {len(sel)} labelled points)\n"
                         f"path mean IoU={ious[sel].mean():.3f}   "
                         f"collision-free on this path="
                         f"{sum(int(not ellipse_obstacle_intersections(L_pred[i], cen[i], pch[i])[0]) for i in sel)}"
                         f"/{len(sel)}", fontsize=10)
            ax.legend(loc="upper right", fontsize=8)
            fig.tight_layout()
            out = os.path.join(out_dir, f"wallhug_{key}_path{pi}.png")
            fig.savefig(out, dpi=120)
            plt.close(fig)
            print(f"  wrote {out}")
    return label_stats, metrics


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-maps", type=int, default=3)
    ap.add_argument("--n-paths", type=int, default=4)
    ap.add_argument("--part", default="both", choices=["a", "b", "both"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg, "wall_hugging")
    pp = project_paths(cfg)
    res = float(cfg["map"]["resolution"])
    eta = float(cfg["label"]["point_inside_eta"])
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    out_dir = os.path.join(pp.reports_dir, "wall_hugging")
    os.makedirs(out_dir, exist_ok=True)

    model, ckpt = load_model(args.ckpt or os.path.join(pp.outputs_dir, "checkpoints",
                                                      "ellipse_net_best.pt"), device)
    print(f"model: epoch {ckpt.get('epoch')}")

    print("\n== Part A: clearance strata of the existing test split ==")
    data = SplitData(os.path.join(pp.splits_dir, f"{args.split}.npz"))
    if args.part in ("a", "both"):
        part_a(model, device, data, res, eta, cfg, out_dir)
    if args.part == "a":
        return 0

    print("\n== Part B: purpose-built wall-hugging paths on unseen maps ==")
    with open(os.path.join(pp.labels_dir, "map_index.json"), encoding="utf-8") as fh:
        map_index = {int(k): v for k, v in json.load(fh).items()}
    with open(os.path.join(pp.splits_dir, "map_group_split.json"), encoding="utf-8") as fh:
        split = json.load(fh)["split"]
    from pathsafe_ellipse.map_io import map_group_id
    catalog = {map_key(r): r for r in build_map_catalog(pp.map_dir, cfg["map"]["resolution"]) if r.parsed_ok}
    test_groups = set(split[args.split])
    test_maps = [k for k in map_index.values() if map_group_id(k) in test_groups]
    obstacles = {}
    with open(os.path.join(pp.reports_dir, "map_inventory.csv"), encoding="utf-8") as fh:
        import csv
        for row in csv.DictReader(fh):
            obstacles[os.path.splitext(row["file_name"])[0]] = float(row["obstacle_ratio"])
    records = sorted((catalog[k] for k in test_maps),
                     key=lambda r: -obstacles.get(map_key(r), 0.0))[: args.n_maps]
    print(f"  maps: {[map_key(r) for r in records]}")
    part_b(model, device, cfg, pp, res, eta, records, out_dir, n_maps=args.n_maps,
           n_paths=args.n_paths)
    print(f"\nwrote results -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
