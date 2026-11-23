"""Reference value of the continuity metrics on the *GT* ellipses (read-only).

The reported ``boundary cover rate`` (mutual boundary containment) depends strongly
on how large the ellipses are relative to the path spacing, so the model has to be
compared against the labelled ellipses of the same dataset, not against 1.0.

    python scripts/measure_gt_continuity.py --n 200 --out results/metrics/gt_continuity_reference.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from ellipsenet.config import load_config  # noqa: E402
from ellipsenet.data.dataset import (TrajectoryDataset, load_index,  # noqa: E402
                                  split_by_group)
from ellipsenet.evaluation.metrics import exact_trajectory, quantiles  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--out", default="results/metrics/gt_continuity_reference.json")
    args = ap.parse_args()

    cfg = load_config()
    entries = load_index(cfg["data"]["sdf_dir"])
    tr, va, te = split_by_group(entries, frac=tuple(cfg["data"]["split_frac"]),
                                seed=int(cfg["data"]["split_seed"]))
    pool = {"train": tr, "val": va, "test": te}[args.split]
    pool = [i for i in pool if int(entries[i]["n_points"]) <= int(cfg["data"]["max_points"])]
    rng = np.random.default_rng(args.seed)
    pick = [pool[i] for i in sorted(rng.choice(len(pool), size=min(args.n, len(pool)),
                                               replace=False))]
    ds = TrajectoryDataset([entries[i] for i in pick], cfg["data"]["trajectory_dir"],
                           cfg["data"]["sdf_dir"], augment=False)

    covers, mids, corridors, areas, safe = [], [], [], [], []
    for j in range(len(ds)):
        s = ds[j]
        occ = np.asarray(s["occupancy"])[0].astype(np.uint8)
        res = float(s["r_scene"])
        center = np.asarray(s["diag_center"], np.float64)
        ab = np.asarray(s["diag_ab"], np.float64)
        th = np.asarray(s["diag_angle"], np.float64)
        valid = np.asarray(s["diag_valid"]).astype(bool)
        if valid.sum() < 2:
            continue
        r = exact_trajectory(center, ab[:, 0], ab[:, 1], th, occ, res, rho_max=0.85)
        # only judge the adjacencies where both endpoints are labelled
        keep = valid[:-1] & valid[1:]
        if keep.any():
            covers.append(r["pair_cover"][keep])
            mids.append(r["pair_ok"][keep])
        areas.append(np.pi * ab[valid, 0] * ab[valid, 1])
        safe.append(r["safe"][valid])
        corridors.append(bool(r["safe"][valid].all()
                              and (r["pair_ok"][keep].all() if keep.any() else True)))

    cover = np.concatenate(covers) if covers else np.zeros(0)
    mid = np.concatenate(mids) if mids else np.zeros(0)
    area = np.concatenate(areas) if areas else np.zeros(0)
    saf = np.concatenate(safe) if safe else np.zeros(0)
    out = {
        "split": args.split, "n_trajectories": len(ds),
        "boundary_cover_rate": quantiles(cover, "cover_"),
        "boundary_cover_pair_pass_gt_0.9": float(np.mean(cover >= 0.9)) if cover.size else None,
        "midpoint_continuity": float(mid.mean()) if mid.size else None,
        "corridor_success_rate": float(np.mean(corridors)) if corridors else None,
        "exact_safe_rate": float(saf.mean()) if saf.size else None,
        "ellipse_area_m2": quantiles(area),
        "path_spacing_m": quantiles(np.abs(np.diff(np.asarray(
            [0.0])))) if False else None,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False, default=float)
    print(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
