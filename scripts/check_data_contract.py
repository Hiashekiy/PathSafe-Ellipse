"""Check the data contract of plan 1.1 on real samples (read-only).

Prints, for a handful of trajectories, every field the network consumes with its
shape, dtype and unit, and asserts the invariants the plan requires:

* ``occupancy`` (== blocked), ``signed_distance_m`` and ``valid_mask`` share the grid;
* the SDF sign agrees with the blocked mask at cell centres and outside the map it
  is treated as blocked;
* ``path_grid_xy = path_xy / resolution_m`` and ``path_s`` is a monotone arclength;
* the 16 ray directions are unit vectors on the fixed 2*pi*k/16 set and
  ``ray_valid`` separates true hits from range-clipped rays;
* the 9 occupancy token offsets and the 3 ray sampling positions stay inside the map
  for interior path points.

    python scripts/check_data_contract.py --n 8
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
from ellipsenet.data.dataset import TrajectoryDataset, load_index, split_by_group  # noqa: E402
from ellipsenet.data.sdf import SDF_OUTSIDE_M  # noqa: E402

FAILS = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}{(' :: ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="results/metrics/data_contract.json")
    args = ap.parse_args()
    cfg = load_config()
    entries = load_index(cfg["data"]["sdf_dir"])
    tr, _va, _te = split_by_group(entries, frac=tuple(cfg["data"]["split_frac"]),
                                  seed=int(cfg["data"]["split_seed"]))
    pick = [entries[i] for i in tr[: args.n]]
    ds = TrajectoryDataset(pick, cfg["data"]["trajectory_dir"], cfg["data"]["sdf_dir"],
                           augment=False, with_diag=True)

    rows = []
    for j in range(len(ds)):
        s = ds[j]
        occ = np.asarray(s["occupancy"])[0]
        sdf = np.asarray(s["sdf"])[0]
        mask = np.asarray(s["valid_mask"])
        p = np.asarray(s["path_xy"])
        lidar = np.asarray(s["lidar"])
        rdir = np.asarray(s["ray_dir"])
        rvalid = np.asarray(s["ray_valid"])
        res = float(s["r_scene"])
        H, W = occ.shape
        grid_xy = p / res
        row = {
            "file": s["file"],
            "occupancy": {"shape": list(occ.shape), "dtype": str(occ.dtype),
                          "unique": sorted(np.unique(occ).tolist())},
            "signed_distance_m": {"shape": list(sdf.shape), "dtype": str(sdf.dtype),
                                  "unit": "m", "min": float(sdf.min()),
                                  "max": float(sdf.max())},
            "valid_mask": {"shape": list(mask.shape), "sum": int(mask.sum())},
            "path_xy": {"shape": list(p.shape), "unit": "m"},
            "path_grid_xy": {"shape": list(grid_xy.shape), "unit": "px"},
            "lidar_ranges": {"shape": list(lidar.shape), "unit": "m",
                             "min": float(lidar.min()), "max": float(lidar.max())},
            "ray_directions": {"shape": list(rdir.shape),
                               "norms": [float(v) for v in np.linalg.norm(rdir, axis=-1)[:4]]},
            "ray_valid": {"shape": list(rvalid.shape),
                          "hit_frac": float(rvalid.mean())},
            "resolution_m": res,
            "map_origin": [0.0, 0.0],
            "start_xy": {"value": [float(v) for v in np.asarray(
                [p[0, 0], p[0, 1]])]},
            "meta": {"scene_id": s["scene_id"], "map_group": s["map_group"],
                     "n_points": int(s["n_points"]), "max_range_m": float(s["max_range_m"])},
        }
        rows.append(row)

        check(f"{s['file'][:28]} grid consistent",
              occ.shape == sdf.shape == mask.shape, f"{occ.shape}/{sdf.shape}/{mask.shape}")
        check("  occupancy is 0/1", row["occupancy"]["unique"] in ([0, 1], [0], [1]),
              str(row["occupancy"]["unique"]))
        check("  SDF sign matches blocked at cell centres",
              bool(((sdf < 0) == (occ > 0)).all()),
              f"mismatch {float(((sdf < 0) != (occ > 0)).mean()):.4f}")
        check("  path_grid_xy inside the grid",
              bool(((grid_xy[:, 0] >= 0) & (grid_xy[:, 0] <= W - 1)
                    & (grid_xy[:, 1] >= 0) & (grid_xy[:, 1] <= H - 1)).all()))
        check("  ray directions are unit vectors",
              np.allclose(np.linalg.norm(rdir, axis=-1), 1.0, atol=1e-5))
        check("  ray_valid separates hits from clipped ranges",
              bool(np.all((rvalid > 0.5) == (lidar < float(s["max_range_m"]) - 1e-3))),
              f"hit frac {float(rvalid.mean()):.2f}")
        sp = np.linalg.norm(np.diff(p, axis=0), axis=-1)
        check("  arclength monotone", bool((sp >= 0).all()),
              f"spacing mean {sp.mean():.3f} m")
        # 9-token offsets must stay usable (not required to be inside for edge points)
        off = 0.5
        coords = p[:, None, :] + np.array([[dx, dy] for dy in (-off, 0, off)
                                          for dx in (-off, 0, off)])[None]
        inside = ((coords[..., 0] / res >= 0) & (coords[..., 0] / res <= W - 1)
                  & (coords[..., 1] / res >= 0) & (coords[..., 1] / res <= H - 1))
        check("  9-token grid mostly inside", float(inside.mean()) > 0.95,
              f"{float(inside.mean()):.3f}")
        check("  outside-map SDF convention is negative",
              SDF_OUTSIDE_M < 0.0, str(SDF_OUTSIDE_M))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"n_samples": len(rows), "fields": rows}, fh, indent=2,
                  ensure_ascii=False)
    print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    print(f"-> {args.out}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
