"""V2.1 stage 2: choose the effort settings from measurements.

Sweeps (max_optimised_stations, segment_collocation_points, max_iter, multi_start)
on the two hardest scenarios and reports quality vs runtime against the
expensive 600-iteration / 3-start baseline.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from pathsafe_ellipse.v2 import centerline as C  # noqa: E402
from pathsafe_ellipse.v2.pipeline import anchor_indices  # noqa: E402
from pathsafe_ellipse.v2.ellipse import make_label  # noqa: E402
from v2_1_scenarios import build_layout, load_cfg, make_scene  # noqa: E402

BASE = dict(knots=32, coll=2, iters=600, inits=3)
GRID = [
    dict(knots=32, coll=2, iters=600, inits=3),   # baseline
    dict(knots=32, coll=2, iters=200, inits=1),
    dict(knots=24, coll=2, iters=200, inits=1),
    dict(knots=24, coll=1, iters=200, inits=1),
    dict(knots=20, coll=1, iters=150, inits=1),
    dict(knots=16, coll=1, iters=150, inits=1),
]


def main() -> int:
    cfg0 = load_cfg(os.path.join(PROJECT_ROOT, "configs", "data_generation_v2_1.yaml"))
    rows = []
    for name in ("wall_one_side", "s_channel"):
        occ, start, goal = build_layout(name)
        scene = make_scene(occ, cfg0, name)
        boundary = scene.obstacle_boundary_cells()
        for g in GRID:
            cfg = json.loads(json.dumps(cfg0))
            cfg["centerline"]["max_optimised_stations"] = g["knots"]
            cfg["centerline"]["segment_collocation_points"] = g["coll"]
            cfg["centerline"]["multi_start_count"] = g["inits"]
            cfg["centerline"]["max_iter"] = g["iters"]
            t0 = time.perf_counter()
            path = C.build_path(scene, start, goal, cfg)
            dt = (time.perf_counter() - t0) * 1e3
            anchors = anchor_indices(scene, path, cfg["anchor"])
            labels = [make_label(scene, path.q_xy[i], boundary, cfg) for i in anchors]
            areas = np.array([l.area for l in labels])
            row = {"scenario": name, **g, "wall_ms": round(dt, 1),
                   "solve_ms": round(path.centerline_solver_time_ms, 1),
                   "gain": round(float(path.centerline_checks["clearance_gain_mean"]), 4),
                   "offset_max": round(float(path.centerline_checks["max_offset_m"]), 3),
                   "area_p50": round(float(np.median(areas)), 4),
                   "area_min": round(float(areas.min()), 4),
                   "homotopy": bool(path.centerline_checks["homotopy_verified"]),
                   "status": path.status}
            rows.append(row)
            print(f"{name:15s} knots {g['knots']:3d} coll {g['coll']} iters {g['iters']:3d} "
                  f"inits {g['inits']} | solve {row['solve_ms']:7.0f} ms | gain {row['gain']:+.4f} "
                  f"| area p50 {row['area_p50']:7.3f} | homo {row['homotopy']} | {row['status']}")

    # relative quality loss vs baseline
    print("\nquality retention vs baseline (gain / area p50):")
    for name in ("wall_one_side", "s_channel"):
        base = next(r for r in rows if r["scenario"] == name and
                    all(r[k] == BASE[k] for k in BASE))
        for r in [r for r in rows if r["scenario"] == name]:
            g_ret = (r["gain"] / base["gain"] * 100) if base["gain"] else 100.0
            a_ret = (r["area_p50"] / base["area_p50"] * 100) if base["area_p50"] else 100.0
            print(f"  {name:15s} knots {r['knots']:3d} coll {r['coll']} iters {r['iters']:3d} "
                  f"inits {r['inits']} | gain {g_ret:6.1f}% | area {a_ret:6.1f}% | "
                  f"{r['solve_ms']:7.0f} ms")

    dst = os.path.join(cfg0["paths"]["report_dir"], "effort_sweep.json")
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump({"grid": GRID, "rows": rows}, fh, indent=2)
    print(f"-> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
