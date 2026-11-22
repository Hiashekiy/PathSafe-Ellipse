"""V2.1 stage 2: profile the centreline and find data-driven effort settings.

1. Phase timings of build_path (A*, resample, solve, homotopy, validate).
2. cProfile of the optimiser (where SLSQP vs our fun/jac/constraints spend time).
3. Budget sweep: max_iter x multi_start vs quality (gain, area) and time.
4. Ellipse timing breakdown per anchor (analysis only, algorithm untouched).
"""
from __future__ import annotations

import cProfile
import io
import json
import os
import pstats
import sys
import time

import numpy as np
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from pathsafe_ellipse.v2 import centerline as C  # noqa: E402
from pathsafe_ellipse.v2.ellipse import exact_clearance, make_label, build_halfspaces  # noqa: E402
from pathsafe_ellipse.v2.pipeline import anchor_indices  # noqa: E402
from v2_1_scenarios import build_layout, load_cfg, make_scene  # noqa: E402


def phases(scene, start, goal, cfg):
    t = {}
    t0 = time.perf_counter()
    cells = C.astar_m(scene, start, goal)
    t["astar"] = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    p, _ = C.resample_path(scene, cells, cfg["resample"])
    t["resample"] = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    q, info = C.optimise_centerline(scene, p, cfg)
    t["solve"] = (time.perf_counter() - t0) * 1e3
    q[0], q[-1] = p[0], p[-1]
    t0 = time.perf_counter()
    lam, n_lam = C.homotopy_max_lambda(scene, p, q,
                                       float(cfg["centerline"]["segment_check_step_cells"]),
                                       float(cfg["centerline"]["homotopy_check_step_cells"]))
    t["homotopy"] = (time.perf_counter() - t0) * 1e3
    t["homotopy_states"] = n_lam
    t["lambda_max"] = lam
    t0 = time.perf_counter()
    chk = C.validate_centerline(scene, p, q, cfg, lam_max=lam)
    t["validate"] = (time.perf_counter() - t0) * 1e3
    t["n_stations"] = len(p)
    t["n_free"] = len(p) - 2
    t["coarse"] = info.get("coarse_stations", len(p))
    t["iters"] = info.get("iterations", 0)
    t["gain"] = chk["clearance_gain_mean"]
    t["offset_max"] = chk["max_offset_m"]
    return p, q, t


def main() -> int:
    cfg = load_cfg(os.path.join(PROJECT_ROOT, "configs", "data_generation_v2_1.yaml"))
    out = {}
    for name in ("wall_one_side", "s_channel"):
        occ, start, goal = build_layout(name)
        scene = make_scene(occ, cfg, name)
        p, q, t = phases(scene, start, goal, cfg)
        print(f"[{name}] stations {t['n_stations']} (coarse {t['coarse']}, iters {t['iters']}) "
              f"| astar {t['astar']:.1f} | resample {t['resample']:.1f} | solve {t['solve']:.0f} "
              f"| homotopy {t['homotopy']:.0f} ({t['homotopy_states']} states) "
              f"| validate {t['validate']:.1f} ms | gain {t['gain']:+.2f} m")
        out[name] = t

    # ---- cProfile of the optimiser on the long S-path
    occ, start, goal = build_layout("s_channel")
    scene = make_scene(occ, cfg, "s_channel")
    p, _ = C.resample_path(scene, C.astar_m(scene, start, goal), cfg["resample"])
    pr = cProfile.Profile()
    pr.enable()
    C.optimise_centerline(scene, p, cfg)
    pr.disable()
    st = io.StringIO()
    pstats.Stats(pr, stream=st).sort_stats("cumulative").print_stats(18)
    prof_txt = st.getvalue()
    keep = [ln for ln in prof_txt.splitlines()
            if any(k in ln for k in ("cumtime", "pathsafe", "scipy", "minimize", "_minimize",
                                     "gradient", "distance", "solve", "is_free", "homotopy"))]
    print("\n".join(keep[:22]))

    # ---- budget sweep (quality vs time)
    sweep = []
    for name in ("wall_one_side", "s_channel"):
        occ, start, goal = build_layout(name)
        scene = make_scene(occ, cfg, name)
        boundary = scene.obstacle_boundary_cells()
        for max_iter in (100, 200, 300, 600):
            for n_init in (1, 3):
                c = json.loads(json.dumps(cfg))
                c["centerline"]["max_iter"] = 600        # 600 is the base the scaling uses
                c["centerline"]["multi_start_count"] = n_init
                c["centerline"]["_iter_override"] = max_iter
                t0 = time.perf_counter()
                path = C.build_path(scene, start, goal, c)
                dt = (time.perf_counter() - t0) * 1e3
                anchors = anchor_indices(scene, path, c["anchor"])
                labels = [make_label(scene, path.q_xy[i], boundary, c) for i in anchors]
                areas = np.array([l.area for l in labels])
                row = {"scenario": name, "max_iter": max_iter, "n_init": n_init,
                       "wall_ms": round(dt, 1),
                       "centreline_ms": round(path.centerline_solver_time_ms, 1),
                       "gain": round(float(path.centerline_checks["clearance_gain_mean"]), 3),
                       "offset_max": round(float(path.centerline_checks["max_offset_m"]), 3),
                       "area_p50": round(float(np.median(areas)), 3),
                       "homotopy": bool(path.centerline_checks["homotopy_verified"]),
                       "status": path.status}
                sweep.append(row)
                print(f"  {name:15s} iter {max_iter:4d} init {n_init} | wall {dt:7.0f} ms "
                      f"(solve {row['centreline_ms']:7.0f}) | gain {row['gain']:+.3f} "
                      f"| area p50 {row['area_p50']:7.3f} | homo {row['homotopy']} | {row['status']}")

    # ---- ellipse timing breakdown (analysis only)
    occ, start, goal = build_layout("wall_one_side")
    scene = make_scene(occ, cfg, "wall_one_side")
    boundary = scene.obstacle_boundary_cells()
    path = C.build_path(scene, start, goal, cfg)
    anchors = anchor_indices(scene, path, cfg["anchor"])[:12]
    t = {"build_halfspaces": 0.0, "solve": 0.0, "verify": 0.0, "total": 0.0}
    for i in anchors:
        qq = path.q_xy[i]
        t0 = time.perf_counter()
        hs = build_halfspaces(np.eye(2) * (0.5 ** 2), qq, boundary, scene.r_scene,
                              float(cfg["ellipse"]["epsilon_m"]), int(cfg["ellipse"]["sectors"]),
                              int(cfg["ellipse"]["direction_candidates"]),
                              int(cfg["ellipse"]["active_target"]),
                              int(cfg["ellipse"]["active_hard_limit"]))
        t["build_halfspaces"] += (time.perf_counter() - t0) * 1e3
        n = np.stack([h.n for h in hs]) if hs else np.zeros((0, 2))
        b2 = np.array([h.d2 for h in hs]) if hs else np.zeros(0)
        t0 = time.perf_counter()
        S, _st = C.__dict__ and __import__("pathsafe_ellipse.v2.ellipse", fromlist=["x"]).solve_s_barrier(
            n, b2, float(cfg["ellipse"]["min_eig_m2"]), float(cfg["ellipse"]["r_axis_max_m"]) ** 2)
        t["solve"] += (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        exact_clearance(S, qq, scene.blocked.astype(np.uint8), scene.r_scene)
        t["verify"] += (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        make_label(scene, qq, boundary, cfg)
        t["total"] += (time.perf_counter() - t0) * 1e3
    n_a = max(len(anchors), 1)
    ell = {k: round(v / n_a, 2) for k, v in t.items()}
    print(f"\nellipse per anchor (n={n_a}): total {ell['total']} ms = "
          f"halfspaces {ell['build_halfspaces']} + solve {ell['solve']} + verify {ell['verify']} "
          f"(one-shot parts; make_label repeats them per outer round)")

    res = {"phases": out, "budget_sweep": sweep, "ellipse_ms_per_anchor": ell,
           "profile_top": keep[:22]}
    dst = os.path.join(cfg["paths"]["report_dir"], "profiling.json")
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print(f"-> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
