"""Diagnose why wall-adjacent anchors get small ellipses.

For the tightest anchors of one path this reports
  * the clearance D(q_i) and the current ellipse,
  * whether the ORIGINAL path point p_i is inside the ellipse (to test the
    "ellipse must contain p_i" hypothesis: V2 does not impose that),
  * the maximal inscribed ellipse found by scanning candidate centres around
    q_i (so we can see whether a bigger ellipse exists next to the wall),
  * the same path re-optimised with a stronger clearance weight.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from pathsafe_ellipse.map_io import build_map_catalog, load_moving_ai_map  # noqa: E402
from pathsafe_ellipse.v2.centerline import build_path, optimise_centerline  # noqa: E402
from pathsafe_ellipse.v2.ellipse import make_label, midpoint_metric  # noqa: E402
from pathsafe_ellipse.v2.scene import build_scene  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "data_generation_v2.yaml"))
    ap.add_argument("--map", default="Berlin_2_512_part_0_0.map")
    ap.add_argument("--crop", default="247,209")
    ap.add_argument("--field", type=float, default=48.0)
    ap.add_argument("--w-clear", type=float, default=1.2)
    ap.add_argument("--scan-radius", type=float, default=5.0)
    ap.add_argument("--scan-step", type=float, default=1.0)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["paths"]["map_dir"] = os.path.normpath(os.path.join(PROJECT_ROOT, cfg["paths"]["map_dir"]))
    cfg["centerline"]["w_clear"] = args.w_clear

    rec = next(r for r in build_map_catalog(cfg["paths"]["map_dir"], 0.25)
               if r.parsed_ok and r.file_name == args.map)
    src, meta = load_moving_ai_map(os.path.join(cfg["paths"]["map_dir"], rec.rel_path))
    r0 = float(meta["resolution"] or 0.25)
    crop_px = int(round(args.field / r0))
    x0, y0 = (int(v) for v in args.crop.split(","))
    scene = build_scene(src, r0, (x0, y0), crop_px, int(cfg["scene"]["size"]), args.field,
                        float(cfg["robot"]["radius_m"]), float(cfg["robot"]["safety_margin_m"]),
                        int(cfg["robot"]["edt_upsample"]), "diag", rec.file_name)
    boundary = scene.obstacle_boundary_cells()

    # reproduce the first path of the pilot scene with the same RNG stream
    rng = np.random.default_rng(42)
    pair = None
    from pathsafe_ellipse.v2.pipeline import sample_start_goal
    for _ in range(3):
        pair = sample_start_goal(scene, rng, cfg, [])
        if pair is not None:
            break
    path = build_path(scene, pair[0], pair[1], cfg)
    print(f"scene {rec.file_name} crop=({x0},{y0}) FOV={args.field} r_scene={scene.r_scene:.4f} "
          f"w_clear={args.w_clear}  stations={len(path.p_xy)}")
    print(f"clearance: A* mean {np.mean(path.clearance_p):.2f} m -> "
          f"centreline mean {np.mean(path.clearance_q):.2f} m  (status {path.status})")

    # anchors: the sparse set used by the pipeline
    from pathsafe_ellipse.v2.pipeline import anchor_indices
    anchors = anchor_indices(scene, path, cfg["anchor"])
    labels = [make_label(scene, path.q_xy[i], boundary, cfg) for i in anchors]
    areas = np.array([l.area for l in labels])
    order = np.argsort(areas)[:3]
    print(f"anchors={len(anchors)}  area p50={np.median(areas):.2f} m^2  "
          f"min={areas.min():.2f} max={areas.max():.2f}")

    rows = []
    for oi in order:
        i = anchors[oi]
        q, p = path.q_xy[i], path.p_xy[i]
        lab = labels[oi]
        # 1) is the ORIGINAL path point inside the ellipse? (V2 does not require it)
        rho_p = midpoint_metric(lab.S, q, p)
        # 2) scan candidate centres: how much area is available next to the wall?
        best = {"area": lab.area, "offset": np.zeros(2), "clearance": lab.clearance,
                "ab": lab.ab}
        grid = np.arange(-args.scan_radius, args.scan_radius + 1e-9, args.scan_step)
        for dx in grid:
            for dy in grid:
                c = q + np.array([dx, dy])
                if not bool(scene.is_free(c[None, :])[0]):
                    continue
                if abs(dx) + abs(dy) < 1e-9:
                    continue
                cand = make_label(scene, c, boundary, cfg)
                if cand.safety_verified and cand.area > best["area"]:
                    best = {"area": cand.area, "offset": np.array([dx, dy]),
                            "clearance": cand.clearance, "ab": cand.ab}
        rows.append({
            "station": int(i),
            "q_xy_m": q.round(3).tolist(),
            "p_xy_m": p.round(3).tolist(),
            "clearance_q_m": round(float(lab.clearance), 3),
            "ellipse_ab_m": [round(float(lab.ab[0]), 3), round(float(lab.ab[1]), 3)],
            "area_m2": round(float(lab.area), 3),
            "rho_of_p_i": round(float(rho_p), 4),
            "p_i_inside": bool(rho_p <= 1.0),
            "best_area_m2": round(float(best["area"]), 3),
            "best_offset_m": np.asarray(best["offset"]).round(2).tolist(),
            "best_clearance_m": round(float(best["clearance"]), 3),
            "best_ab_m": [round(float(best["ab"][0]), 3), round(float(best["ab"][1]), 3)],
            "area_ratio": round(float(best["area"] / max(lab.area, 1e-9)), 2),
        })
        print(f"  station {i}: D(q)={lab.clearance:.2f} m -> a={lab.ab[0]:.2f} b={lab.ab[1]:.2f} "
              f"area={lab.area:.2f} m2; rho(p_i)={rho_p:.3f} (inside={rho_p <= 1.0}); "
              f"best nearby centre offset {np.asarray(best['offset']).round(2)} -> "
              f"area={best['area']:.2f} m2 (x{best['area']/max(lab.area,1e-9):.1f}) "
              f"at clearance {best['clearance']:.2f} m")

    out = os.path.join(PROJECT_ROOT, "reports_v2", f"v2_small_ellipse_diag_w{args.w_clear}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"scene": rec.file_name, "w_clear": args.w_clear,
                   "clearance_p_mean": float(np.mean(path.clearance_p)),
                   "clearance_q_mean": float(np.mean(path.clearance_q)),
                   "area_p50": float(np.median(areas)),
                   "rows": rows}, fh, indent=2)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
