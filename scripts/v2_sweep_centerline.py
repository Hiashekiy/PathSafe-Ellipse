"""Sweep the centreline weights / offset cap and report how the label size responds.

Answers: "the ellipse is small although free space exists nearby - what is the
binding constraint?" by comparing (w_clear, r_max) configurations on one path.
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
from pathsafe_ellipse.v2.centerline import build_path  # noqa: E402
from pathsafe_ellipse.v2.ellipse import make_label  # noqa: E402
from pathsafe_ellipse.v2.pipeline import anchor_indices, sample_start_goal  # noqa: E402
from pathsafe_ellipse.v2.scene import build_scene  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "data_generation_v2.yaml"))
    ap.add_argument("--map", default="Berlin_2_512_part_0_0.map")
    ap.add_argument("--crop", default="247,209")
    ap.add_argument("--field", type=float, default=48.0)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        base = yaml.safe_load(fh)
    base["paths"]["map_dir"] = os.path.normpath(os.path.join(PROJECT_ROOT, base["paths"]["map_dir"]))
    rec = next(r for r in build_map_catalog(base["paths"]["map_dir"], 0.25)
               if r.parsed_ok and r.file_name == args.map)
    src, meta = load_moving_ai_map(os.path.join(base["paths"]["map_dir"], rec.rel_path))
    r0 = float(meta["resolution"] or 0.25)
    crop_px = int(round(args.field / r0))
    x0, y0 = (int(v) for v in args.crop.split(","))
    scene = build_scene(src, r0, (x0, y0), crop_px, int(base["scene"]["size"]), args.field,
                        float(base["robot"]["radius_m"]), float(base["robot"]["safety_margin_m"]),
                        int(base["robot"]["edt_upsample"]), "sweep", rec.file_name)
    boundary = scene.obstacle_boundary_cells()

    rng = np.random.default_rng(42)
    pair = sample_start_goal(scene, rng, base, [])

    rows = []
    for w_clear, r_max in ((1.2, 3.0), (4.0, 3.0), (4.0, 4.0), (8.0, 6.0), (12.0, 8.0)):
        cfg = json.loads(json.dumps(base))     # deep copy
        cfg["centerline"]["w_clear"] = w_clear
        cfg["centerline"]["r_max_m"] = r_max
        path = build_path(scene, pair[0], pair[1], cfg)
        anchors = anchor_indices(scene, path, cfg["anchor"])
        labels = [make_label(scene, path.q_xy[i], boundary, cfg) for i in anchors]
        areas = np.array([l.area for l in labels])
        clr = np.array([l.clearance for l in labels])
        offsets = np.linalg.norm(path.q_xy - path.p_xy, axis=1)
        row = {
            "w_clear": w_clear,
            "r_max_m": r_max,
            "centreline_status": path.status,
            "max_offset_m": round(float(offsets.max()), 2),
            "clearance_q_mean_m": round(float(clr.mean()), 2),
            "clearance_q_min_m": round(float(clr.min()), 2),
            "area_p50_m2": round(float(np.median(areas)), 2),
            "area_min_m2": round(float(areas.min()), 2),
            "area_max_m2": round(float(areas.max()), 2),
            "n_anchors": int(len(anchors)),
            "n_area_lt_5m2": int((areas < 5.0).sum()),
            "all_verified": bool(all(l.safety_verified for l in labels)),
        }
        rows.append(row)
        print(f"w_clear={w_clear:5.1f} r_max={r_max:4.1f} | offset max {row['max_offset_m']:5.2f} m | "
              f"D(q) mean {row['clearance_q_mean_m']:5.2f} min {row['clearance_q_min_m']:5.2f} | "
              f"area p50 {row['area_p50_m2']:7.2f} min {row['area_min_m2']:6.2f} max {row['area_max_m2']:7.2f} | "
              f"anchors<5m2: {row['n_area_lt_5m2']}/{row['n_anchors']} | verified {row['all_verified']} "
              f"| {row['centreline_status']}")

    out = os.path.join(PROJECT_ROOT, "reports_v2", "v2_centerline_sweep.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"map": rec.file_name, "rows": rows}, fh, indent=2)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
