"""V2 self-check: barrier 2x2 max-det vs the CVXPY/CLARABEL reference.

Spec 4.6.1 requires the reference solver to validate the mathematical model
before the fast solver is trusted, so this compares the two on real active
half-space sets produced by the pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from pathsafe_ellipse.map_io import build_map_catalog, load_moving_ai_map  # noqa: E402
from pathsafe_ellipse.v2.centerline import build_path  # noqa: E402
from pathsafe_ellipse.v2.ellipse import (  # noqa: E402
    make_label, solve_s_barrier, solve_s_cvxpy,
)
from pathsafe_ellipse.v2.pipeline import anchor_indices, sample_start_goal  # noqa: E402
from pathsafe_ellipse.v2.scene import build_scene  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "data_generation_v2.yaml"))
    ap.add_argument("--n-anchors", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for k in ("map_dir",):
        cfg["paths"][k] = os.path.normpath(os.path.join(PROJECT_ROOT, cfg["paths"][k]))

    records = [r for r in build_map_catalog(cfg["paths"]["map_dir"], 0.25) if r.parsed_ok]
    rng = np.random.default_rng(args.seed)
    rec = records[int(rng.integers(0, len(records)))]
    src, meta = load_moving_ai_map(os.path.join(cfg["paths"]["map_dir"], rec.rel_path))
    r0 = float(meta["resolution"] or 0.25)
    field_m = 64.0
    crop_px = int(round(field_m / r0))
    h, w = src.shape
    x0 = int(rng.integers(0, max(1, w - crop_px)))
    y0 = int(rng.integers(0, max(1, h - crop_px)))
    scene = build_scene(src, r0, (x0, y0), crop_px, int(cfg["scene"]["size"]), field_m,
                        float(cfg["robot"]["radius_m"]), float(cfg["robot"]["safety_margin_m"]),
                        int(cfg["robot"]["edt_upsample"]), "check", rec.file_name)
    boundary = scene.obstacle_boundary_cells()

    pair = sample_start_goal(scene, rng, cfg, [])
    if pair is None:
        print("no start/goal pair")
        return 2
    path = build_path(scene, pair[0], pair[1], cfg)
    if path is None:
        print("no path")
        return 2
    anchors = anchor_indices(scene, path, cfg["anchor"])[: args.n_anchors]
    print(f"scene {rec.file_name} crop={crop_px}px r_scene={scene.r_scene:.4f} "
          f"anchors compared: {len(anchors)}")

    rows = []
    for i in anchors:
        lab = make_label(scene, path.q_xy[i], boundary, cfg)
        if not lab.halfspaces:
            continue
        n = np.stack([h.n for h in lab.halfspaces])
        b2 = np.array([h.d2 for h in lab.halfspaces])
        S_ref, st_ref = solve_s_cvxpy(n, b2, float(cfg["ellipse"]["min_eig_m2"]),
                                      float(cfg["ellipse"]["r_axis_max_m"]) ** 2)
        if S_ref is None:
            rows.append({"anchor": int(i), "cvxpy": st_ref, "ok": False})
            continue
        eig_bar = np.sort(np.linalg.eigvalsh(lab.S))[::-1]
        eig_cv = np.sort(np.linalg.eigvalsh(S_ref))[::-1]
        rows.append({
            "anchor": int(i),
            "n_constraints": int(len(n)),
            "barrier_axes_m": [float(np.sqrt(eig_bar[0])), float(np.sqrt(eig_bar[1]))],
            "cvxpy_axes_m": [float(np.sqrt(eig_cv[0])), float(np.sqrt(eig_cv[1]))],
            "d_axis_m": float(np.max(np.abs(np.sqrt(eig_bar) - np.sqrt(eig_cv)))),
            "d_area_m2": float(abs(np.pi * np.sqrt(np.linalg.det(lab.S))
                                   - np.pi * np.sqrt(np.linalg.det(S_ref)))),
            "barrier_status": lab.status,
            "cvxpy_status": st_ref,
            "ok": True,
        })
    ok_rows = [r for r in rows if r.get("ok")]
    worst_axis = max((r["d_axis_m"] for r in ok_rows), default=None)
    worst_area = max((r["d_area_m2"] for r in ok_rows), default=None)
    print(f"compared {len(ok_rows)} ellipses: worst |d semi-axis| = {worst_axis:.3e} m, "
          f"worst |d area| = {worst_area:.3e} m^2")
    out = os.path.join(PROJECT_ROOT, "reports_v2", "v2_solver_crosscheck.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"scene": rec.file_name, "rows": rows,
                   "worst_d_axis_m": worst_axis, "worst_d_area_m2": worst_area}, fh, indent=2)
    print(f"-> {out}")
    return 0 if (worst_axis is not None and worst_axis < 0.02) else 1


if __name__ == "__main__":
    raise SystemExit(main())
