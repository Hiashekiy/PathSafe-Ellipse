"""Diagnose the V2.1 SLSQP run on one path: per-init convergence and binding limits."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import yaml
from scipy.optimize import minimize

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from pathsafe_ellipse.map_io import build_map_catalog, load_moving_ai_map  # noqa: E402
from pathsafe_ellipse.v2 import centerline as C  # noqa: E402
from pathsafe_ellipse.v2.pipeline import sample_start_goal  # noqa: E402
from pathsafe_ellipse.v2.scene import build_scene  # noqa: E402


def main() -> int:
    with open(os.path.join(PROJECT_ROOT, "configs", "data_generation_v2_1.yaml"), encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["paths"]["map_dir"] = os.path.normpath(os.path.join(PROJECT_ROOT, cfg["paths"]["map_dir"]))
    rec = next(r for r in build_map_catalog(cfg["paths"]["map_dir"], 0.25)
               if r.parsed_ok and r.file_name == "Berlin_2_512_part_0_0.map")
    src, meta = load_moving_ai_map(os.path.join(cfg["paths"]["map_dir"], rec.rel_path))
    r0 = float(meta["resolution"] or 0.25)
    crop = int(round(48.0 / r0))
    scene = build_scene(src, r0, (247, 209), crop, 256, 48.0, 0.25, 0.04, 4, "dbg", rec.file_name)
    rng = np.random.default_rng(42)
    pair = sample_start_goal(scene, rng, cfg, [])
    cells = C.astar_m(scene, pair[0], pair[1])
    p, _ = C.resample_path(scene, cells, cfg["resample"])
    print(f"stations={len(p)} path length {np.sum(np.linalg.norm(np.diff(p,axis=0),axis=1)):.2f} m")

    cons = C.build_route_constraints(p, scene, cfg)
    R, L_i, eta = cons["R"], cons["L_i"], cons["eta"]
    print(f"R_i: min {R.min():.2f} mean {R.mean():.2f} max {R.max():.2f} m  "
          f"(r_max {cfg['centerline']['r_max_m']});  stations at r_max: "
          f"{int(np.sum(R >= cfg['centerline']['r_max_m'] - 1e-6))}/{len(R)}")
    print(f"tangent limit eta*L_i: mean {np.mean(eta*L_i):.3f} m; L_ref {cons['L_ref']:.3f} m")
    print(f"d_safe {cons['d_safe']:.2f} m; D(p): mean {np.mean(scene.distance(p)):.2f} "
          f"min {np.min(scene.distance(p)):.2f}")

    slsqp_cons, m = C.make_constraints(scene, p, cons, cfg["centerline"])

    def fun(x):
        return C.centreline_objective(scene, C._unpack(x, p), p, cons, cfg["centerline"])[0]

    def jac(x):
        return C.centreline_grad(scene, C._unpack(x, p), p, cons, cfg["centerline"])

    for name, q0 in C.make_clearance_initializations(scene, p, cons, cfg):
        x0 = C._pack(q0)
        # constraint activity of the seed
        vals = [c["fun"](x0).min() for c in slsqp_cons]
        res = minimize(fun, x0, jac=jac, method="SLSQP", constraints=slsqp_cons,
                       options={"maxiter": int(cfg["centerline"]["max_iter"]),
                                "ftol": float(cfg["centerline"]["ftol"])})
        q = C._unpack(np.asarray(res.x), p)
        off = np.linalg.norm(q - p, axis=1)
        D = scene.distance(q)
        J, parts = C.centreline_objective(scene, q, p, cons, cfg["centerline"])
        print(f"  [{name:18s}] nit={res.nit:4d} success={res.success} msg='{res.message}' "
              f"J={J:+.4f} (JD {parts['J_D']:.3f} Js {parts['J_s']:.4f} Jp {parts['J_p']:.4f}) "
              f"offset max {off.max():.2f} mean {off.mean():.2f} | D(q) mean {D.mean():.2f} "
              f"| seed min-constraint {np.round(vals,3)}")
        final = [c["fun"](np.asarray(res.x)).min() for c in slsqp_cons]
        print(f"      final min per constraint block: {np.round(final, 4)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
