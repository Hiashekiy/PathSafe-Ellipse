"""Why does SLSQP report 'iteration limit reached'? Sweep ftol / max_iter."""
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from pathsafe_ellipse.v2 import centerline as C  # noqa: E402
from v2_1_scenarios import build_layout, load_cfg, make_scene  # noqa: E402


def main() -> int:
    cfg0 = load_cfg(os.path.join(PROJECT_ROOT, "configs", "data_generation_v2_1.yaml"))
    for name in ("wall_one_side", "s_channel"):
        occ, start, goal = build_layout(name)
        scene = make_scene(occ, cfg0, name)
        cells = C.astar_m(scene, start, goal)
        p, _ = C.resample_path(scene, cells, cfg0["resample"])
        idx = np.unique(np.concatenate([[0], np.round(np.linspace(1, len(p) - 2, 32)).astype(int),
                                        [len(p) - 1]]))
        pk = p[idx]
        cfg = json.loads(json.dumps(cfg0))
        cons = C.build_route_constraints(pk, scene, cfg)
        cache = C._FieldCache(scene, pk)
        cl, m = C.make_constraints(scene, pk, cons, cfg["centerline"], cache=cache)

        def fun(x):
            return C.centreline_objective(scene, cache.q(x), pk, cons, cfg["centerline"],
                                          field=cache, x=x)[0]

        def jac(x):
            return C.centreline_grad(scene, cache.q(x), pk, cons, cfg["centerline"],
                                     field=cache, x=x)

        q0 = p[idx]
        x0 = C._pack(q0)
        print(f"\n{name}: {len(pk)} stations, {m} free vars")
        for ftol in (1e-9, 1e-6, 1e-4):
            for maxiter in (200, 600, 1500):
                cache = C._FieldCache(scene, pk)
                res = minimize(fun, x0, jac=jac, method="SLSQP", constraints=cl,
                               options={"maxiter": maxiter, "ftol": ftol})
                q = C._unpack(np.asarray(res.x), pk)
                J = C.centreline_objective(scene, q, pk, cons, cfg["centerline"])[0]
                viol = max(-float(np.min(c["fun"](np.asarray(res.x)))) for c in cl)
                print(f"  ftol {ftol:.0e} iters {maxiter:5d} | success {res.success!s:5s} "
                      f"nit {res.nit:5d} | J {J:+.6f} | max_viol {viol:.2e} | "
                      f"msg '{res.message}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
