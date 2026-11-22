"""V2.1 stage 1 automated tests (spec section 3.6).

Ten checks; every failure is reported with numbers.  Run:
    python scripts/v2_1_tests.py
"""
from __future__ import annotations

import json
import os
import sys
import traceback

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from pathsafe_ellipse.v2 import centerline as C  # noqa: E402
from pathsafe_ellipse.v2.scene import build_scene  # noqa: E402
from v2_1_scenarios import build_layout, load_cfg, make_scene  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append({"test": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def random_dirs(rng, k):
    th = rng.uniform(0, 2 * np.pi, k)
    return np.stack([np.cos(th), np.sin(th)], axis=1)


# --------------------------------------------------------------------------- #
def test_distance_gradient(scene, cfg, rng):
    """1. bilinear distance field derivative vs central differences."""
    lo, hi = 1.0, scene.size * scene.r_scene - 1.0
    pts = rng.uniform(lo, hi, size=(200, 2))
    h = 1e-5
    errs = []
    for direction in random_dirs(rng, 6):
        d_plus = scene.distance(pts + h * direction)
        d_minus = scene.distance(pts - h * direction)
        fd = (d_plus - d_minus) / (2 * h)
        an = np.sum(scene.gradient(pts) * direction, axis=1)
        errs.append(float(np.max(np.abs(fd - an))))
    worst = max(errs)
    check("distance field gradient == central differences (random directions)",
          worst < 2e-3, f"worst rel/abs error {worst:.3e} m/m")


def test_objective_gradient(scene, cfg, rng):
    """2. analytic objective gradient vs random-direction finite differences."""
    _, start, goal = build_layout("wall_one_side")
    cells = C.astar_m(scene, start, goal)
    p, _ = C.resample_path(scene, cells, cfg["resample"])
    idx = np.unique(np.concatenate([[0], np.round(np.linspace(1, len(p) - 2, 20)).astype(int),
                                    [len(p) - 1]]))
    pk = p[idx]
    cons = C.build_route_constraints(pk, scene, cfg)
    # disable the clearance cap for the test: min(D, cap) is intentionally
    # non-smooth at D == cap and would pollute the finite-difference comparison
    cc = dict(cfg["centerline"])
    cc["clearance_cap_m"] = 1.0e9
    q = C._unpack(np.random.default_rng(1).normal(0, 0.05, size=2 * (len(pk) - 2)), pk)
    g = C.centreline_grad(scene, q, pk, cons, cc)
    h = 1e-6
    errs = []
    for direction in random_dirs(np.random.default_rng(2), 5):
        dx = np.repeat(direction[None, :], len(pk) - 2, axis=0).ravel()
        f_plus = C.centreline_objective(scene, C._unpack(C._pack(q) + h * dx, pk),
                                        pk, cons, cc)[0]
        f_minus = C.centreline_objective(scene, C._unpack(C._pack(q) - h * dx, pk),
                                         pk, cons, cc)[0]
        fd = (f_plus - f_minus) / (2 * h)
        an = float(g @ dx)
        errs.append(abs(fd - an) / max(1.0, abs(fd)))
    worst = max(errs)
    check("centreline objective gradient == finite differences (random directions)",
          worst < 1e-5, f"worst relative error {worst:.3e}")


def test_constraint_jacobians(scene, cfg):
    """3. analytic constraint Jacobians vs random-direction finite differences."""
    _, start, goal = build_layout("wall_sideways" if False else "wall_one_side")
    cells = C.astar_m(scene, start, goal)
    p, _ = C.resample_path(scene, cells, cfg["resample"])
    pk = p[np.unique(np.concatenate([[0], np.round(np.linspace(1, len(p) - 2, 16)).astype(int),
                                     [len(p) - 1]]))]
    cons = C.build_route_constraints(pk, scene, cfg)
    cache = C._FieldCache(scene, pk)
    cons_list, m = C.make_constraints(scene, pk, cons, cfg["centerline"], cache=cache)
    rng = np.random.default_rng(5)
    x = rng.normal(0, 0.1, size=2 * m)
    h = 1e-6
    worst = 0.0
    for c in cons_list:
        J = np.asarray(c["jac"](x))
        for direction in random_dirs(rng, 3):
            dx = np.repeat(direction[None, :], m, axis=0).ravel()
            fd = (np.asarray(c["fun"](x + h * dx)) - np.asarray(c["fun"](x - h * dx))) / (2 * h)
            an = J @ dx
            denom = max(1.0, float(np.max(np.abs(fd))))
            worst = max(worst, float(np.max(np.abs(fd - an))) / denom)
    check("constraint Jacobians == finite differences (random directions)",
          worst < 1e-4, f"worst relative error {worst:.3e}")


def test_out_of_bounds(scene, cfg):
    """4. points outside the scene must never be judged free / safe."""
    lo, hi = 0.0, scene.size * scene.r_scene
    pts = np.array([[lo - 0.5, lo + 3.0], [hi + 0.5, hi - 3.0],
                    [lo + 3.0, lo - 0.5], [hi - 3.0, hi + 0.5], [-5.0, -5.0]])
    d = scene.distance(pts)
    free = scene.is_free(pts)
    safe = C.centreline_free(scene, pts, 0.30)
    check("out-of-field points are not free and not safe",
          bool(np.all(d < 0) and not np.any(free) and not np.any(safe)),
          f"distances {np.round(d, 3).tolist()}")


def test_d_safe_consistency(scene, cfg):
    """5. node, segment and homotopy checks all use the configured d_safe."""
    d_safe = (float(cfg["robot"]["radius_m"]) + float(cfg["robot"]["safety_margin_m"])
              + float(cfg["centerline"].get("numeric_margin_m", 0.01)))
    lo, hi = 2.0, scene.size * scene.r_scene - 2.0
    rng = np.random.default_rng(11)
    pts = rng.uniform(lo, hi, size=(6000, 2))
    d = scene.distance(pts)
    # node and segment tests must agree with the threshold, point by point
    node = C.centreline_free(scene, pts, d_safe)
    agree_node = bool(np.all(node == (d >= d_safe - 1e-9)))
    # a degenerate segment (a == b) must reproduce the node verdict at the same d_safe
    seg_ok = True
    for pt in pts[:200]:
        same = C.segment_path_free(scene, np.vstack([pt, pt + 1e-9]), 0.05, d_safe=d_safe)
        if bool(same) != bool(C.centreline_free(scene, pt[None, :], d_safe)[0]):
            seg_ok = False
            break
    # the map threshold must be strictly weaker than d_safe
    weak = scene.free_distance_m < d_safe - 1e-12
    check("node / segment / map thresholds are consistent and d_safe is stricter",
          agree_node and seg_ok and weak,
          f"d_safe {d_safe:.3f} m, map threshold {scene.free_distance_m:.3f} m, "
          f"min D {d.min():.3f} m, node==(D>=d_safe) {agree_node}, segment agrees {seg_ok}")


def test_hard_constraint_gate(scene, cfg):
    """6. any single hard failure must block acceptance."""
    _, start, goal = build_layout("wall_one_side")
    path = C.build_path(scene, start, goal, cfg)
    p, q = path.p_xy, path.q_xy
    base = C.final_acceptance(scene, p, q, cfg)
    broken = []
    q_off = q.copy(); q_off[:, 0] += 10.0                        # offset out of the domain
    broken.append(("offset", C.final_acceptance(scene, p, q_off, cfg)))
    q_end = q.copy(); q_end[0] = q_end[0] + np.array([1.0, 1.0])
    broken.append(("endpoints", C.final_acceptance(scene, p, q_end, cfg)))
    q_swap = q.copy(); q_swap[[len(q) // 3, len(q) // 3 + 4]] = q_swap[[len(q) // 3 + 4, len(q) // 3]]
    broken.append(("station_order", C.final_acceptance(scene, p, q_swap, cfg)))
    cfg_tight = json.loads(json.dumps(cfg))
    cfg_tight["centerline"]["r_max_m"] = 0.01                    # forces tangent/offset failures
    broken.append(("offset_tight", C.final_acceptance(scene, p, q, cfg_tight)))
    ok = bool(base["accepted"]) and all(not b["accepted"] for _, b in broken)
    detail = ", ".join(f"{n}:{'rejected' if not b['accepted'] else 'ACCEPTED'}"
                       for n, b in broken)
    check("hard-constraint gate rejects every broken path", ok, detail)


def test_station_order(scene, cfg):
    """7. station order detects a swapped station, and tolerates a clean path."""
    _, start, goal = build_layout("u_loop")
    path = C.build_path(scene, start, goal, cfg)
    p, q = path.p_xy, path.q_xy
    good = C.validate_centerline(scene, p, q, cfg)["station_order_ok"]
    q_bad = q.copy()
    i, j = len(q) // 3, len(q) // 3 + 6
    q_bad[[i, j]] = q_bad[[j, i]]
    bad = C.validate_centerline(scene, p, q_bad, cfg)["station_order_ok"]
    check("station order: clean path passes, swapped stations fail", bool(good) and not bad,
          f"clean={good}, swapped={bad}")


def test_solver_status(scene, cfg):
    """8. solver convergence and geometric safety are recorded independently."""
    # (a) the classifier itself: no combination may label an unsafe path optimal
    mapping = {
        (True, True, ""): "optimal_safe",
        (False, True, ""): "feasible_not_converged",
        (True, False, ""): "infeasible",
        (False, False, ""): "infeasible",
        (True, True, "astar_fallback"): "fallback_astar",
    }
    mapping_ok = all(C.classify_label_status(*k) == v for k, v in mapping.items())
    # (b) integration: 1 iteration must not claim optimality, and the recorded
    # solver flag must match the classifier for the observed geometry
    _, start, goal = build_layout("wall_one_side")
    cfg_fast = json.loads(json.dumps(cfg))
    cfg_fast["centerline"]["_iter_override"] = 1
    path_fast = C.build_path(scene, start, goal, cfg_fast)
    repaired = "astar_fallback" if path_fast.label_status == "fallback_astar" else ""
    expected = C.classify_label_status(path_fast.solver_success, path_fast.accepted, repaired)
    if not path_fast.accepted:
        expected = "failed"
    ok = (mapping_ok and path_fast.label_status != "optimal_safe"
          and path_fast.label_status == expected and not path_fast.solver_success)
    check("solver status separated from geometric status", ok,
          f"classifier {'ok' if mapping_ok else 'BROKEN'}; 1-iter run: "
          f"success={path_fast.solver_success} accepted={path_fast.accepted} "
          f"status={path_fast.label_status} (expected {expected})")


def test_homotopy_density(scene, cfg):
    """9. a fine lambda sweep must catch what a coarse sweep can miss."""
    _, start, goal = build_layout("wall_one_side")
    path = C.build_path(scene, start, goal, cfg)
    p = path.p_xy
    q_shallow = p.copy()
    q_shallow[:, 0] += 0.55                     # small move that grazes an obstacle band
    coarse, _ = C.homotopy_max_lambda(scene, p, q_shallow, 0.25, 0.25, d_safe=0.30)
    fine, _ = C.homotopy_max_lambda(scene, p, q_shallow, 0.05, 0.02, d_safe=0.30)
    ok = fine <= coarse + 1e-9
    check("homotopy: fine sweep is never more permissive than the coarse sweep", ok,
          f"coarse lambda_max {coarse:.3f}, fine lambda_max {fine:.3f}")


def test_astar_fallback(scene, cfg):
    """10. the A* fallback path is accepted ONLY if it passes the same gate."""
    _, start, goal = build_layout("narrow_corridor")
    cfg_imp = json.loads(json.dumps(cfg))
    cfg_imp["centerline"]["d_safe_m"] = 99.0        # impossible clearance -> must fail
    cfg_imp["centerline"]["numeric_margin_m"] = 98.99
    path = C.build_path(scene, start, goal, cfg_imp)
    ok = (not path.accepted) and path.label_status == "failed"
    check("infeasible clearance -> fallback is rejected too (status failed)",
          ok, f"accepted={path.accepted}, label_status={path.label_status}, "
              f"failures={path.failures}")


def main() -> int:
    cfg = load_cfg(os.path.join(PROJECT_ROOT, "configs", "data_generation_v2_1.yaml"))
    occ, start, goal = build_layout("wall_one_side")
    scene = make_scene(occ, cfg, "test_wall")
    rng = np.random.default_rng(0)

    tests = [
        ("distance_gradient", lambda: test_distance_gradient(scene, cfg, rng)),
        ("objective_gradient", lambda: test_objective_gradient(scene, cfg, rng)),
        ("constraint_jacobians", lambda: test_constraint_jacobians(scene, cfg)),
        ("out_of_bounds", lambda: test_out_of_bounds(scene, cfg)),
        ("d_safe_consistency", lambda: test_d_safe_consistency(scene, cfg)),
        ("hard_constraint_gate", lambda: test_hard_constraint_gate(scene, cfg)),
        ("station_order", lambda: test_station_order(scene, cfg)),
        ("solver_status", lambda: test_solver_status(scene, cfg)),
        ("homotopy_density", lambda: test_homotopy_density(scene, cfg)),
        ("astar_fallback", lambda: test_astar_fallback(scene, cfg)),
    ]
    for name, fn in tests:
        try:
            fn()
        except Exception:
            check(name, False, "raised:\n" + traceback.format_exc(limit=3))

    n_pass = sum(1 for r in RESULTS if r["pass"])
    print(f"\n{n_pass}/{len(RESULTS)} tests passed")
    out = os.path.join(PROJECT_ROOT, "reports_v2_1_followup", "stage1_tests.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"results": RESULTS, "n_pass": n_pass, "n_total": len(RESULTS)}, fh, indent=2)
    print(f"-> {out}")
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
