"""Micro-benchmark: barrier vs CLARABEL agreement, timing, acceptance reasons."""
from __future__ import annotations

import collections
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pathsafe_ellipse.candidate_centers import generate_candidates  # noqa: E402
from pathsafe_ellipse.config import load_config, project_paths  # noqa: E402
from pathsafe_ellipse.ellipse_solver import (  # noqa: E402
    FixedCenterEllipseSolver, ellipse_axes, obstacle_points, patch_walls,
)
from pathsafe_ellipse.ellipse_validation import evaluate_ellipse  # noqa: E402
from pathsafe_ellipse.local_features import build_patch_features  # noqa: E402
from pathsafe_ellipse.map_io import build_map_catalog  # noqa: E402
from pathsafe_ellipse.map_preprocess import preprocess_map  # noqa: E402
from pathsafe_ellipse.mvie import CvxpyMvieSolver, solve_mvie_barrier  # noqa: E402


def main() -> int:
    cfg = load_config()
    pp = project_paths(cfg)
    records = [r for r in build_map_catalog(pp.map_dir, cfg["map"]["resolution"]) if r.parsed_ok]
    inflated = preprocess_map(records[0], pp.map_dir, pp.inflated_maps_dir,
                              robot_radius=cfg["robot"]["radius"], kernel_radius=1, iterations=1)
    occ, clearance = inflated.occupancy_inflated, inflated.clearance
    ps = int(cfg["local_features"]["patch_size"])
    res = cfg["map"]["resolution"]
    ecfg = cfg["ellipse_solver"]
    mu = cfg["label"]["offset_penalty_mu"]
    solver_kwargs = dict(max_iters=ecfg["max_iters"], angle_bins=ecfg["angle_bins"],
                         min_eigenvalue=ecfg["min_eigenvalue"],
                         convergence_volume_change=ecfg["convergence_volume_change"],
                         init_radius_scale=ecfg["init_radius_scale"],
                         max_cut_rounds=ecfg["max_cut_rounds"])
    solver = FixedCenterEllipseSolver(**solver_kwargs, backend="barrier")

    rng = np.random.default_rng(7)
    valid = (occ == 0) & (clearance >= 1.0)
    valid[:70, :] = valid[-70:, :] = False
    valid[:, :70] = valid[:, -70:] = False
    ys, xs = np.nonzero(valid)
    picks = rng.integers(0, xs.size, size=10)

    reasons = collections.Counter()
    n_points = n_eval = n_ok = 0
    t_solve = t_feat = 0.0
    for n, i in enumerate(picks):
        point = (int(xs[i]), int(ys[i]))
        t0 = time.perf_counter()
        feats = build_patch_features(occ, point, ps, int(cfg["local_features"]["ray_count"]),
                                     float(cfg["local_features"]["ray_max_range"]))
        obs_points = obstacle_points(feats.patch)
        cset = generate_candidates(
            feats, offset_ratios=cfg["candidate_centers"]["offset_ratios"],
            ray_margin=cfg["candidate_centers"]["ray_margin"],
            dedupe_distance=cfg["candidate_centers"]["dedupe_distance"],
            min_center_clearance=cfg["candidate_centers"]["min_center_clearance"],
            segment_clearance=cfg["candidate_centers"]["segment_clearance"],
            patch_size=ps, ray_max_range=cfg["local_features"]["ray_max_range"])
        t_feat += time.perf_counter() - t0
        n_points += 1

        cands = cset.candidates
        centers = np.array([c.center_xy for c in cands])
        ub = axes_upper_bound(centers, obs_points, ps)
        order = np.argsort(-ub)
        best = None
        for oi in order:
            cand = cands[oi]
            ub_score = ub[oi] - mu * cand.offset_ratio ** 2
            if best is not None and ub_score <= best[0]:
                break
            t1 = time.perf_counter()
            sol = solver.solve(feats, np.asarray(cand.center_xy, dtype=float), ps, obs_points=obs_points)
            t_solve += time.perf_counter() - t1
            n_eval += 1
            if sol is None:
                reasons["solver_failed"] += 1
                continue
            report, rsn = evaluate_ellipse(
                sol.L, sol.center, feats.patch, np.asarray(feats.local_xy, dtype=float),
                eta=cfg["label"]["point_inside_eta"],
                min_axis_cells=cfg["label"]["min_semi_axis"] / res,
                max_axis_cells=cfg["label"]["max_semi_axis"] / res)
            for r in rsn:
                reasons[r] += 1
            if report.safe and report.point_metric <= cfg["label"]["point_inside_eta"] and report.n_axes_ok:
                n_ok += 1
                score = sol.log_det - mu * cand.offset_ratio ** 2
                if best is None or score > best[0]:
                    best = (score, cand, sol, report)
        if best is None:
            print(f"  [{n}] point={point} clr={clearance[point[1], point[0]]:.1f} "
                  f"cands={len(cands)} evaluated={len(order)} -> NO VALID LABEL")
            continue
        _, cand, sol, report = best
        a, b, _, _, phi = ellipse_axes(sol.L)
        print(f"  [{n}] point={point} clr={clearance[point[1], point[0]]:.1f} "
              f"cands={len(cands)} dir={cand.offset_direction:2d} lam={cand.offset_ratio:.2f} "
              f"a={a*res:.2f}m b={b*res:.2f}m th={np.degrees(phi):6.1f} rho={report.point_metric:.3f} "
              f"minq={report.min_quad:.3f} cuts={sol.cut_rounds} hp={sol.n_halfplanes} "
              f"it={sol.iters} st={sol.status} cv={sol.converged}")

    print(f"\npoints={n_points} evaluated={n_eval} valid={n_ok}")
    print(f"reasons: {dict(reasons)}")
    print(f"feat={t_feat:.2f}s solve={t_solve:.2f}s "
          f"({t_solve/max(n_eval,1)*1e3:.2f} ms/cand, {t_solve/max(n_points,1):.3f} s/point)")
    print(f"solver stats: {solver.stats}")

    # ---- cross-check barrier vs CLARABEL inside an identical polytope
    print("\n== barrier vs CLARABEL (same polytope) ==")
    feats = build_patch_features(occ, (int(xs[0]), int(ys[0])), ps, 16, 64.0)
    obs_points = obstacle_points(feats.patch)
    walls_a, walls_b = patch_walls(ps)
    cv = CvxpyMvieSolver(capacity=solver._capacity, min_eigenvalue=ecfg["min_eigenvalue"])
    cset = generate_candidates(feats, offset_ratios=cfg["candidate_centers"]["offset_ratios"],
                               ray_margin=cfg["candidate_centers"]["ray_margin"],
                               dedupe_distance=cfg["candidate_centers"]["dedupe_distance"],
                               min_center_clearance=cfg["candidate_centers"]["min_center_clearance"],
                               segment_clearance=cfg["candidate_centers"]["segment_clearance"],
                               patch_size=ps, ray_max_range=cfg["local_features"]["ray_max_range"])
    worst_det = worst_axis = 0.0
    n_cmp = 0
    for cand in cset.candidates[:4]:
        c = np.asarray(cand.center_xy, dtype=float)
        sub = FixedCenterEllipseSolver(**solver_kwargs, backend="barrier")
        sol = sub.solve(feats, c, ps, obs_points=obs_points)
        if sol is None:
            continue
        # rebuild the exact polytope the barrier solver ended with
        L = np.eye(2) * max(2.0, 0.5 * float(np.min(np.linalg.norm(obs_points - c, axis=1))))
        A_acc, b_acc = walls_a.copy(), walls_b.copy()
        for _ in range(sub.max_iters):
            L_new, status = solve_mvie_barrier(A_acc, b_acc, c, ecfg["min_eigenvalue"])
            if L_new is None:
                break
            L = L_new
            ca, cb = sub._select_cuts(obs_points, L, c, A_acc, b_acc)
            if ca.shape[0] == 0:
                break
            A_acc, b_acc = np.vstack([A_acc, ca]), np.concatenate([b_acc, cb])
        t0 = time.perf_counter()
        L_cv, status_cv, used = cv.solve(A_acc, b_acc, c, ecfg["min_eigenvalue"])
        t_cv = time.perf_counter() - t0
        if L_cv is None:
            print(f"  cand dir={cand.offset_direction}: cvxpy failed ({status_cv})")
            continue
        d_det = abs(np.log(np.linalg.det(L)) - np.log(np.linalg.det(L_cv)))
        d_axis = float(np.max(np.abs(np.linalg.eigvalsh(L) - np.linalg.eigvalsh(L_cv))))
        worst_det, worst_axis = max(worst_det, d_det), max(worst_axis, d_axis)
        n_cmp += 1
        print(f"  cand dir={cand.offset_direction} lam={cand.offset_ratio} hp={A_acc.shape[0]}: "
              f"barrier axes={np.sort(np.linalg.eigvalsh(L))[::-1].round(4)} "
              f"cvxpy axes={np.sort(np.linalg.eigvalsh(L_cv))[::-1].round(4)} "
              f"dlogdet={d_det:.2e} daxis={d_axis:.2e} t_cvxpy={t_cv*1e3:.0f}ms")
    print(f"compared={n_cmp} worst |dlogdet|={worst_det:.2e} worst |daxis|={worst_axis:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
