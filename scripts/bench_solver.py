"""Dev benchmark: per-candidate solver cost and acceptance behaviour."""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pathsafe_ellipse.candidate_centers import generate_candidates
from pathsafe_ellipse.config import load_config, project_paths, setup_logging
from pathsafe_ellipse.ellipse_solver import FixedCenterEllipseSolver, ellipse_axes, obstacle_points
from pathsafe_ellipse.ellipse_validation import evaluate_ellipse
from pathsafe_ellipse.local_features import build_patch_features
from pathsafe_ellipse.map_io import build_map_catalog
from pathsafe_ellipse.map_preprocess import preprocess_map


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-points", type=int, default=12)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg, "bench")
    pp = project_paths(cfg)
    records = [r for r in build_map_catalog(pp.map_dir, cfg["map"]["resolution"]) if r.parsed_ok]
    rec = records[0]
    inflated = preprocess_map(
        rec, pp.map_dir, pp.inflated_maps_dir,
        robot_radius=cfg["robot"]["radius"],
        kernel_radius=int(cfg["robot"]["inflation_kernel"]),
        iterations=int(cfg["robot"]["inflation_iterations"]),
    )
    occ = inflated.occupancy_inflated
    clearance = inflated.clearance
    ps = int(cfg["local_features"]["patch_size"])
    rng = np.random.default_rng(0)

    solver = FixedCenterEllipseSolver(
        max_iters=cfg["ellipse_solver"]["max_iters"],
        angle_bins=cfg["ellipse_solver"]["angle_bins"],
        min_eigenvalue=cfg["ellipse_solver"]["min_eigenvalue"],
        convergence_volume_change=cfg["ellipse_solver"]["convergence_volume_change"],
        init_radius_scale=cfg["ellipse_solver"]["init_radius_scale"],
        solver_primary=cfg["ellipse_solver"]["solver_primary"],
        solver_backup=cfg["ellipse_solver"]["solver_backup"],
    )

    valid = (occ == 0) & (clearance >= 1.0)
    valid[:8, :] = valid[-8:, :] = False
    valid[:, :8] = valid[:, -8:] = False
    ys, xs = np.nonzero(valid)
    picks = rng.integers(0, xs.size, size=args.n_points)

    tot_cand = tot_solve = tot_accept = 0
    t_cand = t_solve = t_valid = 0.0
    for n, i in enumerate(picks):
        point = (int(xs[i]), int(ys[i]))
        t0 = time.perf_counter()
        feats = build_patch_features(occ, point, ps, int(cfg["local_features"]["ray_count"]),
                                     float(cfg["local_features"]["ray_max_range"]))
        obs_points = obstacle_points(feats.patch)
        cset = generate_candidates(
            feats,
            offset_ratios=cfg["candidate_centers"]["offset_ratios"],
            ray_margin=cfg["candidate_centers"]["ray_margin"],
            dedupe_distance=cfg["candidate_centers"]["dedupe_distance"],
            min_center_clearance=cfg["candidate_centers"]["min_center_clearance"],
            segment_clearance=cfg["candidate_centers"]["segment_clearance"],
            patch_size=ps,
            ray_max_range=cfg["local_features"]["ray_max_range"],
        )
        t_cand += time.perf_counter() - t0

        best = None
        for cand in cset.candidates:
            t1 = time.perf_counter()
            sol = solver.solve(feats, np.asarray(cand.center_xy, dtype=float), ps, obs_points=obs_points)
            t_solve += time.perf_counter() - t1
            tot_cand += 1
            if sol is None:
                continue
            tot_solve += 1
            t2 = time.perf_counter()
            report, reasons = evaluate_ellipse(
                sol.L, sol.center, feats.patch, np.asarray(feats.local_xy, dtype=float),
                eta=cfg["label"]["point_inside_eta"],
                min_axis_cells=cfg["label"]["min_semi_axis"] / cfg["map"]["resolution"],
                max_axis_cells=cfg["label"]["max_semi_axis"] / cfg["map"]["resolution"],
            )
            t_valid += time.perf_counter() - t2
            if report.safe and report.point_metric <= cfg["label"]["point_inside_eta"] and report.n_axes_ok:
                score = sol.log_det - cfg["label"]["offset_penalty_mu"] * cand.offset_ratio ** 2
                tot_accept += 1
                if best is None or score > best[0]:
                    best = (score, cand, sol, report)
        if best is None:
            print(f"  [{n}] point={point} candidates={len(cset.candidates)} -> NO VALID LABEL")
            continue
        score, cand, sol, report = best
        a, b, sin_t, cos_t, phi = ellipse_axes(sol.L)
        print(
            f"  [{n}] point={point} cands={len(cset.candidates)} "
            f"picked(dir={cand.offset_direction},lam={cand.offset_ratio:.2f}) "
            f"a={a:.1f}b={b:.1f} cells (a={a*0.25:.2f}m) rho={report.point_metric:.3f} "
            f"min_q={report.min_q:.3f} iters={sol.iters} status={sol.status} solver={sol.solver}"
        )

    print(f"\ncandidates={tot_cand} solved={tot_solve} accepted={tot_accept}")
    print(f"candidate_time={t_cand:.3f}s  solve_time={t_solve:.3f}s ({t_solve / max(tot_cand,1)*1e3:.1f} ms/cand)  "
          f"validate_time={t_valid:.3f}s ({t_valid / max(tot_cand,1)*1e3:.2f} ms/check)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
