"""Diagnose why corridor candidates still fail the exact safety test."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pathsafe_ellipse.candidate_centers import generate_candidates
from pathsafe_ellipse.config import load_config, project_paths
from pathsafe_ellipse.dataset_builder import get_solver
from pathsafe_ellipse.ellipse_solver import build_cell_tree, obstacle_points
from pathsafe_ellipse.ellipse_validation import ellipse_obstacle_intersections, point_metric
from pathsafe_ellipse.local_features import build_patch_features
from pathsafe_ellipse.map_io import build_map_catalog
from pathsafe_ellipse.map_preprocess import preprocess_map


def main() -> int:
    cfg = load_config()
    pp = project_paths(cfg)
    cand = [r for r in build_map_catalog(pp.map_dir, cfg["map"]["resolution"])
            if r.parsed_ok and "Boston_1_1024_part_1_1" in r.file_name][0]
    inflated = preprocess_map(cand, pp.map_dir, pp.inflated_maps_dir,
                              robot_radius=cfg["robot"]["radius"], kernel_radius=1, iterations=1)
    occ, clr = inflated.occupancy_inflated, inflated.clearance
    # corridor cells: clearance <= 3, pick a few points along one
    narrow = (occ == 0) & (clr <= 3.0)
    ys, xs = np.nonzero(narrow)
    rng = np.random.default_rng(0)
    picks = rng.integers(0, xs.size, size=6)
    solver = get_solver(cfg)
    ps = int(cfg["local_features"]["patch_size"])
    print(f"max_iters={solver.max_iters} cut_rounds={solver.max_cut_rounds} "
          f"max_cuts/round={solver.max_cuts_per_round}")
    for i in picks:
        gx, gy = int(xs[i]), int(ys[i])
        feats = build_patch_features(occ, (gx, gy), ps, 16, 64.0)
        if feats.patch[64, 64] != 0:
            continue
        obs = obstacle_points(feats.patch)
        tree = build_cell_tree(obs)
        cands = generate_candidates(
            feats, offset_ratios=[0.25, 0.5, 0.75],
            ray_margin=cfg["candidate_centers"]["ray_margin"],
            dedupe_distance=cfg["candidate_centers"]["dedupe_distance"],
            min_center_clearance=cfg["candidate_centers"]["min_center_clearance"],
            segment_clearance=cfg["candidate_centers"]["segment_clearance"],
            patch_size=ps, ray_max_range=64.0).candidates
        for c in cands[:2]:
            sol = solver.solve(feats, np.asarray(c.center_xy, dtype=float), ps,
                               obs_points=obs, obs_tree=tree)
            if sol is None:
                print(f"  ({gx},{gy}) clr={clr[gy, gx]:.1f} dir={c.offset_direction} -> SOLVE FAILED")
                continue
            collide, min_q, n_cells = ellipse_obstacle_intersections(sol.L, sol.center, feats.patch)
            rho = point_metric(sol.L, sol.center, np.array([64.0, 64.0]))
            ax = np.sort(np.linalg.eigvalsh(sol.L))[::-1]
            print(f"  ({gx},{gy}) clr={clr[gy, gx]:.1f} dir={c.offset_direction:2d} "
                  f"lam={c.offset_ratio:.2f} it={sol.iters} rounds={sol.cut_rounds} "
                  f"conv={int(sol.converged)} hp={sol.n_halfplanes} "
                  f"axes=({ax[0]:.2f},{ax[1]:.2f}) rho={rho:.2f} "
                  f"collide={int(collide)} minq={min_q:.4f} cells={n_cells}")
        # --- manual trace of one unbiased candidate
        c0 = np.asarray(feats.local_xy, dtype=float)
        L = np.eye(2) * max(0.5 * float(np.min([1.0])), 1e-3)
        from pathsafe_ellipse.ellipse_solver import patch_walls
        A, b = patch_walls(ps)
        print(f"  --- trace for ({gx},{gy}) unbiased ---")
        sa, sb, _ = solver._select_cuts(obs, L, c0, A, b, tree, mode='seed')
        if sa.shape[0]:
            A, b = np.vstack([A, sa]), np.concatenate([b, sb])
        print(f'    seed rows={A.shape[0]}')
        for it in range(solver.max_iters):
            L_new, status, used = solver._solve_polytope(A, b, c0, init_L=L)
            if L_new is None:
                print(f"    it{it}: SOLVE FAILED status={status} rows={A.shape[0]}")
                break
            L = L_new
            ca, cb, min_t = solver._select_cuts(obs, L, c0, A, b, tree)
            axs = np.sort(np.linalg.eigvalsh(L))[::-1]
            print(f"    it{it}: rows={A.shape[0]:3d} status={status:16s} axes=({axs[0]:5.2f},{axs[1]:5.2f}) "
                  f"min_t={min_t:.3f} new_cuts={ca.shape[0]}")
            if ca.shape[0] == 0 or min_t >= 1.0 - solver.separation_margin:
                break
            A, b = np.vstack([A, ca]), np.concatenate([b, cb])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
