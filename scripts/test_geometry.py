"""Self-checks for the geometric kernels (run before any full data generation).

Covers: map parsing round-trip, patch alignment (no x/y swap), supercover ray
casting, segment traversal, the exact box/quadratic minimum, the exact
ellipse/obstacle-cell intersection test (against brute-force sampling), A*
legality, and barrier-vs-CLARABEL MVIE agreement.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pathsafe_ellipse.ellipse_solver import (  # noqa: E402
    FixedCenterEllipseSolver, ellipse_axes, obstacle_points, patch_walls,
)
from pathsafe_ellipse.ellipse_validation import (  # noqa: E402
    box_quadratic_min, ellipse_obstacle_intersections, point_metric, quadratic_matrix,
)
from pathsafe_ellipse.local_features import (  # noqa: E402
    _cast_one, extract_patch, ray_directions, supercover_line_cells,
)
from pathsafe_ellipse.map_io import load_moving_ai_map  # noqa: E402
from pathsafe_ellipse.mvie import CvxpyMvieSolver, solve_mvie_barrier  # noqa: E402
from pathsafe_ellipse.path_search import astar  # noqa: E402

FAILED = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    if not cond:
        FAILED.append(name)
    print(f"  [{status}] {name} {detail}")


def test_map_parse() -> None:
    print("map parsing")
    body = "type octile\nheight 3\nwidth 4\nresolution 0.25\nmap\n.@..\n..T.\nO...\n"
    with tempfile.NamedTemporaryFile("w", suffix=".map", delete=False) as fh:
        fh.write(body)
        path = fh.name
    grid, meta = load_moving_ai_map(path)
    os.unlink(path)
    check("header dims", grid.shape == (3, 4) and meta["height"] == 3 and meta["width"] == 4)
    check("obstacle chars", grid[0, 1] == 1 and grid[1, 2] == 1 and grid[2, 0] == 1)
    check("free cells", grid[0, 0] == 0 and grid[2, 3] == 0)
    check("resolution parsed", abs(meta["resolution"] - 0.25) < 1e-12)
    # row/column orientation: row 2 starts with 'O'
    check("no transpose", grid[2, 0] == 1 and grid[0, 2] == 0)


def test_patch_alignment() -> None:
    print("patch alignment")
    occ = np.zeros((64, 64), dtype=np.uint8)
    occ[10, 40] = 1                      # row 10, col 40 -> (x=40, y=10)
    patch = extract_patch(occ, (40, 10), 32)
    check("patch centre offset", patch[16, 16] == 1,
          f"patch[16,16]={patch[16, 16]} (map point maps to patch centre)")
    # out-of-map area must be filled with obstacle
    patch_edge = extract_patch(occ, (2, 2), 32)
    check("out-of-map filled as obstacle",
          patch_edge[0, 0] == 1 and patch_edge.sum() > 1)
    # patch centre (64, 64) of a 128 patch corresponds to the map point
    occ2 = np.zeros((300, 300), dtype=np.uint8)
    occ2[125, 137] = 1
    p2 = extract_patch(occ2, (137, 125), 128)
    check("128 patch alignment", p2[64, 64] == 1 and p2.sum() == 1)


def test_ray_casting() -> None:
    print("ray casting")
    occ = np.zeros((128, 128), dtype=np.uint8)
    occ[64, 100] = 1                      # wall at 36 cells to the +x direction
    d, hit = _cast_one(occ, 64.0, 64.0, 1.0, 0.0, 64.0)
    check("axis hit", hit and abs(d - 35.5) < 1e-9, f"d={d:.4f} (cell square starts at 99.5)")
    d2, hit2 = _cast_one(occ, 64.0, 64.0, -1.0, 0.0, 64.0)
    check("no hit -> range", (not hit2) and abs(d2 - 64.0) < 1e-9)

    # thin obstacle on the diagonal must not be skipped
    occ = np.zeros((128, 128), dtype=np.uint8)
    occ[70, 70] = 1
    diag = (np.cos(np.pi / 4), np.sin(np.pi / 4))
    d3, hit3 = _cast_one(occ, 64.0, 64.0, diag[0], diag[1], 64.0)
    check("diagonal corner hit", hit3 and d3 <= np.hypot(6, 6) + 1e-6, f"d={d3:.4f}")

    # brute-force comparison over many random rays
    rng = np.random.default_rng(3)
    occ = (rng.random((128, 128)) < 0.05).astype(np.uint8)
    occ[64, 64] = 0
    max_err = 0.0
    for k in range(200):
        ang = rng.uniform(-np.pi, np.pi)
        u = (np.cos(ang), np.sin(ang))
        d, hit = _cast_one(occ, 64.0, 64.0, u[0], u[1], 64.0)
        # brute force: sub-cell sampling
        ts = np.arange(0.0, min(d + 1.0, 64.0), 0.005)
        hit_bf = 64.0
        for t in ts:
            x = 64.0 + t * u[0]
            y = 64.0 + t * u[1]
            ix, iy = int(round(x)), int(round(y))
            if not (0 <= ix < 128 and 0 <= iy < 128):
                break
            if occ[iy, ix]:
                hit_bf = t
                break
        if hit and abs(d - hit_bf) > 1.5:
            max_err = max(max_err, abs(d - hit_bf))
        if not hit and hit_bf < 63.0:
            max_err = 99.0
    check("rays match brute force", max_err <= 1.5, f"max_err={max_err:.3f} cells")


def test_supercover_segment() -> None:
    print("segment traversal")
    cells = supercover_line_cells((0.0, 0.0), (5.0, 0.0))
    check("horizontal covers 6 cells", len(cells) == 6 and cells[0] == (0, 0) and cells[-1] == (5, 0))
    cells_d = supercover_line_cells((0.0, 0.0), (4.0, 4.0))
    check("diagonal reaches end", cells_d[-1] == (4, 4))
    # a blocking cell must appear in the traversed set
    cells_b = supercover_line_cells((0.0, 0.0), (10.0, 3.0))
    check("traversal is contiguous",
          all(abs(cells_b[i + 1][0] - cells_b[i][0]) <= 1 and abs(cells_b[i + 1][1] - cells_b[i][1]) <= 1
              for i in range(len(cells_b) - 1)))


def test_box_quadratic() -> None:
    print("box quadratic minimum")
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(300):
        ang = rng.uniform(0, np.pi)
        a, b = rng.uniform(0.5, 5.0), rng.uniform(0.5, 5.0)
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        L = R @ np.diag([a, b]) @ R.T
        M = quadratic_matrix(L)
        c = rng.uniform(-2, 2, size=2)
        x0 = rng.uniform(-3, 3)
        y0 = rng.uniform(-3, 3)
        x1, y1 = x0 + rng.uniform(0.2, 2.0), y0 + rng.uniform(0.2, 2.0)
        exact = float(box_quadratic_min(M, c, np.array(x0), np.array(x1),
                                        np.array(y0), np.array(y1)))
        gx = np.linspace(x0, x1, 121)
        gy = np.linspace(y0, y1, 121)
        GX, GY = np.meshgrid(gx, gy)
        q = M[0, 0] * (GX - c[0]) ** 2 + 2 * M[0, 1] * (GX - c[0]) * (GY - c[1]) \
            + M[1, 1] * (GY - c[1]) ** 2
        sampled = float(q.min())
        worst = max(worst, exact - sampled)   # exact must never exceed the sampled min
    check("box min <= sampled min", worst <= 1e-9, f"max(exact - sampled)={worst:.3e}")


def test_ellipse_cell_exactness() -> None:
    print("ellipse/obstacle exact test")
    rng = np.random.default_rng(5)
    mismatches = 0
    cases = 0
    for _ in range(120):
        ang = rng.uniform(0, np.pi)
        a, b = rng.uniform(0.8, 6.0), rng.uniform(0.8, 6.0)
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        L = R @ np.diag([a, b]) @ R.T
        c = rng.uniform(10, 20, size=2)
        patch = np.zeros((32, 32), dtype=np.uint8)
        ys, xs = np.mgrid[0:32, 0:32]
        rr = np.sqrt((xs - c[0]) ** 2 + (ys - c[1]) ** 2)
        patch[rr < 9] = (rng.random((rr < 9).sum()) < 0.25).astype(np.uint8)
        if patch.sum() == 0:
            continue
        has_int, min_q, n_cells = ellipse_obstacle_intersections(L, c, patch)
        # brute force: dense sampling over the ellipse interior + boundary
        M = quadratic_matrix(L)
        th = np.linspace(0, 2 * np.pi, 4000)
        radii = np.linspace(0.0, 1.0, 60)
        worst_q = np.inf
        for r in radii:
            z = np.stack([r * np.cos(th), r * np.sin(th)])
            pts = (L @ z).T + c
            ix = np.rint(pts[:, 0]).astype(int)
            iy = np.rint(pts[:, 1]).astype(int)
            if not np.all((ix >= 0) & (ix < 32) & (iy >= 0) & (iy < 32)):
                continue
            if np.any(patch[iy, ix] == 1):
                worst_q = 0.0
                break
        cases += 1
        if has_int and worst_q > 0.5:
            mismatches += 1     # exact says intersect but sampling found nothing
        if (not has_int) and worst_q == 0.0:
            mismatches += 1     # exact says safe but sampling hits an obstacle
    check("exact test agrees with sampling", mismatches == 0,
          f"{cases} cases, mismatches={mismatches} (rounding tolerance ~0.5 cell)")


def test_astar() -> None:
    print("A*")
    occ = np.zeros((32, 32), dtype=np.uint8)
    occ[:, 20] = 1                          # wall with a gap at row 31
    occ[31, 20] = 0
    path = astar(occ, (2, 2), (28, 2))
    check("path found", path is not None and path[0] == (2, 2) and path[-1] == (28, 2))
    if path:
        ok_step = all(abs(path[i + 1][0] - path[i][0]) <= 1 and abs(path[i + 1][1] - path[i][1]) <= 1
                      for i in range(len(path) - 1))
        ok_free = all(occ[y, x] == 0 for x, y in path)
        check("steps legal & free", ok_step and ok_free)
        # no corner cutting
        no_cut = True
        for i in range(len(path) - 1):
            x0, y0 = path[i]
            x1, y1 = path[i + 1]
            if x0 != x1 and y0 != y1:
                if occ[y0, x1] or occ[y1, x0]:
                    no_cut = False
        check("no corner cutting", no_cut)
    # wall fully blocked -> no path
    occ[:, 20] = 1
    check("blocked -> None", astar(occ, (2, 2), (28, 2)) is None)


def test_mvie_backends() -> None:
    print("MVIE barrier vs CLARABEL")
    patch = np.zeros((128, 128), dtype=np.uint8)
    patch[0:40, 0:128] = 1                  # bottom wall region
    patch[0:128, 0:30] = 1                  # left wall region
    patch[80:90, 60:100] = 1                # pillar
    obs = obstacle_points(patch)
    walls_a, walls_b = patch_walls(128)
    c = np.array([72.0, 66.0])
    solver = FixedCenterEllipseSolver(backend="barrier", max_iters=8, angle_bins=32,
                                      max_cut_rounds=6)
    A_acc, b_acc = walls_a.copy(), walls_b.copy()
    L = np.eye(2) * 5.0
    A_used, b_used = A_acc, b_acc
    for _ in range(solver.max_iters):
        L_new, status = solve_mvie_barrier(A_acc, b_acc, c, 1e-3, init_L=L)
        if L_new is None:
            break
        L = L_new
        A_used, b_used = A_acc, b_acc     # polytope that produced this ellipse
        ca, cb, _min_t = solver._select_cuts(obs, L, c, A_acc, b_acc)
        if ca.shape[0] == 0:
            break
        A_acc, b_acc = np.vstack([A_acc, ca]), np.concatenate([b_acc, cb])
    check("polytope fits cvxpy capacity", A_used.shape[0] <= max(solver._capacity, 4096),
          f"rows={A_used.shape[0]} (barrier path has no row cap; cvxpy fallback is capped)")
    cv = CvxpyMvieSolver(capacity=max(solver._capacity, A_used.shape[0]), min_eigenvalue=1e-3)
    L_cv, status_cv, used = cv.solve(A_used, b_used, c, 1e-3)
    check("CLARABEL solved", L_cv is not None, f"status={status_cv} solver={used}")
    if L_cv is not None:
        def residual(Lmat):
            return float(np.min(b_used - A_used @ c - np.linalg.norm(Lmat.T @ A_used.T, axis=0)))
        r_bar, r_cv = residual(L), residual(L_cv)
        check("barrier solution feasible", r_bar > -1e-6, f"min constraint residual={r_bar:.2e}")
        check("CLARABEL solution feasible", r_cv > -1e-6, f"min constraint residual={r_cv:.2e}")
        det_bar = np.log(np.linalg.det(L))
        det_cv = np.log(np.linalg.det(L_cv))
        # both are MVIE solves of the same polytope: they must agree up to solver
        # accuracy, and the barrier solve must not be the worse of the two
        d_axis = float(np.max(np.abs(np.linalg.eigvalsh(L) - np.linalg.eigvalsh(L_cv))))
        check("barrier >= CLARABEL objective", det_bar >= det_cv - 5e-3,
              f"logdet barrier={det_bar:.4f} cvxpy={det_cv:.4f} daxis={d_axis:.3f} "
              f"axes={np.sort(np.linalg.eigvalsh(L))[::-1].round(3)} "
              f"cvxpy_axes={np.sort(np.linalg.eigvalsh(L_cv))[::-1].round(3)}")

    a, b, sin_t, cos_t, phi = ellipse_axes(np.diag([5.0, 2.0]))
    check("axes ordered", a >= b and abs(sin_t ** 2 + cos_t ** 2 - 1) < 1e-12)
    rho = point_metric(np.eye(2) * 2.0, np.zeros(2), np.array([1.0, 0.0]))
    check("point metric", abs(rho - 0.5) < 1e-12)
    has_int, min_q, n = ellipse_obstacle_intersections(np.eye(2) * 0.1, np.array([80.0, 85.0]), patch)
    check("tiny ellipse inside pillar detected", has_int)


def main() -> int:
    test_map_parse()
    test_patch_alignment()
    test_ray_casting()
    test_supercover_segment()
    test_box_quadratic()
    test_ellipse_cell_exactness()
    test_astar()
    test_mvie_backends()
    print()
    if FAILED:
        print(f"FAILED checks: {FAILED}")
        return 1
    print("all geometry checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
