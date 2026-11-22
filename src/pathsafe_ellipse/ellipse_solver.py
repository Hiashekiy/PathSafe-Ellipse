"""Fixed-centre IRIS-style maximum-volume ellipse solver (section 8.3), v2.

The centre is *held fixed* to a ray-derived candidate; only the symmetric
positive definite shape matrix ``L`` is optimised, so the ray-offset supervision
semantics are preserved (Neural-IRIS' solver moves the centre; this one must
not).

``E(c, L) = {c + L z : ||z||_2 <= 1}``: the eigenvalues of ``L`` are the
semi-axes (cells), the eigenvectors the axis directions.

Redesign (v2) - why the labels used to starve in narrow corridors
-----------------------------------------------------------------
v1 approximated the free space with the *corner cloud of dilated* obstacle
cells.  Two consequences:

* the polytope was inset by roughly one cell, so every label was smaller than
  the true maximum inscribed ellipse;
* in a corridor only a few cells wide that loss, combined with an absolute
  minimum semi-axis (0.5 m = 2 cells), rejected the whole corridor.

v2 separates the ellipse from the *exact* geometry of each obstacle cell.  For
a cell square ``S_j`` and centre ``c`` let ``p_j = clamp(c, S_j)`` be its
closest point, ``d_j = ||p_j - c||`` and ``n_j`` the unit direction.  ``S_j`` is
convex and ``p_j`` is its closest point, so the supporting plane
``n_j . (x - c) = d_j`` separates ``c`` (hence any ellipse around ``c``) from the
whole square, giving the exact constraint

    ||L^T n_j||_2 <= d_j            (no inflation, tangent to the true boundary)

The IRIS loop keeps midpoint separation, now expressed through the normalised
clearance ``t_j = d_j / ||L^T n_j||`` (``t_j >= 1`` <=> ellipse clear of cell
``j``): cuts are placed at ``rho = (1 + t_j) / 2`` so the MVIE can keep growing
until it becomes tangent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .local_features import PatchFeatures
from .mvie import CvxpyMvieSolver, solve_mvie_barrier

# Patch walls in the "cell centre at integer coordinate" convention.
PATCH_WALL_NORMALS = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])


@dataclass
class EllipseSolution:
    L: np.ndarray            # (2, 2) shape matrix in cells
    center: np.ndarray       # (2,) centre in patch coordinates (cells)
    iters: int
    status: str
    solver: str
    log_det: float
    min_axis_cells: float
    max_axis_cells: float
    converged: bool
    n_halfplanes: int = 0
    cut_rounds: int = 0


# --------------------------------------------------------------------------- #
# Exact obstacle geometry
# --------------------------------------------------------------------------- #
def obstacle_points(patch: np.ndarray) -> np.ndarray:
    """Centres of every obstacle cell in the patch, as float ``(x, y)``."""
    ys, xs = np.nonzero(patch)
    if xs.size == 0:
        return np.empty((0, 2), dtype=float)
    return np.column_stack([xs, ys]).astype(float)


def build_cell_tree(cell_centres: np.ndarray):
    """KD-tree over obstacle cell centres (``None`` when there is no obstacle)."""
    if cell_centres.size == 0:
        return None
    from scipy.spatial import cKDTree

    return cKDTree(cell_centres)


def _clamp_distance(point: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Distance from ``point`` to each obstacle cell square."""
    px = np.clip(point[0], cells[:, 0] - 0.5, cells[:, 0] + 0.5)
    py = np.clip(point[1], cells[:, 1] - 0.5, cells[:, 1] + 0.5)
    return np.hypot(px - point[0], py - point[1])


def nearest_square_distance(point: np.ndarray, cell_centres: np.ndarray,
                            tree=None, k: int = 9) -> float:
    """Exact distance from ``point`` to the nearest obstacle *square*."""
    if cell_centres.size == 0:
        return float("inf")
    if tree is not None:
        _, idx = tree.query(np.asarray(point, dtype=float), k=min(k, len(cell_centres)))
        cand = cell_centres[np.atleast_1d(idx)]
    else:
        cand = cell_centres
    return float(np.min(_clamp_distance(np.asarray(point, dtype=float), cand)))


def patch_walls(patch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Half-plane description of the patch square (``A x <= b``)."""
    hi = patch_size - 0.5
    lo = -0.5
    b = np.array([hi, -lo, hi, -lo], dtype=float)
    return PATCH_WALL_NORMALS.copy(), b


def ellipse_axes(L: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Semi-axes (a >= b), canonical orientation (sin phi, cos phi) and the angle.

    Angle canonicalisation: the long-axis eigenvector is flipped so that
    ``cos(phi) >= 0`` (ties resolved on ``sin(phi) >= 0``), removing the
    ``pi`` ambiguity of the ellipse orientation.
    """
    L = (np.asarray(L, dtype=float) + np.asarray(L, dtype=float).T) / 2.0
    vals, vecs = np.linalg.eigh(L)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    a, b = float(vals[0]), float(vals[1])
    v = vecs[:, 0]
    if v[0] < 0 or (abs(v[0]) < 1e-12 and v[1] < 0):
        v = -v
    phi = float(np.arctan2(v[1], v[0]))
    return a, b, float(np.sin(phi)), float(np.cos(phi)), phi


# --------------------------------------------------------------------------- #
# Candidate pruning
# --------------------------------------------------------------------------- #
def candidate_priorities(
    centers: np.ndarray,
    cell_centres: np.ndarray,
    patch_size: int,
    path_point: np.ndarray,
    eta: float,
    tree=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-candidate ``(log det upper bound, viable)``.

    ``log det L <= 2 log r`` with ``r`` the distance from the centre to the
    nearest obstacle *square* / patch wall, and an ellipse can only contain the
    path point if ``|p - c| <= eta * r``.  Both are exact necessary conditions,
    so pruning with them never changes the argmax.
    """
    centers = np.atleast_2d(np.asarray(centers, dtype=float))
    path_point = np.asarray(path_point, dtype=float)
    if cell_centres.size == 0:
        d_obs = np.full(len(centers), (patch_size - 1.0) / 2.0)
    elif tree is not None:
        k = min(9, len(cell_centres))
        _, idx = tree.query(centers, k=k)
        idx = np.atleast_2d(idx).reshape(len(centers), -1)
        cand = cell_centres[idx]                                   # (n, k, 2)
        px = np.clip(centers[:, None, 0], cand[:, :, 0] - 0.5, cand[:, :, 0] + 0.5)
        py = np.clip(centers[:, None, 1], cand[:, :, 1] - 0.5, cand[:, :, 1] + 0.5)
        d_obs = np.min(np.hypot(px - centers[:, None, 0], py - centers[:, None, 1]), axis=1)
    else:
        d_obs = np.array([float(np.min(_clamp_distance(p, cell_centres))) for p in centers])
    d_wall = np.min(
        np.stack([
            centers[:, 0] + 0.5,
            patch_size - 0.5 - centers[:, 0],
            centers[:, 1] + 0.5,
            patch_size - 0.5 - centers[:, 1],
        ], axis=1),
        axis=1,
    )
    r = np.maximum(np.minimum(d_obs, d_wall), 1e-6)
    dist_to_point = np.linalg.norm(centers - path_point[None, :], axis=1)
    viable = dist_to_point <= eta * r
    return 2.0 * np.log(r), viable


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
class FixedCenterEllipseSolver:
    """IRIS-style solver with the ellipse centre held fixed."""

    def __init__(self, max_iters: int = 8, angle_bins: int = 32, min_eigenvalue: float = 1e-3,
                 convergence_volume_change: float = 0.05, init_radius_scale: float = 0.5,
                 solver_primary: str = "CLARABEL", solver_backup: str = "SCS",
                 max_cut_rounds: int = 6, backend: str = "barrier",
                 separator: str = "midpoint", time_budget_s: float = 5.0,
                 cut_radius_scale: float = 3.0, separation_margin: float = 1e-4,
                 max_cuts_per_round: int = 32):
        self.max_iters = int(max_iters)
        self.angle_bins = int(angle_bins)
        self.min_eigenvalue = float(min_eigenvalue)
        self.convergence_volume_change = float(convergence_volume_change)
        self.init_radius_scale = float(init_radius_scale)
        self.solver_primary = solver_primary
        self.solver_backup = solver_backup
        self.max_cut_rounds = int(max_cut_rounds)
        self.backend = backend
        self.separator = separator
        self.time_budget_s = float(time_budget_s)
        self.cut_radius_scale = float(cut_radius_scale)
        self.separation_margin = float(separation_margin)
        self.max_cuts_per_round = int(max_cuts_per_round)
        self._capacity = len(PATCH_WALL_NORMALS) + self.angle_bins * min(self.max_cut_rounds, 4)
        self._cvxpy: Optional[CvxpyMvieSolver] = None
        self.stats = {"calls": 0, "barrier": 0, "cvxpy": 0, "cvxpy_fallbacks": 0}

    # ---------------------------------------------------------------- internal
    def _cvxpy_solver(self) -> CvxpyMvieSolver:
        if self._cvxpy is None:
            self._cvxpy = CvxpyMvieSolver(
                capacity=self._capacity,
                min_eigenvalue=self.min_eigenvalue,
                solver_primary=self.solver_primary,
                solver_backup=self.solver_backup,
            )
        return self._cvxpy

    def _solve_polytope(self, A: np.ndarray, b: np.ndarray, c: np.ndarray,
                        init_L: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], str, str]:
        self.stats["calls"] += 1
        if self.backend != "barrier":
            if A.shape[0] > self._capacity:
                A, b = A[: self._capacity], b[: self._capacity]
            L, status, solver = self._cvxpy_solver().solve(A, b, c, self.min_eigenvalue)
            self.stats["cvxpy"] += 1
            return L, status, solver

        L, status = solve_mvie_barrier(A, b, c, self.min_eigenvalue, init_L=init_L)
        if L is not None and status == "barrier_optimal":
            residual = b - A @ c - np.linalg.norm(L.T @ A.T, axis=0)
            if float(np.min(residual)) > -1e-6:
                self.stats["barrier"] += 1
                return L, status, "barrier"
        # barrier failed (or stopped short of the optimum): cross-check with the
        # reference solver and keep the larger feasible ellipse.  Tight corridors
        # are exactly where the barrier's line search stalls.
        if A.shape[0] > self._capacity:
            A_c, b_c = A[: self._capacity], b[: self._capacity]
        else:
            A_c, b_c = A, b
        L_cv, status_cv, solver_cv = self._cvxpy_solver().solve(A_c, b_c, c, self.min_eigenvalue)
        self.stats["cvxpy"] += 1
        options = []
        for Lc, st, sv in ((L, status, "barrier"), (L_cv, status_cv, solver_cv)):
            if Lc is None or not np.all(np.isfinite(Lc)) or np.linalg.eigvalsh(Lc).min() <= 0:
                continue
            res = float(np.min(b_c - A_c @ c - np.linalg.norm(Lc.T @ A_c.T, axis=0)))
            if res <= -1e-6:
                continue
            options.append((float(np.log(max(np.linalg.det(Lc), 1e-300))), Lc, st, sv))
        if not options:
            self.stats["cvxpy_fallbacks"] += 1
            return None, status_cv, solver_cv
        options.sort(key=lambda t: -t[0])
        _, best, best_status, best_solver = options[0]
        if best_solver == "barrier":
            self.stats["barrier"] += 1
        else:
            self.stats["cvxpy_fallbacks"] += 1
        return best, best_status, best_solver

    @staticmethod
    def _cell_separation(cell_centres: np.ndarray, c: np.ndarray, L: np.ndarray,
                         subset=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Supporting half-planes of obstacle cells: ``n_j``, ``d_j``, ``t_j``."""
        cells = cell_centres if subset is None else cell_centres[subset]
        if cells.size == 0:
            return np.empty((0, 2)), np.empty(0), np.empty(0)
        px = np.clip(c[0], cells[:, 0] - 0.5, cells[:, 0] + 0.5)
        py = np.clip(c[1], cells[:, 1] - 0.5, cells[:, 1] + 0.5)
        dvec = np.stack([px - c[0], py - c[1]], axis=1)
        dist = np.linalg.norm(dvec, axis=1)
        keep = dist > 1e-9
        if not np.any(keep):
            return np.empty((0, 2)), np.empty(0), np.empty(0)
        dvec, dist = dvec[keep], dist[keep]
        n_hat = dvec / dist[:, None]
        scale = np.linalg.norm(n_hat @ L.T, axis=1)      # ||L^T n_j||
        keep2 = scale > 1e-12
        if not np.any(keep2):
            return np.empty((0, 2)), np.empty(0), np.empty(0)
        n_hat, dist, scale = n_hat[keep2], dist[keep2], scale[keep2]
        return n_hat, dist, dist / scale                 # t_j = d_j / ||L^T n_j||

    def _select_cuts(self, cell_centres: np.ndarray, L: np.ndarray, c: np.ndarray,
                     A_acc: np.ndarray, b_acc: np.ndarray, cell_tree=None,
                     mode: str = "violated") -> Tuple[np.ndarray, np.ndarray, float]:
        """Half-planes of the binding obstacle cells, cut at their exact support.

        Two selection modes:

        ``seed``      one cell per angular bin (nearest in the ellipse-normalised
                      metric).  Used before the first solve so the polytope is
                      bounded in every direction - without it the first MVIE
                      inflates to the whole patch and convergence takes ~10
                      rounds.
        ``violated``  every cell the current ellipse still overlaps (``t < 1``),
                      tightest first.  Driving this set to empty is the
                      convergence test.

        Both cut at ``rho = t``, i.e. exactly through the cell's closest point.
        That is a valid constraint (the centre sits strictly on the safe side),
        so the subproblem stays feasible, and the ellipse converges *tangent* to
        the obstacles instead of asymptotically approaching them as the midpoint
        separator does.
        """
        try:
            L_inv = np.linalg.inv(L)
        except np.linalg.LinAlgError:
            return np.empty((0, 2)), np.empty(0), 0.0
        radius = float(np.clip(self.cut_radius_scale * max(np.linalg.norm(L, axis=1).max(), 1e-6)
                               + 1.0, 3.0, 200.0))
        if cell_tree is not None:
            subset = np.asarray(cell_tree.query_ball_point(c, radius), dtype=int)
            if subset.size == 0:
                return np.empty((0, 2)), np.empty(0), float("inf")
        else:
            subset = None
        n_hat, dist, t = self._cell_separation(cell_centres, c, L, subset)
        if n_hat.shape[0] == 0:
            return np.empty((0, 2)), np.empty(0), float("inf")

        if mode == "seed":
            y = (n_hat * dist[:, None]) @ L_inv.T
            angles = np.arctan2(y[:, 1], y[:, 0])
            bins = np.clip(np.floor((angles + np.pi) / (2 * np.pi) * self.angle_bins).astype(int),
                           0, self.angle_bins - 1)
            order = np.lexsort((t, bins))
            _, first = np.unique(bins[order], return_index=True)
            take = order[first]
            take = take[t[take] < np.inf][: self.angle_bins]
        else:
            violated = np.flatnonzero(t < 1.0 - self.separation_margin)
            if violated.size == 0:
                return np.empty((0, 2)), np.empty(0), float(np.min(t))
            take = violated[np.argsort(t[violated])][: self.max_cuts_per_round]

        sel_t = t[take]
        Lt_n = n_hat[take] @ L.T
        scale = np.linalg.norm(Lt_n, axis=1)
        keep = scale > 1e-12
        if not np.any(keep):
            return np.empty((0, 2)), np.empty(0), float(np.min(sel_t))
        w_hat = Lt_n[keep] / scale[keep, None]           # unit normal, normalised space
        if self.separator == "midpoint":
            rho = 0.5 * (1.0 + sel_t[keep])
        else:                                            # exact supporting plane
            rho = sel_t[keep] - 1e-9
        rho = np.maximum(rho, 1e-9)
        a = w_hat @ L_inv                                # back to the physical frame
        a_norm = np.linalg.norm(a, axis=1)
        keep2 = a_norm > 1e-12
        if not np.any(keep2):
            return np.empty((0, 2)), np.empty(0), float(np.min(sel_t))
        normals = a[keep2] / a_norm[keep2, None]
        bounds = (a[keep2] @ c + rho[keep2]) / a_norm[keep2]
        return normals, bounds, float(np.min(sel_t))

    # --------------------------------------------------------------------- main
    def solve(self, features: PatchFeatures, center: np.ndarray, patch_size: int,
              obs_points: Optional[np.ndarray] = None,
              obs_tree=None) -> Optional[EllipseSolution]:
        c = np.asarray(center, dtype=float)
        if obs_points is None:
            obs_points = obstacle_points(features.patch)
        if obs_tree is None and obs_points.size:
            obs_tree = build_cell_tree(obs_points)
        walls_a, walls_b = patch_walls(patch_size)

        if obs_points.size == 0:
            half = (patch_size - 1.0) / 2.0
            L = np.eye(2) * half
            return EllipseSolution(
                L=L, center=c, iters=0, status="no_obstacles", solver="closed_form",
                log_det=2.0 * float(np.log(half)), min_axis_cells=half,
                max_axis_cells=half, converged=True, n_halfplanes=4,
            )

        wall_dist = float(min(c[0] + 0.5, patch_size - 0.5 - c[0],
                              c[1] + 0.5, patch_size - 0.5 - c[1]))
        nearest = nearest_square_distance(c, obs_points, obs_tree)
        init_r = max(min(nearest, wall_dist) * self.init_radius_scale,
                     self.min_eigenvalue * 10.0)
        L = np.eye(2) * init_r

        A_acc = walls_a.copy()
        b_acc = walls_b.copy()
        status = "init"
        solver_used = self.backend
        converged = False
        iters = 0
        cut_rounds = 0

        # 1) seed the polytope: one supporting plane per angular direction, taken
        #    from the small initial ellipse.  Without this the first MVIE inflates
        #    to the whole patch and needs ~10 halving rounds to come back down.
        seed_a, seed_b, _ = self._select_cuts(obs_points, L, c, A_acc, b_acc, obs_tree,
                                              mode="seed")
        if seed_a.shape[0]:
            A_acc = np.vstack([A_acc, seed_a])
            b_acc = np.concatenate([b_acc, seed_b])

        # 2) iterate: solve inside P, then add the exact supporting planes of the
        #    cells the ellipse still overlaps, until none remain (tangency) or the
        #    budget runs out.  Every cut is a valid constraint, so P stays
        #    feasible and the ellipse can only shrink towards the true maximum.
        for it in range(self.max_iters):
            L_new, status, solver_used = self._solve_polytope(A_acc, b_acc, c, init_L=L)
            if L_new is None:
                break
            iters = it + 1
            L = L_new
            cuts_a, cuts_b, min_t = self._select_cuts(obs_points, L, c, A_acc, b_acc,
                                                      obs_tree, mode="violated")
            if cuts_a.shape[0] == 0 or min_t >= 1.0 - self.separation_margin:
                converged = True
                break
            if cut_rounds >= self.max_cut_rounds:
                break
            A_acc = np.vstack([A_acc, cuts_a])
            b_acc = np.concatenate([b_acc, cuts_b])
            cut_rounds += 1

        eig = np.linalg.eigvalsh(L)
        if not np.all(np.isfinite(L)) or eig.min() <= 0:
            return None
        return EllipseSolution(
            L=L, center=c, iters=iters, status=str(status), solver=str(solver_used),
            log_det=float(np.log(max(np.linalg.det(L), 1e-300))),
            min_axis_cells=float(eig.min()), max_axis_cells=float(eig.max()),
            converged=converged, n_halfplanes=int(A_acc.shape[0]), cut_rounds=cut_rounds,
        )
