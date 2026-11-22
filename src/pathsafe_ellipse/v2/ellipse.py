"""V2 ellipse labels: fixed centre, 2x2 max-det with linear half-space constraints.

Math (per spec section 4)
------------------------
Ellipse ``E_i = {x : (x - q_i)^T S_i^{-1} (x - q_i) <= 1}`` with ``S_i ≻ 0`` (m²),
area ``pi sqrt(det S_i)``.  For a unit direction ``n`` the square ``C_j``
(centre ``o_j``, side ``r_scene``) has minimum projection

    b_j(n) = n·o_j - (r_scene/2)(|n_x| + |n_y|)

and the ellipse support is ``h_E(n) = n·q_i + sqrt(n^T S_i n)``.  Requiring
``h_E(n) <= b_j(n)`` and squaring gives a **linear** constraint in ``S_i``

    n^T S_i n <= d_j^2,    d_j = b_j(n) - n·q_i

so the fixed-centre problem is a small max-det SDP in three unknowns::

    maximise  log det S   s.t.  n_j^T S n_j <= d_j^2,   eps I <= S <= R_E^2 I

Safety is decided afterwards by the exact geometric test (analytic minimum of the
ellipse quadratic over every obstacle square in the bounding box).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Exact geometry in metres
# --------------------------------------------------------------------------- #
def box_quadratic_min(M: np.ndarray, q: np.ndarray, x0, x1, y0, y1) -> np.ndarray:
    """Exact min of ``(x-q)^T M (x-q)`` over axis-aligned boxes (vectorised)."""
    m11, m12, m22 = M[0, 0], M[0, 1], M[1, 1]
    cx, cy = q[0], q[1]
    best = np.full(np.broadcast(x0, x1, y0, y1).shape, np.inf)

    def evaluate(x, y):
        dx = x - cx
        dy = y - cy
        return m11 * dx * dx + 2.0 * m12 * dx * dy + m22 * dy * dy

    inside = (cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1)
    best = np.where(inside, 0.0, best)
    for xf in (x0, x1):
        y_star = cy - (m12 * (xf - cx)) / m22 if m22 > 0 else cy
        best = np.minimum(best, evaluate(xf, np.clip(y_star, y0, y1)))
        best = np.minimum(best, evaluate(xf, y0))
        best = np.minimum(best, evaluate(xf, y1))
    for yf in (y0, y1):
        x_star = cx - (m12 * (yf - cy)) / m11 if m11 > 0 else cx
        best = np.minimum(best, evaluate(np.clip(x_star, x0, x1), yf))
        best = np.minimum(best, evaluate(x0, yf))
        best = np.minimum(best, evaluate(x1, yf))
    return best


def ellipse_bbox_cells(occupancy: np.ndarray, r_scene: float, q: np.ndarray,
                       radii: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Occupied cell centres (metres) inside the ellipse bounding box."""
    n = occupancy.shape[0]
    c0 = int(np.floor((q[0] - radii[0]) / r_scene)) - 1
    c1 = int(np.ceil((q[0] + radii[0]) / r_scene)) + 1
    r0 = int(np.floor((q[1] - radii[1]) / r_scene)) - 1
    r1 = int(np.ceil((q[1] + radii[1]) / r_scene)) + 1
    c0 = max(c0, 0); r0 = max(r0, 0)
    c1 = min(c1, n - 1); r1 = min(r1, n - 1)
    if c1 < c0 or r1 < r0:
        return np.empty((0, 2)), np.empty(0, dtype=int)
    sub = occupancy[r0:r1 + 1, c0:c1 + 1]
    rr, cc = np.nonzero(sub)
    if rr.size == 0:
        return np.empty((0, 2)), np.empty(0, dtype=int)
    centres = np.column_stack([(cc + c0) * r_scene, (rr + r0) * r_scene])
    flat_idx = (rr + r0) * n + (cc + c0)
    return centres, flat_idx


def exact_clearance(S: np.ndarray, q: np.ndarray, occupancy: np.ndarray,
                    r_scene: float, eps: float = 0.0) -> Tuple[bool, float, np.ndarray, np.ndarray]:
    """Exact test: ``(is_clear, min_quadratic, violating_centres, violating_idx)``.

    ``min_quadratic > 1`` means the ellipse does not intersect any occupied
    square inside its bounding box (continuous analytic test, not sampling).
    """
    S = (np.asarray(S, float) + np.asarray(S, float).T) / 2.0
    eig = np.linalg.eigvalsh(S)
    if eig.min() <= 0:
        return False, -np.inf, np.empty((0, 2)), np.empty(0, dtype=int)
    radii = np.sqrt(np.maximum(np.diag(S), 0.0))   # exact bounding half-extents
    centres, idx = ellipse_bbox_cells(occupancy, r_scene, q, radii)
    if centres.size == 0:
        return True, np.inf, np.empty((0, 2)), np.empty(0, dtype=int)
    M = np.linalg.inv(S)
    h = 0.5 * r_scene
    vals = box_quadratic_min(M, q, centres[:, 0] - h, centres[:, 0] + h,
                             centres[:, 1] - h, centres[:, 1] + h)
    worst = int(np.argmin(vals))
    min_val = float(vals[worst])
    bad = vals <= (1.0 + eps)
    return (not np.any(bad)), min_val, centres[bad], idx[bad]


def support(n: np.ndarray, S: np.ndarray) -> float:
    """Ellipse support offset ``sqrt(n^T S n)``."""
    return float(np.sqrt(max(n @ S @ n, 0.0)))


# --------------------------------------------------------------------------- #
# Separation half-spaces (current-ellipse driven, section 4.5)
# --------------------------------------------------------------------------- #
@dataclass
class HalfSpace:
    n: np.ndarray      # unit normal
    d: float           # distance from q to the obstacle support plane (metres)
    d2: float          # d^2, the bound used by the SDP
    cell_xy: np.ndarray
    gap: float         # separation gap at selection time


def separation_direction(S_ref: np.ndarray, q: np.ndarray, o: np.ndarray,
                         r_scene: float, n_candidates: int) -> Tuple[Optional[np.ndarray], float]:
    """Best unit direction maximising the separation gap of one obstacle square."""
    base = np.arctan2(o[1] - q[1], o[0] - q[0])
    spread = np.deg2rad(60.0)
    angles = base + np.linspace(-spread, spread, max(3, n_candidates))
    best_n, best_gap = None, -np.inf
    half = 0.5 * r_scene
    for a in angles:
        n = np.array([np.cos(a), np.sin(a)])
        b = float(n @ o) - half * (abs(n[0]) + abs(n[1]))
        h_e = float(n @ q) + support(n, S_ref)
        gap = b - h_e
        if gap > best_gap:
            best_gap, best_n = gap, n
    return best_n, best_gap


def build_halfspaces(S_ref: np.ndarray, q: np.ndarray, boundary_cells: np.ndarray,
                     r_scene: float, eps_m: float, sectors: int, n_candidates: int,
                     active_target: int, hard_limit: int) -> List[HalfSpace]:
    """Nearest boundary cell per angular sector -> active separation half-spaces."""
    if boundary_cells.size == 0:
        return []
    rel = boundary_cells - q[None, :]
    dist = np.linalg.norm(rel, axis=1)
    ang = np.arctan2(rel[:, 1], rel[:, 0])
    bins = np.clip(np.floor((ang + np.pi) / (2 * np.pi) * sectors).astype(int), 0, sectors - 1)
    order = np.lexsort((dist, bins))
    _, first = np.unique(bins[order], return_index=True)
    picked = order[first][:active_target]

    half_systems: List[HalfSpace] = []
    for idx in picked:
        o = boundary_cells[idx]
        n, gap = separation_direction(S_ref, q, o, r_scene, n_candidates)
        if n is None:
            continue
        b = float(n @ o) - 0.5 * r_scene * (abs(n[0]) + abs(n[1]))
        d = b - float(n @ q) - eps_m
        if d <= 1e-6:
            continue
        half_systems.append(HalfSpace(n=n, d=d, d2=d * d, cell_xy=o, gap=gap))
        if len(half_systems) >= hard_limit:
            break
    return half_systems


# --------------------------------------------------------------------------- #
# 2x2 max-det: production barrier solver
# --------------------------------------------------------------------------- #
def _pack(S: np.ndarray) -> np.ndarray:
    return np.array([S[0, 0], S[0, 1], S[1, 1]], dtype=float)


def _unpack(v: np.ndarray) -> np.ndarray:
    return np.array([[v[0], v[1]], [v[1], v[2]]], dtype=float)


def _det_min(v: np.ndarray) -> Tuple[float, float]:
    sxx, sxy, syy = v
    det = sxx * syy - sxy * sxy
    lmax = 0.5 * (sxx + syy) + np.sqrt(max(0.25 * (sxx - syy) ** 2 + sxy * sxy, 0.0))
    lmin = 0.5 * (sxx + syy) - np.sqrt(max(0.25 * (sxx - syy) ** 2 + sxy * sxy, 0.0))
    return det, (lmin, lmax)


def solve_s_barrier(normals: np.ndarray, bounds2: np.ndarray, min_eig: float,
                    r_max2: float, mus=(1e-1, 1e-2, 1e-3, 1e-5, 1e-7),
                    max_newton: int = 60,
                    init_S: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], str]:
    """max log det S s.t. n_j^T S n_j <= b_j, min_eig I <= S <= r_max2 I.

    The eigenvalue window is enforced *exactly* with log-barrier terms on
    ``lambda_min - min_eig`` and ``r_max2 - lambda_max`` (an 8-direction bound
    would let a rotated ellipse exceed ``R_E`` by up to ~8%).
    """
    normals = np.asarray(normals, dtype=float).reshape(-1, 2)
    bounds2 = np.asarray(bounds2, dtype=float).reshape(-1)
    A = np.stack([normals[:, 0] ** 2, 2 * normals[:, 0] * normals[:, 1],
                  normals[:, 1] ** 2], axis=1)          # (m,3)
    b = bounds2
    H_D = np.array([[0.0, 0.0, 1.0], [0.0, -2.0, 0.0], [1.0, 0.0, 0.0]])

    def eig_info(v):
        S = _unpack(v)
        w, V = np.linalg.eigh(S)
        vmin, vmax = V[:, 0], V[:, 1]
        d_lmin = np.array([vmin[0] ** 2, 2 * vmin[0] * vmin[1], vmin[1] ** 2])
        d_lmax = np.array([vmax[0] ** 2, 2 * vmax[0] * vmax[1], vmax[1] ** 2])
        return float(w[0]), float(w[1]), d_lmin, d_lmax

    def feasible(v):
        det, (lmin, lmax) = _det_min(v)          # closed form, no eigendecomposition
        if det <= 1e-18 or lmin <= min_eig or lmax >= r_max2:
            return False
        return bool(np.all(b - A @ v > 1e-12))

    def objective(v):
        det, (lmin, lmax) = _det_min(v)          # closed form: the line search is hot
        if det <= 0:
            return np.inf
        g = b - A @ v
        h_pd = lmin - min_eig
        h_ub = r_max2 - lmax
        if h_pd <= 0 or h_ub <= 0 or np.any(g <= 0):
            return np.inf
        return (-np.log(det) - mu * float(np.sum(np.log(g)))
                - mu * np.log(h_pd) - mu * np.log(h_ub))

    # feasible start: warm start from the previous/initial ellipse when possible
    v = None
    if init_S is not None:
        base = _pack(np.asarray(init_S, dtype=float))
        for shrink in (1.0, 0.9, 0.75, 0.6, 0.45, 0.3, 0.2, 0.1, 0.05, 0.01):
            trial = base * shrink
            if feasible(trial):
                v = trial
                break
    if v is None:
        start = max(min(min_eig * 10.0, 1e-3), 1e-9)
        v = np.array([start, 0.0, start])
        while not feasible(v) and v[0] > 1e-15:
            v = v * 0.5
        if not feasible(v):
            return None, "no_feasible_start"

    for mu in mus:
        for _ in range(max_newton):
            det, (lmin, lmax) = _det_min(v)
            if det <= 1e-18:
                return None, "lost_positivity"
            g = b - A @ v
            if np.any(g <= 0):
                return None, "lost_feasibility"
            grad_D = np.array([v[2], -2.0 * v[1], v[0]])
            grad = -grad_D / det + mu * np.sum(A / g[:, None], axis=0)
            hess = (np.outer(grad_D, grad_D) / (det * det) - H_D / det
                    + mu * (A / g[:, None]).T @ (A / g[:, None]))
            # exact eigenvalue-window barriers
            lmin_e, lmax_e, d_lmin, d_lmax = eig_info(v)
            h_pd = lmin_e - min_eig
            h_ub = r_max2 - lmax_e
            if h_pd <= 0 or h_ub <= 0:
                return None, "eigen_window"
            grad = grad + mu * d_lmin / h_pd - mu * d_lmax / h_ub
            hess = hess + mu * np.outer(d_lmin, d_lmin) / (h_pd * h_pd) \
                + mu * np.outer(d_lmax, d_lmax) / (h_ub * h_ub)
            scale = float(np.trace(np.abs(hess))) / 3.0
            hess = hess + max(1e-16, 1e-12 * scale) * np.eye(3)
            try:
                step = -np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                step = -np.linalg.lstsq(hess, grad, rcond=None)[0]
            if not np.all(np.isfinite(step)):
                return None, "nonfinite_step"
            if np.linalg.norm(step) < 1e-9 * (1.0 + np.linalg.norm(v)):
                break                    # Newton decrement exhausted for this mu
            f_cur = objective(v)
            slope = float(grad @ step)
            t = 1.0
            ok = False
            for _ls in range(30):
                v_new = v + t * step
                f_new = objective(v_new)
                if np.isfinite(f_new) and f_new <= f_cur + 1e-4 * t * slope:
                    ok = True
                    break
                t *= 0.5
            if not ok:
                break
            v = v_new
            if np.linalg.norm(t * step) < 1e-10 * (1.0 + np.linalg.norm(v)):
                break

    S = _unpack(v)
    if not np.all(np.isfinite(S)):
        return None, "nonfinite_result"
    det, (lmin, lmax) = _det_min(v)
    if det <= 0 or lmin <= 0:
        return None, "not_positive_definite"
    return S, "barrier_optimal"


def solve_s_cvxpy(normals: np.ndarray, bounds2: np.ndarray, min_eig: float,
                  r_max2: float, solver: str = "CLARABEL"):
    """Reference solver for the same program (used to validate the model)."""
    import cvxpy as cp

    S = cp.Variable((2, 2), symmetric=True)
    cons = [S - min_eig * np.eye(2) >> 0, r_max2 * np.eye(2) - S >> 0]
    for n, b in zip(normals, bounds2):
        cons.append(cp.quad_form(n, S) <= b)
    prob = cp.Problem(cp.Maximize(cp.log_det(S)), cons)
    try:
        prob.solve(solver=solver)
    except Exception:
        try:
            prob.solve(solver="SCS")
        except Exception:
            return None, "solver_error"
    if S.value is None:
        return None, prob.status
    return (np.asarray(S.value) + np.asarray(S.value).T) / 2.0, prob.status


# --------------------------------------------------------------------------- #
# One anchor label
# --------------------------------------------------------------------------- #
@dataclass
class EllipseLabel:
    q: np.ndarray
    S: np.ndarray
    ab: np.ndarray            # semi-axes (m), a >= b
    angle: float              # radians
    area: float               # m^2
    clearance: float          # D(q) in m
    status: str               # optimal | fallback_circle | failed
    safety_verified: bool
    min_quadratic: float
    rounds: int
    active: int
    solve_ms: float
    halfspaces: List[HalfSpace]


def _axes(S: np.ndarray) -> Tuple[np.ndarray, float, float]:
    vals, vecs = np.linalg.eigh(S)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order], 0.0)
    v = vecs[:, order[0]]
    angle = float(np.arctan2(v[1], v[0]))
    area = float(np.pi * np.sqrt(max(np.linalg.det(S), 0.0)))
    return np.sqrt(vals), angle, area


def make_label(scene, q: np.ndarray, boundary_cells: np.ndarray, cfg) -> EllipseLabel:
    """Fixed-centre maximal ellipse at ``q`` with bounded active-set iteration."""
    import time

    t0 = time.perf_counter()
    ec = cfg["ellipse"]
    r_scene = scene.r_scene
    clearance = float(scene.distance(np.asarray(q, dtype=float)))
    r_init = min(float(ec["r_init_max_m"]), 0.5 * clearance)
    min_eig = float(ec["min_eig_m2"])
    r_max2 = float(ec["r_axis_max_m"]) ** 2

    if r_init <= 0:
        return EllipseLabel(q=q, S=np.eye(2) * min_eig, ab=np.array([0.0, 0.0]), angle=0.0,
                            area=0.0, clearance=clearance, status="failed",
                            safety_verified=False, min_quadratic=-np.inf,
                            rounds=0, active=0, solve_ms=0.0, halfspaces=[])

    # 1) safe initial circle, shrunk until the exact test passes
    S_ref = np.eye(2) * (r_init ** 2)
    for _ in range(20):
        ok, _, _, _ = exact_clearance(S_ref, q, scene.blocked.astype(np.uint8), r_scene)
        if ok:
            break
        S_ref = S_ref * 0.7
    else:
        return EllipseLabel(q=q, S=S_ref, ab=_axes(S_ref)[0], angle=_axes(S_ref)[1],
                            area=_axes(S_ref)[2], clearance=clearance,
                            status="failed", safety_verified=False,
                            min_quadratic=-np.inf, rounds=0, active=0,
                            solve_ms=(time.perf_counter() - t0) * 1e3, halfspaces=[])
    initial_safe = S_ref.copy()
    # 4 map-border half-spaces (spec 4.2): an ellipse may not leave the scene
    field_m = scene.size * r_scene
    used: List[HalfSpace] = []
    for n in (np.array([1.0, 0.0]), np.array([-1.0, 0.0]),
              np.array([0.0, 1.0]), np.array([0.0, -1.0])):
        border = field_m if (n[0] > 0 or n[1] > 0) else 0.0
        d = border - float(n @ q) - float(ec["epsilon_m"])
        if d > 1e-6:
            used.append(HalfSpace(n=n, d=d, d2=d * d,
                                  cell_xy=np.array([np.nan, np.nan]), gap=np.inf))
    S_best = S_ref.copy()
    area_prev = _axes(S_ref)[2]
    rounds = 0
    status = "fallback_circle"

    for rnd in range(int(ec["max_outer_rounds"])):
        hs = build_halfspaces(S_ref, q, boundary_cells, r_scene,
                              float(ec["epsilon_m"]), int(ec["sectors"]),
                              int(ec["direction_candidates"]), int(ec["active_target"]),
                              int(ec["active_hard_limit"]))
        if not hs:
            break
        # accumulate (never drop a previously added constraint)
        for h in hs:
            if not any(abs(h.n @ u.n - 1.0) < 1e-9 and abs(h.d - u.d) < 1e-9 for u in used):
                used.append(h)
        normals = np.stack([h.n for h in used])
        bounds2 = np.array([h.d2 for h in used])
        S_new, st = solve_s_barrier(normals, bounds2, min_eig, r_max2)
        rounds += 1
        if S_new is None:
            status = f"solver_failed:{st}"
            break
        ok, min_q, bad_xy, _ = exact_clearance(S_new, q, scene.blocked.astype(np.uint8), r_scene)
        if ok:
            area_new = _axes(S_new)[2]
            S_best, status = S_new, "optimal"
            if area_prev > 1e-12 and abs(area_new - area_prev) / area_prev < float(ec["area_change_tol"]):
                break
            area_prev = area_new
            S_ref = S_new          # verified safe: becomes the new reference ellipse
            continue
        # verification found obstacles outside the active set: build the new
        # constraints from the LAST VERIFIED SAFE ellipse, per spec 4.7 step 6
        extra = []
        for xy in bad_xy[: int(ec["active_hard_limit"])]:
            n, _gap = separation_direction(S_ref, q, xy, r_scene, int(ec["direction_candidates"]))
            if n is None:
                continue
            b = float(n @ xy) - 0.5 * r_scene * (abs(n[0]) + abs(n[1]))
            d = b - float(n @ q) - float(ec["epsilon_m"])
            if d > 1e-6:
                extra.append(HalfSpace(n=n, d=d, d2=d * d, cell_xy=xy, gap=_gap))
        if not extra:
            status = "verification_failed_no_constraint"
            break
        used.extend(extra)
        normals = np.stack([h.n for h in used])
        bounds2 = np.array([h.d2 for h in used])
        S_new, st = solve_s_barrier(normals, bounds2, min_eig, r_max2)
        rounds += 1
        if S_new is None:
            status = f"solver_failed:{st}"
            break
        ok, min_q, _, _ = exact_clearance(S_new, q, scene.blocked.astype(np.uint8), r_scene)
        S_best = S_new
        status = "optimal" if ok else "unverified"
        if ok:
            S_ref = S_new

    ok_final, min_q_final, _, _ = exact_clearance(S_best, q, scene.blocked.astype(np.uint8), r_scene)
    if not ok_final:
        S_best = initial_safe
        ok_final, min_q_final, _, _ = exact_clearance(S_best, q, scene.blocked.astype(np.uint8), r_scene)
        status = "fallback_circle"
    elif status == "fallback_circle":
        status = "initial_circle_kept"
    ab, angle, area = _axes(S_best)
    return EllipseLabel(q=np.asarray(q, dtype=float), S=S_best, ab=ab, angle=angle,
                        area=area, clearance=clearance, status=status,
                        safety_verified=bool(ok_final), min_quadratic=float(min_q_final),
                        rounds=rounds, active=len(used),
                        solve_ms=(time.perf_counter() - t0) * 1e3, halfspaces=used)


def midpoint_metric(S: np.ndarray, q: np.ndarray, m: np.ndarray) -> float:
    """``rho = sqrt((m-q)^T S^{-1} (m-q))`` for the continuity check."""
    d = np.asarray(m, float) - np.asarray(q, float)
    return float(np.sqrt(max(d @ np.linalg.inv(S) @ d, 0.0)))
