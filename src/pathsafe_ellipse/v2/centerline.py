"""V2 / V2.1 centreline: A* -> resampling -> route-constrained clearance optimisation.

Two methods are available and selected by ``cfg["centerline"]["method"]``:

``legacy_weighted`` (V2, kept only to reproduce old results)
    J = w_p |q-p|^2/L^2 - w_d D(q)/L + w_s |second difference|^2/L^2 with soft
    penalties and a post-hoc projection into ``|q-p| <= R_max``.

``route_constrained_clearance`` (V2.1, per the 拓扑约束净空中心线优化方案)
    The A* path defines a *route constraint domain* and a topology, not a
    geometry to match; inside that domain the centreline maximises clearance::

        min_Q  -w_D * sum_i w_i min(D(q_i), D_cap)/D_cap        (clearance gain)
              + w_s * sum_i w_i L_ref^2 |a_i|^2                 (non-uniform curvature)
              + w_p * sum_i w_i |q_i - p_i|^2 / (R_i^2 + eps)   (weak offset reg.)

    with hard constraints solved explicitly by SLSQP (no soft penalties):
    ``D(q_i) >= d_safe``, ``|q_i - p_i| <= R_i``,
    ``|t_i . (q_i - p_i)| <= eta * L_i`` and both endpoints fixed.
    ``w_i`` are arc-length weights, ``a_i`` is the second derivative in *arc
    length* (valid for the adaptive resampling).

    Candidate centre lines are additionally validated with a homotopy test
    (``P_lam = (1-lam) P + lam Q`` must stay collision free), and a failed
    deformation is repaired by bisecting the blend factor before falling back.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..path_search import astar
from .scene import Scene


@dataclass
class PathData:
    start_xy: np.ndarray
    goal_xy: np.ndarray
    astar_cells: np.ndarray
    p_xy: np.ndarray                  # A* stations (metres)
    q_xy: np.ndarray                  # optimised centreline (metres)
    tangent: np.ndarray               # A* tangents (network feature)
    clearance_p: np.ndarray
    clearance_q: np.ndarray
    status: str
    max_offset: float
    segment_ok: bool
    # --- V2.1 additions -----------------------------------------------------
    tangent_q: Optional[np.ndarray] = None
    centerline_method: str = "legacy_weighted"
    centerline_clearance_gain: float = 0.0
    centerline_length_ratio: float = 1.0
    centerline_homotopy_verified: bool = False
    centerline_status: str = ""
    centerline_solver_iterations: int = 0
    centerline_solver_time_ms: float = 0.0
    centerline_checks: Dict[str, object] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# A*
# --------------------------------------------------------------------------- #
def astar_m(scene: Scene, start_xy: np.ndarray, goal_xy: np.ndarray) -> Optional[np.ndarray]:
    """A* on the inflated whole-scene grid; returns integer cells (col,row)."""
    s = tuple(np.rint(np.asarray(start_xy) / scene.r_scene).astype(int))
    g = tuple(np.rint(np.asarray(goal_xy) / scene.r_scene).astype(int))
    if not (0 <= s[0] < scene.size and 0 <= s[1] < scene.size):
        return None
    if not (0 <= g[0] < scene.size and 0 <= g[1] < scene.size):
        return None
    path = astar(scene.blocked.astype(np.uint8), s, g, forbid_corner_cutting=True)
    if path is None:
        return None
    return np.asarray(path, dtype=np.int32)


# --------------------------------------------------------------------------- #
# Adaptive arc-length resampling (metres)
# --------------------------------------------------------------------------- #
def adaptive_spacing(scene: Scene, xy_m: np.ndarray, tangent: np.ndarray, rs: Dict) -> float:
    d = float(scene.distance(xy_m))
    if d <= float(rs["narrow_clearance_m"]):
        return float(rs["spacing_narrow_m"])
    if d >= float(rs["open_clearance_m"]):
        return float(rs["spacing_open_m"])
    return float(rs["spacing_normal_m"])


def resample_path(scene: Scene, cells: np.ndarray, rs: Dict) -> Tuple[np.ndarray, np.ndarray]:
    pts = cells.astype(float) * scene.r_scene
    if len(pts) < 2:
        return pts, np.arange(len(pts))
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s_cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s_cum[-1])

    out: List[np.ndarray] = [pts[0]]
    idx: List[int] = [0]
    s = 0.0
    while s < total - 1e-9:
        i = int(np.searchsorted(s_cum, s, side="right") - 1)
        i = max(0, min(i, len(pts) - 2))
        t = (s - s_cum[i]) / max(s_cum[i + 1] - s_cum[i], 1e-12)
        xy = pts[i] + t * (pts[i + 1] - pts[i])
        nxt = pts[min(i + 1, len(pts) - 1)] - pts[i]
        nrm = np.linalg.norm(nxt)
        tangent = nxt / nrm if nrm > 1e-9 else np.array([1.0, 0.0])
        step = adaptive_spacing(scene, xy, tangent, rs)
        s += step
        if s >= total - 1e-9:
            break
        i2 = int(np.searchsorted(s_cum, s, side="right") - 1)
        i2 = max(0, min(i2, len(pts) - 2))
        t2 = (s - s_cum[i2]) / max(s_cum[i2 + 1] - s_cum[i2], 1e-12)
        out.append(pts[i2] + t2 * (pts[i2 + 1] - pts[i2]))
        idx.append(i2)
    out.append(pts[-1])
    idx.append(len(pts) - 1)
    return np.asarray(out), np.asarray(idx, dtype=int)


def tangents_of(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    n = len(points)
    tan = np.zeros((n, 2))
    for i in range(n):
        if n == 1:
            tan[i] = (1.0, 0.0)
        elif i == 0:
            d = points[1] - points[0]
        elif i == n - 1:
            d = points[-1] - points[-2]
        else:
            d = points[i + 1] - points[i - 1]
        nrm = float(np.linalg.norm(d))
        tan[i] = d / nrm if nrm > 1e-9 else (1.0, 0.0)
    return tan


# --------------------------------------------------------------------------- #
# Geometric primitives
# --------------------------------------------------------------------------- #
def path_samples(q: np.ndarray, step_m: float) -> np.ndarray:
    """Dense samples along a polyline (spacing <= step_m)."""
    q = np.asarray(q, dtype=float)
    if len(q) < 2:
        return q.copy()
    pts = []
    for a, b in zip(q[:-1], q[1:]):
        L = float(np.linalg.norm(b - a))
        n = max(2, int(math.ceil(L / step_m)) + 1)
        ts = np.linspace(0.0, 1.0, n, endpoint=False)
        pts.append(a[None, :] + ts[:, None] * (b - a)[None, :])
    pts.append(q[-1][None, :])
    return np.vstack(pts)


def segment_free(scene: Scene, a: np.ndarray, b: np.ndarray, step_m: Optional[float] = None) -> bool:
    step = step_m if step_m else 0.25 * scene.r_scene
    return bool(np.all(scene.is_free(path_samples(np.vstack([a, b]), step))))


def segment_path_free(scene: Scene, q: np.ndarray, step_cells: float = 0.25) -> bool:
    """Every point of every segment lies in free space (dense sampling)."""
    return bool(np.all(scene.is_free(path_samples(q, step_cells * scene.r_scene))))


def homotopy_max_lambda(scene: Scene, p: np.ndarray, q: np.ndarray,
                        step_cells: float = 0.25,
                        lam_step_cells: float = 0.25) -> Tuple[float, int]:
    """Largest ``lam`` such that every deformation state ``P_lam`` is collision free.

    ``P_lam = (1-lam) P + lam Q``.  ``lam`` is sampled so the largest per-station
    displacement between consecutive states stays below ``lam_step_cells * r_scene``;
    every intermediate path is checked densely.  Returning the *largest safe*
    lambda (instead of only a boolean) lets the caller fall back to ``P_lam`` as
    the accepted centre line, which is exactly the alpha-bisection of spec 4.3
    but in a single sweep.
    """
    disp = float(np.max(np.linalg.norm(q - p, axis=1))) if len(p) else 0.0
    step = max(lam_step_cells * scene.r_scene, 1e-6)
    n_lam = max(2, int(math.ceil(disp / step)) + 1)
    lams = np.linspace(0.0, 1.0, n_lam)
    last_safe = 0.0
    for lam in lams:
        ql = (1.0 - lam) * p + lam * q
        if not bool(np.all(scene.is_free(ql))):
            return last_safe, n_lam
        if not segment_path_free(scene, ql, step_cells):
            return last_safe, n_lam
        last_safe = float(lam)
    return 1.0, n_lam


def homotopy_path_free(scene: Scene, p: np.ndarray, q: np.ndarray,
                       step_cells: float = 0.25, lam_step_cells: float = 0.25) -> Tuple[bool, int]:
    lam, n = homotopy_max_lambda(scene, p, q, step_cells, lam_step_cells)
    return bool(lam >= 1.0 - 1e-9), n


# --------------------------------------------------------------------------- #
# Route constraint domain (V2.1 section 2.2)
# --------------------------------------------------------------------------- #
def _point_segment_distances(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distances from each point in ``pts`` to every segment ``[a_k, b_k]``."""
    ab = b - a                                    # (M,2)
    ap = pts[:, None, :] - a[None, :, :]           # (N,M,2)
    denom = np.maximum(np.sum(ab * ab, axis=1), 1e-12)[None, :]
    t = np.clip(np.sum(ap * ab[None, :, :], axis=2) / denom, 0.0, 1.0)
    proj = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    return np.linalg.norm(pts[:, None, :] - proj, axis=2)      # (N,M)


def build_route_constraints(p: np.ndarray, scene: Scene, cfg: Dict) -> Dict[str, np.ndarray]:
    """Per-station offset radius, tangential limit, arc-length weights and spacing."""
    cc = cfg["centerline"]
    n = len(p)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)           # (n-1,)
    ds = np.maximum(seg, 1e-6)
    s_cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = max(float(s_cum[-1]), 1e-6)
    L_ref = float(np.mean(ds))

    # arc-length weights (trapezoid); avoids over-weighting densely sampled areas
    vor = np.empty(n)
    vor[0] = 0.5 * ds[0]
    vor[-1] = 0.5 * ds[-1]
    if n > 2:
        vor[1:-1] = 0.5 * (ds[:-1] + ds[1:])
    omega = vor / vor.sum()

    # local station spacing L_i (mean of the adjacent segments)
    L_i = np.empty(n)
    L_i[0] = ds[0]
    L_i[-1] = ds[-1]
    if n > 2:
        L_i[1:-1] = 0.5 * (ds[:-1] + ds[1:])

    tangent = tangents_of(p)

    # protective radius against confusing distant path sections (spec 2.2)
    r_max = float(cc.get("r_max_m", 3.0))
    R = np.full(n, r_max)
    if n > 3:
        a_seg = p[:-1]
        b_seg = p[1:]
        dist_all = _point_segment_distances(p, a_seg, b_seg)   # (n, n-1)
        idx = np.arange(n)
        # Only sections that are *arc-length distant* count as a different part of
        # the route; |j-i| > 1 also flags the same corridor bending and collapsed
        # R_i to ~0.5 m in the Berlin diagnostics.
        gap = int(cc.get("nonlocal_min_index_gap", 4))
        non_local = np.abs(idx[:, None] - np.arange(n - 1)[None, :]) > gap
        masked = np.where(non_local, dist_all, np.inf)
        d_nl = np.min(masked, axis=1)
        safe = np.isfinite(d_nl)
        R[safe] = np.minimum(r_max, 0.45 * d_nl[safe])
    d_safe = (float(cfg["robot"]["radius_m"]) + float(cfg["robot"]["safety_margin_m"])
              + float(cc.get("numeric_margin_m", 0.01)))
    return {
        "R": R, "tangent": tangent, "L_i": L_i, "omega": omega, "ds": ds,
        "L_ref": L_ref, "s_cum": s_cum, "total": total,
        "eta": float(cc.get("station_tangent_limit_ratio", 0.25)), "d_safe": d_safe,
    }


def project_into_domain(scene: Scene, q: np.ndarray, p: np.ndarray,
                        cons: Dict[str, np.ndarray], shrink: float = 0.95) -> np.ndarray:
    """Pull a candidate centreline back into offset / tangential / clearance limits."""
    q = np.array(q, dtype=float, copy=True)
    n = len(p)
    R = cons["R"] * shrink
    eta_L = cons["eta"] * cons["L_i"] * shrink
    dev = q - p
    # circular offset cap
    r = np.linalg.norm(dev, axis=1)
    over = r > R
    if np.any(over):
        dev[over] *= (R[over] / r[over])[:, None]
    # tangential cap
    t = cons["tangent"]
    tang = np.sum(dev * t, axis=1)
    lim = eta_L
    bad = np.abs(tang) > lim
    if np.any(bad):
        dev[bad] -= np.sign(tang[bad])[:, None] * (np.abs(tang[bad]) - lim[bad])[:, None] * t[bad]
    q = p + dev
    q[0], q[-1] = p[0], p[-1]
    # clearance: shrink the interior deviation until D >= d_safe
    d_safe = cons["d_safe"]
    for _ in range(25):
        D = scene.distance(q)
        bad = (D < d_safe)
        bad[0] = bad[-1] = False
        if not np.any(bad):
            break
        dev[bad] *= 0.5
        q = p + dev
        q[0], q[-1] = p[0], p[-1]
    return q


def make_clearance_initializations(scene: Scene, p: np.ndarray,
                                  cons: Dict[str, np.ndarray], cfg: Dict) -> List[Tuple[str, np.ndarray]]:
    """Deterministic multi-start seeds (spec 3.1): P, lateral-gradient offsets."""
    cc = cfg["centerline"]
    inits: List[Tuple[str, np.ndarray]] = [("astar", p.copy())]
    grad = scene.gradient(p)
    t = cons["tangent"]
    g_t = np.sum(grad * t, axis=1)[:, None] * t
    g_perp = grad - g_t
    norm = np.linalg.norm(g_perp, axis=1)
    u = np.where(norm[:, None] > 1e-6, g_perp / np.maximum(norm, 1e-9)[:, None], 0.0)

    offsets = cc.get("init_offsets_m", [0.5, 1.0])
    count = int(cc.get("multi_start_count", 3))
    for k, delta in enumerate(offsets[: max(0, count - 1)]):
        q0 = project_into_domain(scene, p + float(delta) * u, p, cons)
        if k == 1 and len(q0) > 3:            # light smoothing for the larger offset
            sm = q0.copy()
            sm[1:-1] = 0.5 * q0[1:-1] + 0.25 * (q0[:-2] + q0[2:])
            q0 = project_into_domain(scene, sm, p, cons)
        inits.append((f"grad_offset_{delta:g}m", q0))
    return inits


# --------------------------------------------------------------------------- #
# V2.1 objective / constraints
# --------------------------------------------------------------------------- #
def _arc_second_derivative(q: np.ndarray, ds: np.ndarray) -> np.ndarray:
    """Second derivative of ``q`` in arc length for non-uniform spacing."""
    dq1 = (q[2:] - q[1:-1]) / ds[1:, None]
    dq0 = (q[1:-1] - q[:-2]) / ds[:-1, None]
    c = 2.0 / (ds[:-1] + ds[1:])[:, None]
    return c * (dq1 - dq0)


def _unpack(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    q = np.empty_like(p)
    q[0], q[-1] = p[0], p[-1]
    q[1:-1] = x.reshape(-1, 2)
    return q


def _pack(q: np.ndarray) -> np.ndarray:
    return q[1:-1].ravel().copy()


def centreline_objective(scene: Scene, q: np.ndarray, p: np.ndarray,
                         cons: Dict[str, np.ndarray], cc: Dict,
                         field: Optional["_FieldCache"] = None,
                         x: Optional[np.ndarray] = None) -> Tuple[float, Dict[str, float]]:
    D = field.D_all(x) if (field is not None and x is not None) else scene.distance(q)
    cap = float(cc["clearance_cap_m"])
    JD = float(np.sum(cons["omega"] * np.minimum(D, cap) / cap))
    a = _arc_second_derivative(q, cons["ds"])
    L_ref2 = cons["L_ref"] ** 2
    Js = float(np.sum(cons["omega"][1:-1] * L_ref2 * np.sum(a * a, axis=1)))
    Jp = float(np.sum(cons["omega"] * np.sum((q - p) ** 2, axis=1)
                      / (cons["R"] ** 2 + 1e-9)))
    w = (float(cc["w_clear"]), float(cc["w_smooth"]), float(cc["w_pull"]))
    J = -w[0] * JD + w[1] * Js + w[2] * Jp
    return J, {"J_D": JD, "J_s": Js, "J_p": Jp}


def centreline_grad(scene: Scene, q: np.ndarray, p: np.ndarray,
                    cons: Dict[str, np.ndarray], cc: Dict,
                    field: Optional["_FieldCache"] = None,
                    x: Optional[np.ndarray] = None) -> np.ndarray:
    """Analytic gradient of the V2.1 objective wrt the interior stations."""
    n = len(p)
    cap = float(cc["clearance_cap_m"])
    wD, wS, wP = float(cc["w_clear"]), float(cc["w_smooth"]), float(cc["w_pull"])
    g = np.zeros((n, 2))

    # clearance term: -wD * omega * min(D,cap)/cap
    if field is not None and x is not None:
        D, gradD = field.D_all(x), field.G_all(x)
    else:
        D, gradD = scene.distance(q), scene.gradient(q)
    active = (D < cap).astype(float)[:, None]
    g += -wD * cons["omega"][:, None] / cap * gradD * active

    # smoothing term
    ds = cons["ds"]
    L_ref2 = cons["L_ref"] ** 2
    a = _arc_second_derivative(q, ds)                     # (n-2,2)
    c = 2.0 / (ds[:-1] + ds[1:])                          # (n-2,)
    w_i = cons["omega"][1:-1] * L_ref2
    coef = 2.0 * w_i[:, None] * a                          # (n-2,2)
    g[2:] += coef * (c / ds[1:])[:, None]
    g[1:-1] += coef * (-c / ds[1:] - c / ds[:-1])[:, None]
    g[:-2] += coef * (c / ds[:-1])[:, None]

    # weak offset regulariser
    g += wP * 2.0 * (q - p) * cons["omega"][:, None] / (cons["R"] ** 2 + 1e-9)[:, None]
    return _pack(g)


class _FieldCache:
    """One-entry/per-x cache for the distance field and its gradient.

    SLSQP evaluates the objective and every constraint function at the *same* x
    during one line search, so without this each station's ``D`` and ``gradient``
    were recomputed 5-10 times: profiling showed 76 380 ``scene.distance`` calls
    and 4.54 s of a 4.71 s solve dominated by them.
    """

    def __init__(self, scene: Scene, p: np.ndarray):
        self.scene = scene
        self.p = p
        self._key = None
        self._q = None
        self._all = None           # (D at every station, grad at every station)
        self._seg = None           # (D at collocation points, grad)

    def q(self, x: np.ndarray) -> np.ndarray:
        if self._q is None or x is not self._key:
            self._q = _unpack(x, self.p)
            self._key = x
            self._all = None
            self._seg = None
        return self._q

    def D_all(self, x: np.ndarray) -> np.ndarray:
        q = self.q(x)
        if self._all is None:
            self._all = (self.scene.distance(q), self.scene.gradient(q))
        return self._all[0]

    def G_all(self, x: np.ndarray) -> np.ndarray:
        q = self.q(x)
        if self._all is None:
            self._all = (self.scene.distance(q), self.scene.gradient(q))
        return self._all[1]

    def node(self, x: np.ndarray):
        return self.D_all(x)[1:-1], self.G_all(x)[1:-1]

    def seg(self, x: np.ndarray):
        q = self.q(x)
        if self._seg is None:
            a, b = q[:-1], q[1:]
            ts = getattr(self, "coll_t", np.array([1.0 / 3.0, 2.0 / 3.0]))
            pts = np.vstack([a + float(ti) * (b - a) for ti in ts])
            self._seg = (self.scene.distance(pts), self.scene.gradient(pts))
        return self._seg


def make_constraints(scene: Scene, p: np.ndarray, cons: Dict[str, np.ndarray],
                     cc: Dict, cache: Optional["_FieldCache"] = None
                     ) -> Tuple[List[Dict[str, object]], int]:
    """SLSQP inequality constraints (``g(x) >= 0``) with analytic Jacobians."""
    n = len(p)
    m = n - 2                                             # free variables
    d_safe = cons["d_safe"]
    R = cons["R"]
    t = cons["tangent"]
    lim = (cons["eta"] * cons["L_i"]) ** 2
    field = cache if cache is not None else _FieldCache(scene, p)

    n_seg = n - 1
    n_coll = int(cc.get("segment_collocation_points", 2))          # 1 = midpoint only
    coll_t = (np.array([0.5]) if n_coll <= 1 else np.array([1.0 / 3.0, 2.0 / 3.0]))
    field.coll_t = coll_t

    def _seg_points(q):
        a, b = q[:-1], q[1:]
        return np.vstack([a + ti * (b - a) for ti in coll_t])

    _seg_at = {
        0.5: (np.array([0.5]), np.array([0.5])),
        1.0 / 3.0: (np.array([2.0 / 3.0]), np.array([1.0 / 3.0])),
        2.0 / 3.0: (np.array([1.0 / 3.0]), np.array([2.0 / 3.0])),
    }
    # derivative weights of the collocation point wrt q_i and q_{i+1}
    _seg_weights = {t: (1.0 - t, t) for t in (0.5, 1.0 / 3.0, 2.0 / 3.0)}

    # All Jacobians are built fully vectorised: dense Python loops over the
    # station count dominated the runtime on long paths (up to 156 s for 116 m).
    _rows = np.arange(m)
    _cols_both = np.stack([2 * _rows, 2 * _rows + 1], axis=1)          # (m,2)

    def _seg_row_cols():
        j = np.arange(n_seg)
        cj = np.where((j >= 1) & (j <= n - 2), 2 * (j - 1), -1)
        cj1 = np.where((j + 1 >= 1) & (j + 1 <= n - 2), 2 * j, -1)
        return j, cj, cj1

    _seg_j, _seg_cj, _seg_cj1 = _seg_row_cols()

    # --- block 1: clearance at stations AND at segment collocation points ----
    def clearance_fun(x):
        d_node = field.node(x)[0]
        d_seg = field.seg(x)[0]
        return np.concatenate([d_node, d_seg]) - d_safe

    def clearance_jac(x):
        J = np.zeros((m + n_coll * n_seg, 2 * m))
        g_node = field.node(x)[1]
        J[_rows[:, None], _cols_both] = g_node
        g_seg = field.seg(x)[1]                     # (n_coll*n_seg, 2), grouped per collocation
        j, cj, cj1 = _seg_j, _seg_cj, _seg_cj1
        for c, ti in enumerate(coll_t):
            ga = g_seg[c * n_seg:(c + 1) * n_seg]
            ca, cb = _seg_weights[float(ti)]
            rows = m + c * n_seg + j
            ok = cj >= 0
            J[rows[ok, None], np.stack([cj[ok], cj[ok] + 1], axis=1)] += ca * ga[ok]
            ok = cj1 >= 0
            J[rows[ok, None], np.stack([cj1[ok], cj1[ok] + 1], axis=1)] += cb * ga[ok]
        return J

    def offset_fun(x):
        q = field.q(x)
        d = q[1:-1] - p[1:-1]
        return R[1:-1] ** 2 - np.sum(d * d, axis=1)

    def offset_jac(x):
        q = field.q(x)
        d = q[1:-1] - p[1:-1]
        J = np.zeros((m, 2 * m))
        J[_rows[:, None], _cols_both] = -2.0 * d
        return J

    def tangent_fun(x):
        q = field.q(x)
        d = q[1:-1] - p[1:-1]
        tang = np.sum(d * t[1:-1], axis=1)
        return lim[1:-1] - tang * tang

    def tangent_jac(x):
        q = field.q(x)
        d = q[1:-1] - p[1:-1]
        tang = np.sum(d * t[1:-1], axis=1)
        J = np.zeros((m, 2 * m))
        J[_rows[:, None], _cols_both] = -2.0 * tang[:, None] * t[1:-1]
        return J

    cons_list = [
        {"type": "ineq", "fun": clearance_fun, "jac": clearance_jac},
        {"type": "ineq", "fun": offset_fun, "jac": offset_jac},
        {"type": "ineq", "fun": tangent_fun, "jac": tangent_jac},
    ]
    return cons_list, m

    def segment_fun(x):
        q = _unpack(x, p)
        p1, p2 = _seg_points(q)
        return np.concatenate([scene.distance(p1), scene.distance(p2)]) - d_safe

    def segment_jac(x):
        q = _unpack(x, p)
        p1, p2 = _seg_points(q)
        g1 = scene.gradient(p1)
        g2 = scene.gradient(p2)
        J = np.zeros((2 * n_seg, 2 * m))
        j, cj, cj1 = _seg_row_cols()
        for row_base, ga, gb, ca, cb in ((0, g1, g1, 2.0 / 3.0, 1.0 / 3.0),
                                         (n_seg, g2, g2, 1.0 / 3.0, 2.0 / 3.0)):
            rows = row_base + j
            ok = cj >= 0
            J[rows[ok, None], np.stack([cj[ok], cj[ok] + 1], axis=1)] += ca * ga[ok]
            ok = cj1 >= 0
            J[rows[ok, None], np.stack([cj1[ok], cj1[ok] + 1], axis=1)] += cb * gb[ok]
        return J

    cons_list = [
        {"type": "ineq", "fun": clearance_fun, "jac": clearance_jac},
        {"type": "ineq", "fun": offset_fun, "jac": offset_jac},
        {"type": "ineq", "fun": tangent_fun, "jac": tangent_jac},
        {"type": "ineq", "fun": segment_fun, "jac": segment_jac},
    ]
    return cons_list, m


# --------------------------------------------------------------------------- #
# Optimiser
# --------------------------------------------------------------------------- #
def _interp_offsets(p: np.ndarray, idx_c: np.ndarray, q_c: np.ndarray) -> np.ndarray:
    """Interpolate a coarse centreline's offsets onto the full station set."""
    def norm_arc(x):
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(x, axis=0), axis=1))])
        return s / max(s[-1], 1e-9)

    s_c = norm_arc(p[idx_c])
    s_f = norm_arc(p)
    off_c = q_c - p[idx_c]
    off = np.stack([np.interp(s_f, s_c, off_c[:, 0]), np.interp(s_f, s_c, off_c[:, 1])], axis=1)
    q = p + off
    q[0], q[-1] = p[0], p[-1]
    return q


def optimise_centerline_v21(scene: Scene, p: np.ndarray, cfg: Dict) -> Tuple[np.ndarray, Dict]:
    """Route-constrained clearance optimisation (multi-resolution for long paths).

    SLSQP cost grows roughly with ``N^2`` per iteration and long paths needed up
    to 156 s.  When the station count exceeds ``max_optimised_stations`` the
    problem is solved on a uniformly thinned *knot* set and the resulting offsets
    are linearly interpolated back onto every station; the full-resolution result
    is then projected into the constraint domain and validated as usual.
    """
    t0 = time.perf_counter()
    cc = cfg["centerline"]
    n = len(p)
    n_free = n - 2
    knot_target = int(cc.get("max_optimised_stations", 32))
    if n_free > knot_target and knot_target >= 6:
        idx_c = np.unique(np.concatenate(
            [[0], np.round(np.linspace(1, n - 2, knot_target)).astype(int), [n - 1]]))
        q_c, info = _solve_v21_fixed_stations(scene, p[idx_c], cfg)
        q = _interp_offsets(p, idx_c, q_c)
        cons = build_route_constraints(p, scene, cfg)
        q = project_into_domain(scene, q, p, cons, shrink=1.0)
        info = dict(info)
        info["status"] = f"{info.get('status', 'ok')}+coarse{len(idx_c)}"
        info["time_ms"] = (time.perf_counter() - t0) * 1e3
        info["coarse_stations"] = int(len(idx_c))
        return q, info
    return _solve_v21_fixed_stations(scene, p, cfg)


def _solve_v21_fixed_stations(scene: Scene, p: np.ndarray, cfg: Dict) -> Tuple[np.ndarray, Dict]:
    """Route-constrained clearance optimisation with deterministic multi-start."""
    from scipy.optimize import minimize

    t0 = time.perf_counter()
    cc = cfg["centerline"]
    n = len(p)
    if n < 3:
        return p.copy(), {"status": "too_short", "iterations": 0,
                          "time_ms": (time.perf_counter() - t0) * 1e3, "checks": {}}

    cons = build_route_constraints(p, scene, cfg)
    cache = _FieldCache(scene, p)
    slsqp_cons, m = make_constraints(scene, p, cons, cc, cache=cache)

    def fun(x):
        q = cache.q(x)
        J, _parts = centreline_objective(scene, q, p, cons, cc, field=cache, x=x)
        return J

    def jac(x):
        q = cache.q(x)
        return centreline_grad(scene, q, p, cons, cc, field=cache, x=x)

    best = None
    total_iters = 0
    # Effort control (profiling, reports_v2_1/profiling.json): the field lookups
    # dominate; multi-start is the only budget lever that actually changes the
    # runtime, and the three seeds converge to the same objective.
    n_free = n - 2
    max_iter = int(cc.get("_iter_override") or cc["max_iter"])
    inits = make_clearance_initializations(scene, p, cons, cfg)
    inits = inits[: max(1, int(cc.get("multi_start_count", 3)))]
    for name, q0 in inits:
        x0 = _pack(q0)
        try:
            res = minimize(fun, x0, jac=jac, method="SLSQP", constraints=slsqp_cons,
                           options={"maxiter": max_iter,
                                    "ftol": float(cc["ftol"])})
        except Exception:                              # pragma: no cover
            continue
        total_iters += int(getattr(res, "nit", 0) or 0)
        q = _unpack(np.asarray(res.x), p)
        q[0], q[-1] = p[0], p[-1]
        J, parts = centreline_objective(scene, q, p, cons, cc)
        cand = {"init": name, "q": q, "J": J, "parts": parts,
                "success": bool(res.success), "message": str(res.message)}
        if best is None or J < best["J"]:
            best = cand

    if best is None:
        return p.copy(), {"status": "solver_failed", "iterations": total_iters,
                          "time_ms": (time.perf_counter() - t0) * 1e3, "checks": {}}

    q = project_into_domain(scene, best["q"], p, cons, shrink=1.0)   # no-op if feasible
    return q, {"status": "ok", "init": best["init"], "J": best["J"], "parts": best["parts"],
               "iterations": total_iters, "time_ms": (time.perf_counter() - t0) * 1e3,
               "checks": {}, "constraints": {k: v for k, v in cons.items()
                                             if k in ("R", "eta", "d_safe", "L_ref")}}


def optimise_centerline_legacy(scene: Scene, p: np.ndarray, cfg: Dict) -> Tuple[np.ndarray, Dict]:
    """V2 objective (kept to reproduce the previous results)."""
    from scipy.optimize import minimize

    t0 = time.perf_counter()
    cc = cfg["centerline"]
    n = len(p)
    if n < 3:
        return p.copy(), {"status": "too_short", "iterations": 0,
                          "time_ms": (time.perf_counter() - t0) * 1e3, "checks": {}}
    mean_spacing = float(np.mean(np.linalg.norm(np.diff(p, axis=0), axis=1))) or 1.0
    w_p = float(cc["w_pull"])
    w_d = float(cc["w_clear"])
    w_s = float(cc["w_smooth"])
    r_max = float(cc["r_max_m"])
    ramp = float(cc.get("end_ramp_m", 1.0))
    clear_min = float(cc["clear_min_m"])
    lam_clear = float(cc.get("clearance_penalty", 50.0))
    lam_rad = float(cc.get("radius_penalty", 200.0))

    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s_cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s_cum[-1])
    ramp_w = np.clip(np.minimum(s_cum, total - s_cum) / max(ramp, 1e-6), 0.0, 1.0)
    ramp_w[[0, -1]] = 0.0

    def objective(x):
        q = x.reshape(n, 2)
        dev = q - p
        f = w_p * np.sum(ramp_w[:, None] * dev ** 2) / (mean_spacing ** 2)
        d = scene.distance(q)
        f += -w_d * np.sum(d) / mean_spacing
        if n > 2:
            d2 = q[2:] - 2 * q[1:-1] + q[:-2]
            f += w_s * np.sum(d2 ** 2) / (mean_spacing ** 2)
        f += lam_clear * np.sum(np.maximum(clear_min - d, 0.0) ** 2)
        r = np.linalg.norm(dev, axis=1)
        f += lam_rad * np.sum(np.maximum(r - r_max, 0.0) ** 2)
        return float(f)

    def grad(x):
        q = x.reshape(n, 2)
        dev = q - p
        g = 2.0 * w_p * ramp_w[:, None] * dev / (mean_spacing ** 2)
        dfield = scene.distance(q)
        gq = scene.gradient(q)
        g += -w_d * gq / mean_spacing
        if n > 2:
            d2 = q[2:] - 2 * q[1:-1] + q[:-2]
            gs = np.zeros_like(q)
            gs[2:] += 2.0 * w_s * d2 / (mean_spacing ** 2)
            gs[1:-1] += -4.0 * w_s * d2 / (mean_spacing ** 2)
            gs[:-2] += 2.0 * w_s * d2 / (mean_spacing ** 2)
            g += gs
        viol = np.maximum(clear_min - dfield, 0.0)
        g += -2.0 * lam_clear * (viol[:, None] * gq)
        r = np.linalg.norm(dev, axis=1)
        viol_r = np.maximum(r - r_max, 0.0)
        unit = dev / np.maximum(r, 1e-12)[:, None]
        g += 2.0 * lam_rad * (viol_r[:, None] * unit)
        return g.ravel()

    res = minimize(objective, p.ravel().copy(), jac=grad, method="L-BFGS-B",
                   options={"maxiter": int(cc["max_iter"]),
                            "maxfun": int(cc["max_iter"]) * 3, "ftol": 1e-10, "gtol": 1e-8})
    q = res.x.reshape(n, 2)
    q[0], q[-1] = p[0], p[-1]
    dev = q - p
    r = np.linalg.norm(dev, axis=1)
    over = r > r_max
    if np.any(over):
        q[over] = p[over] + dev[over] * (r_max / r[over])[:, None]
        q[0], q[-1] = p[0], p[-1]
    return q, {"status": "ok" if (res.success or res.nit > 0) else "not_converged",
               "iterations": int(getattr(res, "nit", 0) or 0),
               "time_ms": (time.perf_counter() - t0) * 1e3, "checks": {}}


def optimise_centerline(scene: Scene, p: np.ndarray, cfg: Dict) -> Tuple[np.ndarray, Dict]:
    method = cfg["centerline"].get("method", "legacy_weighted")
    if method == "route_constrained_clearance":
        return optimise_centerline_v21(scene, p, cfg)
    return optimise_centerline_legacy(scene, p, cfg)


# --------------------------------------------------------------------------- #
# Validation + path assembly
# --------------------------------------------------------------------------- #
def validate_centerline(scene: Scene, p: np.ndarray, q: np.ndarray,
                        cfg: Dict, lam_max: Optional[float] = None) -> Dict[str, object]:
    """All geometric checks required by spec section 4.1."""
    cc = cfg["centerline"]
    method = cc.get("method", "legacy_weighted")
    step_cells = float(cc.get("segment_check_step_cells", 0.25))
    lam_cells = float(cc.get("homotopy_check_step_cells", 0.25))
    cons = build_route_constraints(p, scene, cfg)
    # the per-station radius (protected against confusing distant path sections)
    # belongs to the V2.1 constraint domain; the legacy method only used R_max
    R = cons["R"] if method == "route_constrained_clearance" else np.full(len(p), float(cc.get("r_max_m", 3.0)))
    dev = q - p
    r = np.linalg.norm(dev, axis=1)
    t = cons["tangent"]
    tang = np.abs(np.sum(dev * t, axis=1))
    D_q = scene.distance(q)
    node_free = bool(np.all(scene.is_free(q)))
    seg_free = bool(segment_path_free(scene, q, step_cells))
    endp_ok = bool(np.allclose(q[0], p[0]) and np.allclose(q[-1], p[-1]))
    offset_ok = bool(np.all(r <= R + 1e-6))
    tangent_ok = bool(np.all(tang <= cons["eta"] * cons["L_i"] + 1e-6))
    clearance_ok = bool(np.all(D_q[1:-1] >= cons["d_safe"] - 1e-6))
    clear_D_p = scene.distance(p)
    mono = bool(np.all(np.diff(cons["s_cum"]) >= 0))
    if lam_max is None:
        lam_max, n_lam = homotopy_max_lambda(scene, p, q, step_cells, lam_cells)
    else:
        n_lam = max(2, int(math.ceil(max(float(np.max(r)), 1e-9)
                                      / max(lam_cells * scene.r_scene, 1e-6))) + 1)
    homo = bool(lam_max >= 1.0 - 1e-9)
    length_q = float(np.sum(np.linalg.norm(np.diff(q, axis=0), axis=1)))
    length_p = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
    return {
        "node_free": node_free,
        "segment_free": seg_free,
        "endpoints_fixed": endp_ok,
        "offset_ok": offset_ok,
        "tangent_limit_ok": tangent_ok,
        "clearance_ok": clearance_ok,
        "station_order_ok": mono,
        "homotopy_verified": bool(homo),
        "homotopy_states": n_lam,
        "homotopy_max_lambda": float(lam_max),
        "max_offset_m": float(r.max()) if len(r) else 0.0,
        "offset_p50_m": float(np.median(r)) if len(r) else 0.0,
        "offset_p95_m": float(np.percentile(r, 95)) if len(r) else 0.0,
        "clearance_p_mean": float(np.mean(clear_D_p)),
        "clearance_p_min": float(np.min(clear_D_p)),
        "clearance_q_mean": float(np.mean(D_q)),
        "clearance_q_min": float(np.min(D_q)),
        "clearance_gain_mean": float(np.mean(D_q - clear_D_p)),
        "length_ratio": float(length_q / max(length_p, 1e-9)),
        "n_at_offset_limit": int(np.sum(r >= R - 1e-3)),
        "R_mean": float(np.mean(R)),
        "d_safe": float(cons["d_safe"]),
        "method": method,
    }


def build_path(scene: Scene, start_xy: np.ndarray, goal_xy: np.ndarray, cfg: Dict) -> Optional[PathData]:
    """A* -> resample -> route-constrained centreline -> validation (with fallback)."""
    cells = astar_m(scene, start_xy, goal_xy)
    if cells is None:
        return None
    p, _idx = resample_path(scene, cells, cfg["resample"])
    if len(p) < 3:
        return None
    tangent_p = tangents_of(p)

    q, info = optimise_centerline(scene, p, cfg)
    method = cfg["centerline"].get("method", "legacy_weighted")
    q[0], q[-1] = p[0], p[-1]
    step_cells = float(cfg["centerline"].get("segment_check_step_cells", 0.25))
    lam_cells = float(cfg["centerline"].get("homotopy_check_step_cells", 0.25))
    lam_max, n_lam = homotopy_max_lambda(scene, p, q, step_cells, lam_cells)
    checks = validate_centerline(scene, p, q, cfg, lam_max=lam_max)
    status = info.get("status", "ok")

    hard = ("node_free", "segment_free", "endpoints_fixed", "offset_ok")
    if method == "route_constrained_clearance":
        hard = hard + ("tangent_limit_ok", "clearance_ok")
    failure = [k for k in hard if not checks[k]]

    # Any hard failure, or a deformation that leaves the A* topology, is repaired
    # by accepting the largest safe state of the deformation family P -> Q
    # (spec 4.3), not by throwing the whole optimised path away.
    if method == "route_constrained_clearance" and (failure or lam_max < 1.0 - 1e-9):
        accept = max(0.0, lam_max * 0.98)
        if accept >= 0.05:
            q_try = p + accept * (q - p)
            chk_try = validate_centerline(scene, p, q_try, cfg)
            if chk_try["node_free"] and chk_try["segment_free"] and chk_try["homotopy_verified"]:
                q, checks = q_try, chk_try
                failure = []
                status = ("topology_limited" if accept < 0.30
                          else f"blend_{accept:.2f}")
        if failure or not checks["homotopy_verified"]:
            status = "topology_limited"
            failure = failure or ["homotopy_verified"]

    if failure:
        q = p.copy()
        status = "fallback_astar:" + ",".join(failure)
        checks = validate_centerline(scene, p, q, cfg)

    return PathData(
        start_xy=np.asarray(start_xy, float), goal_xy=np.asarray(goal_xy, float),
        astar_cells=cells, p_xy=p, q_xy=q, tangent=tangent_p,
        clearance_p=scene.distance(p), clearance_q=scene.distance(q),
        status=status, max_offset=float(checks["max_offset_m"]),
        segment_ok=bool(checks["segment_free"]),
        tangent_q=tangents_of(q),
        centerline_method=method,
        centerline_clearance_gain=float(checks["clearance_gain_mean"]),
        centerline_length_ratio=float(checks["length_ratio"]),
        centerline_homotopy_verified=bool(checks["homotopy_verified"]),
        centerline_status=str(status),
        centerline_solver_iterations=int(info.get("iterations", 0)),
        centerline_solver_time_ms=float(info.get("time_ms", 0.0)),
        centerline_checks=checks,
    )
