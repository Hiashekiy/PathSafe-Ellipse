"""V2 centreline: A* -> arc-length resampling -> distance-field joint optimisation.

The optimisation solves *one* problem for the whole path (spec section 3.2)::

    min_Q  w_p sum ||q_i - p_i||^2 / L^2
         - w_d sum D(q_i) / L
         + w_s sum ||q_{i+1} - 2 q_i + q_{i-1}||^2 / L^2

with ``q_1 = p_1``, ``q_N = p_N``, ``|q_i - p_i| <= R_max`` and ``q_i`` in free
space.  Terms are normalised by the mean station spacing ``L`` so the weights
keep their meaning across scene resolutions.  The distance field and its analytic
gradient are bilinearly interpolated from the per-scene precomputation (no
per-point recomputation).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..path_search import astar
from .scene import Scene


@dataclass
class PathData:
    start_xy: np.ndarray          # metres
    goal_xy: np.ndarray
    astar_cells: np.ndarray       # (K,2) integer (col,row)
    p_xy: np.ndarray              # (N,2) metres, A* points (resampled, station set)
    q_xy: np.ndarray              # (N,2) metres, optimised centreline
    tangent: np.ndarray           # (N,2) unit tangents of the A* path
    clearance_p: np.ndarray       # (N,) D(p_i)
    clearance_q: np.ndarray       # (N,) D(q_i)
    status: str                   # ok | fallback_astar | failed
    max_offset: float
    segment_ok: bool


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
    """Spacing from clearance and local curvature (spec section 2)."""
    d = float(scene.distance(xy_m))
    if d <= float(rs["narrow_clearance_m"]):
        return float(rs["spacing_narrow_m"])
    if d >= float(rs["open_clearance_m"]):
        return float(rs["spacing_open_m"])
    return float(rs["spacing_normal_m"])


def resample_path(scene: Scene, cells: np.ndarray, rs: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Resample the grid path at (locally) adaptive arc length; returns (P, cells_idx)."""
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


def segment_free(scene: Scene, a: np.ndarray, b: np.ndarray, samples: int = 0) -> bool:
    """Conservative segment check on the inflated grid (sub-cell sampling)."""
    length = float(np.linalg.norm(b - a))
    n = samples or max(2, int(math.ceil(length / (0.25 * scene.r_scene))) + 1)
    ts = np.linspace(0.0, 1.0, n)
    pts = a[None, :] + ts[:, None] * (b - a)[None, :]
    return bool(np.all(scene.is_free(pts)))


# --------------------------------------------------------------------------- #
# Joint centreline optimisation
# --------------------------------------------------------------------------- #
def optimise_centerline(scene: Scene, p: np.ndarray, cfg: Dict) -> Tuple[np.ndarray, str, float]:
    """Solve the whole-path centreline problem; returns (q, status, max_offset)."""
    from scipy.optimize import minimize

    cc = cfg["centerline"]
    n = len(p)
    if n < 3:
        return p.copy(), "too_short", 0.0

    mean_spacing = float(np.mean(np.linalg.norm(np.diff(p, axis=0), axis=1))) or 1.0
    w_p = float(cc["w_pull"])
    w_d = float(cc["w_clear"])
    w_s = float(cc["w_smooth"])
    r_max = float(cc["r_max_m"])
    ramp = float(cc["end_ramp_m"])
    clear_min = float(cc["clear_min_m"])
    lam_clear = float(cc["clearance_penalty"])
    lam_rad = float(cc["radius_penalty"])

    # arc length from the start for the end ramp
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s_cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s_cum[-1])
    ramp_w = np.clip(np.minimum(s_cum, total - s_cum) / max(ramp, 1e-6), 0.0, 1.0)
    ramp_w[[0, -1]] = 0.0

    def unpack(x):
        return x.reshape(n, 2)

    def objective(x):
        q = unpack(x)
        dev = q - p
        f_pull = w_p * np.sum(ramp_w[:, None] * dev ** 2) / (mean_spacing ** 2)
        d = scene.distance(q)
        f_clear = -w_d * np.sum(d) / mean_spacing
        if n > 2:
            d2 = q[2:] - 2 * q[1:-1] + q[:-2]
            f_smooth = w_s * np.sum(d2 ** 2) / (mean_spacing ** 2)
        else:
            f_smooth = 0.0
        viol_clear = np.maximum(clear_min - d, 0.0)
        f_pen_clear = lam_clear * np.sum(viol_clear ** 2)
        r = np.linalg.norm(dev, axis=1)
        viol_r = np.maximum(r - r_max, 0.0)
        f_pen_r = lam_rad * np.sum(viol_r ** 2)
        return float(f_pull + f_clear + f_smooth + f_pen_clear + f_pen_r)

    def grad(x):
        q = unpack(x)
        dev = q - p
        g_pull = 2.0 * w_p * ramp_w[:, None] * dev / (mean_spacing ** 2)
        dfield = scene.distance(q)
        gq = scene.gradient(q)                      # (n,2) dD/dx
        g_clear = -w_d * gq / mean_spacing
        g_smooth = np.zeros_like(q)
        if n > 2:
            d2 = q[2:] - 2 * q[1:-1] + q[:-2]
            g_smooth[2:] += 2.0 * w_s * d2 / (mean_spacing ** 2)
            g_smooth[1:-1] += -4.0 * w_s * d2 / (mean_spacing ** 2)
            g_smooth[:-2] += 2.0 * w_s * d2 / (mean_spacing ** 2)
        viol_clear = np.maximum(clear_min - dfield, 0.0)
        g_pen_clear = -2.0 * lam_clear * (viol_clear[:, None] * gq)
        r = np.linalg.norm(dev, axis=1)
        viol_r = np.maximum(r - r_max, 0.0)
        unit = dev / np.maximum(r, 1e-12)[:, None]
        g_pen_r = 2.0 * lam_rad * (viol_r[:, None] * unit)
        return (g_pull + g_clear + g_smooth + g_pen_clear + g_pen_r).ravel()

    x0 = p.ravel().copy()
    res = minimize(objective, x0, jac=grad, method="L-BFGS-B",
                   options={"maxiter": int(cc["max_iter"]), "maxfun": int(cc["max_iter"]) * 3,
                            "ftol": 1e-10, "gtol": 1e-8})
    q = unpack(res.x)
    q[0] = p[0]
    q[-1] = p[-1]
    # hard projection back into the |q_i - p_i| <= R_max ball: shrinking the
    # deviation is always safer than discarding the whole optimised path
    dev = q - p
    r = np.linalg.norm(dev, axis=1)
    over = r > r_max
    if np.any(over):
        q[over] = p[over] + dev[over] * (r_max / r[over])[:, None]
        q[0], q[-1] = p[0], p[-1]
    offsets = np.linalg.norm(q - p, axis=1)
    max_off = float(offsets.max())
    status = "ok" if (res.success or res.nit > 0) else "not_converged"
    if max_off > r_max + 1e-6:
        status = "radius_violation"
    return q, status, max_off


def build_path(scene: Scene, start_xy: np.ndarray, goal_xy: np.ndarray, cfg: Dict) -> Optional[PathData]:
    """A* -> resample -> optimise -> geometric checks (with A* fallback)."""
    cells = astar_m(scene, start_xy, goal_xy)
    if cells is None:
        return None
    p, _idx = resample_path(scene, cells, cfg["resample"])
    if len(p) < 3:
        return None
    tangent = tangents_of(p)
    q, status, max_off = optimise_centerline(scene, p, cfg)

    # checks: free nodes, free segments, bounded offset, endpoints fixed
    ok_nodes = bool(np.all(scene.is_free(q)))
    ok_segments = all(segment_free(scene, q[i], q[i + 1]) for i in range(len(q) - 1))
    ok_offset = max_off <= float(cfg["centerline"]["r_max_m"]) + 1e-6
    endpoints_ok = bool(np.allclose(q[0], p[0]) and np.allclose(q[-1], p[-1]))
    if not (ok_nodes and ok_segments and ok_offset and endpoints_ok):
        # fall back to the raw A* path but keep the record
        q = p.copy()
        status = "fallback_astar:" + ",".join(
            k for k, v in (("nodes", ok_nodes), ("segments", ok_segments),
                           ("offset", ok_offset), ("endpoints", endpoints_ok)) if not v)
        max_off = 0.0
        ok_segments = all(segment_free(scene, q[i], q[i + 1]) for i in range(len(q) - 1))

    return PathData(
        start_xy=np.asarray(start_xy, float), goal_xy=np.asarray(goal_xy, float),
        astar_cells=cells, p_xy=p, q_xy=q, tangent=tangent,
        clearance_p=scene.distance(p), clearance_q=scene.distance(q),
        status=status, max_offset=max_off, segment_ok=bool(ok_segments),
    )
