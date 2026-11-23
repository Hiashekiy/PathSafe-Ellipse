"""V3 pipeline: scene sampling -> A* -> centreline -> per-station LiDAR + ellipse
labels -> continuity check -> one self-contained trajectory block per path.

V3 changes the *sampling unit* only: a training sample is one full trajectory.
Every resampled A* station is a query point (no sparse anchor selection, no
64x64 patch, no clearance feature).  The A*, centre-line and ellipse algorithms
themselves are untouched.
"""
from __future__ import annotations

import subprocess

import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..local_features import _cast_one
from ..map_io import MapRecord
from .centerline import PathData, build_path, tangents_of
from .ellipse import EllipseLabel, make_label, midpoint_metric
from .scene import Scene, build_scene, free_components

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


# --------------------------------------------------------------------------- #
# Start/goal sampling
# --------------------------------------------------------------------------- #
def sample_start_goal(scene: Scene, rng: np.random.Generator, cfg: Dict,
                      used: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Sample a start/goal pair in the largest free component with 0.2L..0.75L."""
    pc = cfg["path"]
    field = scene.physical_field_m
    free = scene.free
    lab, n_comp, largest = free_components(free)
    if n_comp == 0 or largest < 200:
        return None
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    big = int(np.argmax(sizes))
    rr, cc = np.nonzero(lab == big)
    if rr.size < 100:
        return None
    overlap = float(pc["start_goal_overlap_m"])
    for _ in range(int(pc["max_attempts"])):
        i, j = rng.integers(0, rr.size, size=2)
        s = np.array([cc[i] * scene.r_scene, rr[i] * scene.r_scene])
        g = np.array([cc[j] * scene.r_scene, rr[j] * scene.r_scene])
        d = float(np.linalg.norm(s - g))
        if not (pc["min_start_goal_ratio"] * field <= d <= pc["max_start_goal_ratio"] * field):
            continue
        if any(np.linalg.norm(s - a) < overlap and np.linalg.norm(g - b) < overlap
               for a, b in used):
            continue
        return s, g
    return None


# --------------------------------------------------------------------------- #
# Sparse adaptive anchors
# --------------------------------------------------------------------------- #
def anchor_indices(scene: Scene, path: PathData, ac: Dict) -> List[int]:
    """Sparse adaptive anchor selection along the centreline (spec section 5.1).

    Spacing, arc length AND curvature all use the optimised centreline ``q``
    itself (the A* tangent is a separate feature and must not be mixed in here).
    """
    q = path.q_xy
    if len(q) < 2:
        return [0]
    seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
    s_cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s_cum[-1])
    tangent_q = path.tangent_q if getattr(path, "tangent_q", None) is not None else tangents_of(q)
    dur = np.diff(tangent_q, axis=0)
    curv = np.zeros(len(q))
    curv[1:] = np.linalg.norm(dur, axis=1) / np.maximum(seg, 1e-9)

    idx = [0]
    s = 0.0
    while s < total - 1e-9:
        i = int(np.searchsorted(s_cum, s, side="right") - 1)
        i = max(0, min(i, len(q) - 2))
        d = float(scene.distance(q[i]))
        if d <= float(ac["narrow_clearance_m"]):
            step = float(ac["spacing_narrow_m"])
        elif d >= float(ac["open_clearance_m"]) and curv[i] <= math.radians(10.0):
            step = float(ac["spacing_open_m"])
        else:
            step = float(ac["spacing_normal_m"])
        s += step
        if s >= total - 1e-9:
            break
        idx.append(int(np.searchsorted(s_cum, s, side="right") - 1))
    if idx[-1] != len(q) - 1:
        idx.append(len(q) - 1)
    return sorted(set(idx))


# --------------------------------------------------------------------------- #
# LiDAR from the original A* point p_i (never from q_i)
# --------------------------------------------------------------------------- #
def lidar_ranges(scene: Scene, p_xy: np.ndarray, ray_count: int, max_range_m: float) -> np.ndarray:
    """16 ranges (metres) cast from ``p_i`` with supercover DDA on the inflated grid."""
    angles = 2.0 * math.pi * np.arange(ray_count) / ray_count
    out = np.zeros(ray_count, dtype=np.float32)
    grid = scene.blocked.astype(np.uint8)
    x_cell, y_cell = float(p_xy[0]) / scene.r_scene, float(p_xy[1]) / scene.r_scene
    max_cells = max_range_m / scene.r_scene
    for k, a in enumerate(angles):
        d_cells, _hit = _cast_one(grid, x_cell, y_cell, math.cos(a), math.sin(a), max_cells)
        out[k] = min(d_cells * scene.r_scene, max_range_m)
    return out


# --------------------------------------------------------------------------- #
# One trajectory: every resampled A* station is a query point (V3)
# --------------------------------------------------------------------------- #
@dataclass
class TrajectoryNode:
    """One query point: A* station ``p``, ellipse centre ``q`` and its label."""

    p: np.ndarray
    q: np.ndarray
    label: EllipseLabel


def build_trajectory(scene: Scene, path: PathData, cfg: Dict,
                     boundary: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Per-point LiDAR + ellipse labels for the whole resampled A* trajectory.

    Point order is shared by every returned array: index ``i`` is exactly one
    query point.  Continuity is evaluated on *every* adjacent pair; a pair whose
    midpoint metric exceeds ``rho_max`` is split by inserting the midpoint of the
    A* station **and** of the centre-line point, which keeps ``path_xy`` and
    ``centerline_xy`` synchronous (``max_insert_per_interval`` levels, and at most
    ``max_points_factor`` times the original station count in total).
    """
    cont_cfg = cfg.get("continuity", {})
    rho_max = float(cont_cfg.get("rho_max", 0.85))
    max_depth = int(cont_cfg.get("max_insert_per_interval", 2))
    max_factor = float(cont_cfg.get("max_points_factor", 2.5))
    rays = int(cfg["lidar"]["ray_count"])
    max_range = float(cfg["lidar"]["max_range_m"])
    if boundary is None:
        boundary = scene.obstacle_boundary_cells()

    p0 = np.asarray(path.p_xy, dtype=float).reshape(-1, 2)
    q0 = np.asarray(path.q_xy, dtype=float).reshape(-1, 2)
    if len(p0) != len(q0):
        raise ValueError(f"path_xy {p0.shape} and centerline_xy {q0.shape} disagree")
    limit = max(len(p0) + 1, int(np.floor(max_factor * len(p0))))

    created: List[TrajectoryNode] = []

    def _stored(xy: np.ndarray) -> np.ndarray:
        """The exact value this coordinate will have in the NPZ (float32)."""
        return np.asarray(xy, dtype=np.float32).astype(np.float64)

    def node(p_i: np.ndarray, q_i: np.ndarray) -> TrajectoryNode:
        # labels and LiDAR are derived from the *stored* coordinates, so every
        # saved sample can be reproduced bit-for-bit from its own arrays
        nd = TrajectoryNode(p=_stored(p_i), q=_stored(q_i),
                            label=make_label(scene, _stored(q_i), boundary, cfg))
        created.append(nd)
        return nd

    base = [node(p0[i], q0[i]) for i in range(len(p0))]

    def pair_rho(a: TrajectoryNode, b: TrajectoryNode) -> Tuple[float, float]:
        mid = 0.5 * (a.label.q + b.label.q)
        return (midpoint_metric(a.label.S, a.label.q, mid),
                midpoint_metric(b.label.S, b.label.q, mid))

    def needs_split(a: TrajectoryNode, b: TrajectoryNode) -> bool:
        # a failed ellipse has no meaningful metric: never split, just flag it
        if not (a.label.safety_verified and b.label.safety_verified):
            return False
        # an inserted query point must itself stay in traversable space; a midpoint
        # whose A* point falls into the inflated obstacle band is not inserted (the
        # pair is then simply reported as not continuity-verified)
        if not bool(scene.is_free(0.5 * (a.p + b.p))):
            return False
        ra, rb = pair_rho(a, b)
        return bool(ra > rho_max or rb > rho_max)

    def split(a: TrajectoryNode, b: TrajectoryNode, depth: int) -> List[TrajectoryNode]:
        if depth >= max_depth or len(created) >= limit or not needs_split(a, b):
            return []
        m = node(0.5 * (a.p + b.p), 0.5 * (a.q + b.q))
        return split(a, m, depth + 1) + [m] + split(m, b, depth + 1)

    seq: List[TrajectoryNode] = [base[0]]
    for a, b in zip(base[:-1], base[1:]):
        seq.extend(split(a, b, 0))
        seq.append(b)

    n = len(seq)
    rho_left = np.full(n, np.nan)
    rho_right = np.full(n, np.nan)
    cont = np.ones(n, dtype=np.uint8)
    for i in range(1, n):
        a, b = seq[i - 1], seq[i]
        if not (a.label.safety_verified and b.label.safety_verified):
            cont[i - 1] = 0
            cont[i] = 0
            continue
        ra, rb = pair_rho(a, b)
        rho_left[i], rho_right[i] = ra, rb
        if ra > rho_max or rb > rho_max:
            cont[i - 1] = 0
            cont[i] = 0

    p_xy = np.asarray([nd.p for nd in seq], dtype=np.float32).reshape(-1, 2)
    q_xy = np.asarray([nd.q for nd in seq], dtype=np.float32).reshape(-1, 2)
    safety = np.asarray([int(nd.label.safety_verified) for nd in seq], dtype=np.uint8)
    return {
        "path_xy": p_xy,
        "centerline_xy": q_xy,
        "center_offset": q_xy - p_xy,                      # exact in float32
        "lidar_ranges": np.asarray(
            [lidar_ranges(scene, nd.p, rays, max_range) for nd in seq],
            dtype=np.float32).reshape(n, rays),
        "ellipse_S": np.asarray([np.asarray(nd.label.S, float).ravel() for nd in seq],
                                dtype=np.float64).reshape(n, 2, 2),
        "ellipse_ab": np.asarray([np.asarray(nd.label.ab, float) for nd in seq],
                                 dtype=np.float64).reshape(n, 2),
        "ellipse_angle": np.asarray([nd.label.angle for nd in seq], dtype=np.float64),
        "ellipse_area": np.asarray([nd.label.area for nd in seq], dtype=np.float64),
        "safety_verified": safety,
        "continuity_verified": cont,
        "label_valid": (safety & cont).astype(np.uint8),
        "solver_status": np.asarray([nd.label.status for nd in seq], dtype="U32"),
        # ---- diagnostics (not saved in the V3 NPZ) ---------------------------
        "rho_left": rho_left,
        "rho_right": rho_right,
        "clearance": np.asarray([nd.label.clearance for nd in seq], dtype=np.float64),
        "path_tangent": tangents_of(p_xy.astype(float)),
        "solve_ms": np.asarray([nd.label.solve_ms for nd in seq], dtype=np.float64),
        "halfspaces": [[{"n": [float(h.n[0]), float(h.n[1])], "d": float(h.d)}
                        for h in nd.label.halfspaces if np.all(np.isfinite(h.n))]
                       for nd in seq],
        "n_base_points": int(len(p0)),
        "n_inserted": int(n - len(p0)),
    }


# --------------------------------------------------------------------------- #
# One scene
# --------------------------------------------------------------------------- #
@dataclass
class SceneResult:
    scene: Scene
    paths: List[Dict[str, object]]
    labels: Dict[str, np.ndarray]
    stats: Dict[str, object] = field(default_factory=dict)
    halfspaces: Dict[int, List[Dict[str, List[float]]]] = field(default_factory=dict)
    trajectories: List[Dict[str, object]] = field(default_factory=list)


def _git_commit(short: bool = True) -> str:
    """Best-effort git commit of the working tree (for label provenance)."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short" if short else "HEAD", "HEAD"],
                             cwd=str(PROJECT_ROOT), capture_output=True, text=True,
                             timeout=5)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def process_scene(scene: Scene, cfg: Dict, rng: np.random.Generator) -> SceneResult:
    """Run A* -> centreline -> per-point LiDAR + ellipses -> continuity for one scene.

    V3: every resampled A* station of an accepted path is a query point, and each
    accepted path yields one self-contained trajectory block (``result.trajectories``)
    whose arrays are strictly aligned.  Only paths that passed the unified
    acceptance gate produce data; a rejected path is recorded and contributes none.
    """
    t_scene = time.perf_counter()
    boundary = scene.obstacle_boundary_cells()
    pc = cfg["path"]

    fields = {k: [] for k in (
        "point_id", "anchor_id", "path_id", "p_xy", "q_xy", "center_offset", "path_tangent",
        "lidar_ranges", "ellipse_S", "ellipse_ab", "ellipse_angle", "ellipse_area",
        "clearance", "solver_status", "safety_verified", "continuity_verified",
        "label_valid", "generation_time_ms", "rho_left", "rho_right",
    )}
    statuses: Dict[str, int] = {}
    path_records: List[Dict[str, object]] = []
    trajectories: List[Dict[str, object]] = []
    halfspaces_all: Dict[int, List[Dict[str, List[float]]]] = {}
    used_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    solve_times: List[float] = []
    n_inserted = 0

    for path_id in range(int(pc["paths_per_scene"])):
        pair = sample_start_goal(scene, rng, cfg, used_pairs)
        if pair is None:
            statuses["no_start_goal"] = statuses.get("no_start_goal", 0) + 1
            path_records.append({"path_id": path_id, "status": "no_start_goal",
                                 "accepted": False, "label_status": "failed",
                                 "failure_stage": "start_goal", "n_stations": 0,
                                 "n_anchors": 0, "n_labels": 0})
            continue
        used_pairs.append(pair)
        path = build_path(scene, pair[0], pair[1], cfg)
        if path is None:
            statuses["astar_failed"] = statuses.get("astar_failed", 0) + 1
            path_records.append({"path_id": path_id, "status": "astar_failed",
                                 "accepted": False, "label_status": "failed",
                                 "failure_stage": "astar", "n_stations": 0,
                                 "n_anchors": 0, "n_labels": 0})
            continue

        common = {
            "path_id": path_id,
            "status": path.status,
            "label_status": getattr(path, "label_status", ""),
            "accepted": bool(getattr(path, "accepted", True)),
            "hard_failures": list(getattr(path, "failures", [])),
            "solver_success": bool(getattr(path, "solver_success", False)),
            "solver_message": str(getattr(path, "solver_message", "")),
            "solver_iterations": int(getattr(path, "centerline_solver_iterations", 0)),
            "solver_objective": float(getattr(path, "solver_objective", float("nan"))),
            "solver_max_violation": float(getattr(path, "solver_max_violation", float("nan"))),
            "centerline_ms": float(getattr(path, "centerline_solver_time_ms", 0.0)),
            "start_xy": np.asarray(pair[0], float),
            "goal_xy": np.asarray(pair[1], float),
            "n_stations": int(len(path.p_xy)),
        }
        if not common["accepted"]:
            # spec stage 1: a rejected path must not generate safety labels
            statuses[common["label_status"] or "rejected"] = \
                statuses.get(common["label_status"] or "rejected", 0) + 1
            path_records.append({**common, "n_anchors": 0, "n_labels": 0,
                                 "failure_stage": "acceptance"})
            continue

        # ---- one self-contained trajectory block for this path (V3) ----------
        traj = build_trajectory(scene, path, cfg, boundary=boundary)
        n_points = int(len(traj["path_xy"]))
        trajectories.append({**traj, "path_id": int(path_id),
                             "start_xy": np.asarray(pair[0], float),
                             "goal_xy": np.asarray(pair[1], float)})

        # flat rows (one per query point) are kept for the legacy scene summaries
        for k in range(n_points):
            lab_status = str(traj["solver_status"][k])
            statuses[lab_status] = statuses.get(lab_status, 0) + 1
            fields["point_id"].append(len(fields["p_xy"]))
            fields["anchor_id"].append(k)
            fields["path_id"].append(path_id)
            fields["p_xy"].append(traj["path_xy"][k])
            fields["q_xy"].append(traj["centerline_xy"][k])
            fields["center_offset"].append(traj["center_offset"][k])
            fields["path_tangent"].append(traj["path_tangent"][k])
            fields["lidar_ranges"].append(traj["lidar_ranges"][k])
            fields["ellipse_S"].append(np.asarray(traj["ellipse_S"][k]).ravel())
            fields["ellipse_ab"].append(traj["ellipse_ab"][k])
            fields["ellipse_angle"].append(float(traj["ellipse_angle"][k]))
            fields["ellipse_area"].append(float(traj["ellipse_area"][k]))
            fields["clearance"].append(float(traj["clearance"][k]))
            fields["solver_status"].append(lab_status)
            fields["safety_verified"].append(int(traj["safety_verified"][k]))
            fields["continuity_verified"].append(int(traj["continuity_verified"][k]))
            fields["label_valid"].append(int(traj["label_valid"][k]))
            fields["generation_time_ms"].append(float(traj["solve_ms"][k]))
            fields["rho_left"].append(float(traj["rho_left"][k]))
            fields["rho_right"].append(float(traj["rho_right"][k]))
            if path_id == 0:
                # keep the half-spaces of the first path for the figures
                halfspaces_all[len(fields["p_xy"]) - 1] = traj["halfspaces"][k]

        solve_times.extend(float(v) for v in traj["solve_ms"])
        n_inserted += int(traj["n_inserted"])
        checks = dict(getattr(path, "centerline_checks", {}))
        path_records.append({
            **common,
            "n_points": int(n_points),
            "n_base_points": int(traj["n_base_points"]),
            "n_anchors": int(n_points),
            "n_labels": int(n_points),
            "n_ellipse_safe": int(np.sum(traj["safety_verified"])),
            "n_continuity_verified": int(np.sum(traj["continuity_verified"])),
            "n_both_verified": int(np.sum(traj["label_valid"])),
            "n_inserted": int(traj["n_inserted"]),
            "max_offset_m": float(path.max_offset),
            "segment_ok": bool(path.segment_ok),
            "centerline_method": getattr(path, "centerline_method", "legacy_weighted"),
            "homotopy_verified": bool(getattr(path, "centerline_homotopy_verified", False)),
            "clearance_gain_mean": float(getattr(path, "centerline_clearance_gain", 0.0)),
            "length_ratio": float(getattr(path, "centerline_length_ratio", 1.0)),
            "clearance_p_mean": float(np.mean(path.clearance_p)),
            "clearance_q_mean": float(np.mean(path.clearance_q)),
            "length_m": float(np.sum(np.linalg.norm(np.diff(path.q_xy, axis=0), axis=1))),
            "ellipse_ms_median": float(np.median(traj["solve_ms"])),
            "ellipse_ms_p95": float(np.percentile(traj["solve_ms"], 95)),
            "d_safe": checks.get("d_safe"),
            "segment_step_m": checks.get("segment_step_m"),
            "homotopy_lam_step_m": checks.get("homotopy_lam_step_m"),
            "station_order_ok": checks.get("station_order_ok"),
            "p_xy": path.p_xy,
            "q_xy": path.q_xy,
            "clearance_p": path.clearance_p,
            "clearance_q": path.clearance_q,
        })

    arrays = {}
    for k, v in fields.items():
        if k in ("solver_status",):
            arrays[k] = np.asarray(v, dtype="U32")
        elif k in ("ellipse_angle",):
            arrays[k] = np.asarray(v, dtype=np.float32)
        elif len(v) and isinstance(v[0], np.ndarray):
            arrays[k] = np.asarray(v, dtype=np.float32)
        else:
            arrays[k] = np.asarray(v)

    n_labels = int(len(arrays["anchor_id"]))
    n_safe = int(np.sum(arrays["safety_verified"])) if n_labels else 0
    n_cont = int(np.sum(arrays["continuity_verified"])) if n_labels else 0
    n_both = int(np.sum(arrays["label_valid"])) if n_labels else 0
    by_status: Dict[str, int] = {}
    for rec in path_records:
        s = str(rec.get("label_status", "")) or "unknown"
        by_status[s] = by_status.get(s, 0) + 1
    cc = cfg["centerline"]
    metadata = {
        "label_version": str(cfg["project"].get("label_version", "v2")),
        "centerline_method": str(cc.get("method", "legacy_weighted")),
        "r_max_m": float(cc.get("r_max_m", float("nan"))),
        "d_safe_m": float(cfg["robot"]["radius_m"]) + float(cfg["robot"]["safety_margin_m"])
        + float(cc.get("numeric_margin_m", 0.01)),
        "segment_check_step_cells": float(cc.get("segment_check_step_cells", 0.25)),
        "homotopy_check_step_cells": float(cc.get("homotopy_check_step_cells", 0.25)),
        "w_clear": float(cc.get("w_clear", float("nan"))),
        "w_smooth": float(cc.get("w_smooth", float("nan"))),
        "w_pull": float(cc.get("w_pull", float("nan"))),
        "max_iter": int(cc.get("max_iter", 0)),
        "multi_start_count": int(cc.get("multi_start_count", 1)),
        "max_optimised_stations": cc.get("max_optimised_stations"),
        "scene_id": scene.scene_id,
        "source_map_id": scene.source_map_id,
        "physical_field_m": scene.physical_field_m,
        "r_scene": scene.r_scene,
        "crop_px": int(scene.crop_px),
        "crop_origin": [int(v) for v in scene.crop_origin],
        "robot_radius_m": float(cfg["robot"]["radius_m"]),
        "safety_margin_m": float(cfg["robot"]["safety_margin_m"]),
        "lidar_rays": int(cfg["lidar"]["ray_count"]),
        "lidar_max_range_m": float(cfg["lidar"]["max_range_m"]),
        "query_point_policy": "every_resampled_astar_station",
        "anchor_spacing": ({k: float(v) for k, v in cfg["anchor"].items()}
                           if "anchor" in cfg else None),
        "rho_max": float(cfg["continuity"]["rho_max"]),
        "git_commit": _git_commit(),
    }
    stats = {
        "scene_id": scene.scene_id,
        "source_map_id": scene.source_map_id,
        "physical_field_m": scene.physical_field_m,
        "r_scene": scene.r_scene,
        "crop_px": scene.crop_px,
        "crop_origin": list(scene.crop_origin),
        "density_inflated": scene.density_inflated,
        "density_raw": scene.density_raw,
        "n_paths": len(path_records),
        "n_paths_accepted": int(sum(1 for r in path_records if r.get("accepted"))),
        "n_paths_failed": int(sum(1 for r in path_records if not r.get("accepted"))),
        "n_trajectories": int(len(trajectories)),
        "n_query_points": n_labels,
        "points_per_trajectory_mean": (float(np.mean([len(t["path_xy"])
                                                      for t in trajectories]))
                                       if trajectories else None),
        "n_anchors": n_labels,
        "n_ellipse_safe": n_safe,
        "n_continuity_verified": n_cont,
        "n_both_verified": n_both,
        "n_solver_converged": int(sum(1 for r in path_records if r.get("solver_success"))),
        "n_feasible_not_converged": by_status.get("feasible_not_converged", 0),
        "n_optimal_safe": by_status.get("optimal_safe", 0),
        "n_fallback": by_status.get("fallback_astar", 0),
        "n_failed": by_status.get("failed", 0) + by_status.get("infeasible", 0),
        "n_inserted": n_inserted,
        "statuses": statuses,
        "path_label_status": by_status,
        "solve_ms_median": float(np.median(solve_times)) if solve_times else None,
        "solve_ms_p95": float(np.percentile(solve_times, 95)) if solve_times else None,
        "total_ellipse_ms": float(np.sum(solve_times)) if solve_times else 0.0,
        "scene_ms": (time.perf_counter() - t_scene) * 1e3,
        "label_version": metadata["label_version"],
        "centerline_method": metadata["centerline_method"],
        "centerline_topology_verified": (
            all(bool(r.get("homotopy_verified")) for r in path_records
                if r.get("accepted")) if path_records else None),
        "centerline_clearance_gain_mean": (
            float(np.mean([r["clearance_gain_mean"] for r in path_records
                           if r.get("accepted")])) if path_records else None),
        "metadata": metadata,
        "paths": [{k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
                  for r in path_records],
    }
    return SceneResult(scene=scene, paths=path_records, labels=arrays, stats=stats,
                       halfspaces=halfspaces_all, trajectories=trajectories)
