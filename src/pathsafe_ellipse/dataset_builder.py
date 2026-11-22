"""Dataset main flow: paths, per-point features, ellipse labels, per-map artefacts.

Every map is processed independently so the work parallelises cleanly: the
worker receives one map, writes ``data/intermediate/samples/<map_key>.npz`` and
returns counters (nothing large crosses the process boundary).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .candidate_centers import UNBIASED_DIRECTION, generate_candidates
from .ellipse_solver import (
    FixedCenterEllipseSolver, build_cell_tree, candidate_priorities, ellipse_axes,
    obstacle_points,
)
from .ellipse_validation import evaluate_ellipse
from .local_features import build_patch_features
from .map_io import MapRecord, map_key
from .map_preprocess import InflatedMap, preprocess_map
from .path_sampling import sample_path, path_tangents
from .path_search import astar, path_geometry_stats, sample_free_targets

# --------------------------------------------------------------------------- #
# Path generation
# --------------------------------------------------------------------------- #
def generate_paths_for_map(
    inflated: InflatedMap,
    cfg: Dict,
    seed: int,
) -> Dict[str, object]:
    """Sample start/goal pairs and search A* paths on the inflated map."""
    ps = cfg["path_search"]
    resolution = inflated.resolution
    rng = np.random.default_rng(seed)
    occ = inflated.occupancy_inflated
    clearance = inflated.clearance

    target = int(ps["paths_per_map"])
    max_attempts = int(ps["max_path_attempts"])
    min_sg = float(ps["min_start_goal_distance"])
    min_len = float(ps["min_path_length"])
    border = int(ps["boundary_margin"])
    reuse = float(ps["max_start_goal_overlap_reuse"])

    paths: List[np.ndarray] = []
    starts: List[Tuple[int, int]] = []
    goals: List[Tuple[int, int]] = []
    stats: List[Dict[str, float]] = []
    counters = {
        "attempts": 0,
        "astar_failed": 0,
        "too_short": 0,
        "invalid_start_or_goal": 0,
        "duplicate_pair": 0,
    }

    def too_similar(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        for s, g in zip(starts, goals):
            if (np.hypot(a[0] - s[0], a[1] - s[1]) < reuse
                    and np.hypot(b[0] - g[0], b[1] - g[1]) < reuse):
                return True
        return False

    for _ in range(max_attempts):
        if len(paths) >= target:
            break
        counters["attempts"] += 1
        pair = sample_free_targets(rng, occ, clearance, 2, border, min_clearance=1.0)
        if len(pair) < 2:
            counters["invalid_start_or_goal"] += 1
            continue
        start, goal = pair[0], pair[1]
        if start == goal:
            counters["invalid_start_or_goal"] += 1
            continue
        if np.hypot(start[0] - goal[0], start[1] - goal[1]) < min_sg:
            counters["invalid_start_or_goal"] += 1
            continue
        path = astar(occ, start, goal, forbid_corner_cutting=True)
        if path is None:
            counters["astar_failed"] += 1
            continue
        geo = path_geometry_stats(path, clearance, resolution)
        if geo["arc_length_cells"] < min_len:
            counters["too_short"] += 1
            continue
        if too_similar(start, goal):
            counters["duplicate_pair"] += 1
            continue
        paths.append(np.asarray(path, dtype=np.int16))
        starts.append(start)
        goals.append(goal)
        stats.append(geo)

    return {
        "paths": paths,
        "starts": np.asarray(starts, dtype=np.int32).reshape(-1, 2),
        "goals": np.asarray(goals, dtype=np.int32).reshape(-1, 2),
        "stats": stats,
        "counters": counters,
    }


# --------------------------------------------------------------------------- #
# Sample + label construction for one map
# --------------------------------------------------------------------------- #
def _empty_sample_dict() -> Dict[str, List]:
    return {k: [] for k in (
        "map_patch", "map_patch_raw", "ray_distances", "ray_distances_raw", "ray_hits",
        "path_tangent", "offset_direction", "offset_ratio", "ellipse_ab", "ellipse_angle",
        "ellipse_center_offset", "ellipse_area", "point_inside_metric", "angle_valid",
        "label_valid", "point_xy", "path_id", "point_id", "clearance", "n_candidates",
        "n_candidates_valid", "ellipse_iters", "ellipse_converged", "clearance_class",
    )}


def build_samples_for_map(
    inflated: InflatedMap,
    raw_map: np.ndarray,
    paths_data: Dict[str, object],
    cfg: Dict,
    solver: FixedCenterEllipseSolver,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Turn every A* path of one map into path-point samples with ellipse labels."""
    ps = cfg["path_sampling"]
    lf = cfg["local_features"]
    cc = cfg["candidate_centers"]
    lab = cfg["label"]
    patch_size = int(lf["patch_size"])
    ray_count = int(lf["ray_count"])
    ray_max = float(lf["ray_max_range"])
    resolution = inflated.resolution
    eta = float(lab["point_inside_eta"])
    mu = float(lab["offset_penalty_mu"])
    min_axis_cells = float(lab["min_semi_axis"]) / resolution
    max_axis_cells = float(lab["max_semi_axis"]) / resolution

    samples = _empty_sample_dict()
    reasons: Dict[str, int] = {}
    candidate_stats = {
        "points": 0,
        "points_without_label": 0,
        "candidates_examined": 0,
        "candidates_pruned": 0,
        "candidates_solved": 0,
        "candidates_valid": 0,
        "unbiased_labels": 0,
        "offset_labels": 0,
        "unconverged_labels": 0,
        "solver_calls": 0,
        "barrier_solves": 0,
        "cvxpy_solves": 0,
        "cvxpy_fallbacks": 0,
    }

    def bump(key: str) -> None:
        reasons[key] = reasons.get(key, 0) + 1

    for path_id, path in enumerate(paths_data["paths"]):
        path_cells = [tuple(map(int, p)) for p in np.asarray(path)]
        tangent_by_cell = path_tangents(path_cells)
        cell_to_index = {cell: i for i, cell in enumerate(path_cells)}
        stations, _ = sample_path(
            path_cells,
            spacing=float(ps["spacing"]),
            angle_threshold_deg=float(ps["corner_angle_threshold_deg"]),
            corner_spacing_min=float(ps["corner_spacing_min"]),
            dedupe_distance=float(ps["dedupe_distance"]),
            include_endpoints=bool(ps["include_endpoints"]),
        )
        for point_id, station in enumerate(stations):
            gx, gy = station["grid_xy"]
            if inflated.occupancy_inflated[gy, gx] != 0:
                bump("patch_invalid")
                continue
            feats = build_patch_features(inflated.occupancy_inflated, (gx, gy),
                                         patch_size, ray_count, ray_max)
            if feats.patch[64, 64] != 0:
                bump("patch_invalid")
                continue
            raw_patch = _extract_raw_patch(raw_map, (gx, gy), patch_size)
            obs_points = obstacle_points(feats.patch)
            obs_tree = build_cell_tree(obs_points)

            cand_set = generate_candidates(
                feats,
                offset_ratios=[float(r) for r in cc["offset_ratios"]],
                ray_margin=float(cc["ray_margin"]),
                dedupe_distance=float(cc["dedupe_distance"]),
                min_center_clearance=float(cc["min_center_clearance"]),
                segment_clearance=float(cc["segment_clearance"]),
                patch_size=patch_size,
                ray_max_range=ray_max,
            )
            for key, value in cand_set.rejected.items():
                if value:
                    reasons[key] = reasons.get(key, 0) + int(value)
            if not cand_set.candidates:
                bump("no_candidate_centers")
                candidate_stats["points_without_label"] += 1
                continue

            candidate_stats["points"] += 1
            cands = cand_set.candidates
            centers = np.array([c.center_xy for c in cands])
            point_xy = np.asarray(feats.local_xy, dtype=float)
            ub, viable = candidate_priorities(centers, obs_points, patch_size, point_xy, eta, obs_tree)
            order = np.argsort(-ub)
            candidate_stats["candidates_pruned"] += int(np.count_nonzero(~viable))

            best: Optional[Tuple[float, object, object, object]] = None
            n_valid = 0
            examined = 0
            for oi in order:
                cand = cands[int(oi)]
                if not viable[oi]:
                    continue
                if best is not None and (ub[oi] - mu * cand.offset_ratio ** 2) <= best[0]:
                    break
                sol = solver.solve(feats, np.asarray(cand.center_xy, dtype=float), patch_size,
                                   obs_points=obs_points, obs_tree=obs_tree)
                examined += 1
                candidate_stats["candidates_examined"] += 1
                if sol is None:
                    bump("iris_solver_failed")
                    continue
                candidate_stats["candidates_solved"] += 1
                report, why = evaluate_ellipse(
                    sol.L, sol.center, feats.patch, point_xy,
                    eta=eta, min_axis_cells=min_axis_cells,
                    max_axis_cells=max_axis_cells,
                    safety_epsilon=float(lab["safety_epsilon"]),
                )
                for reason in why:
                    bump(reason)
                if why:
                    continue
                n_valid += 1
                candidate_stats["candidates_valid"] += 1
                score = sol.log_det - mu * cand.offset_ratio ** 2
                if best is None or score > best[0]:
                    best = (score, cand, sol, report)
            candidate_stats["candidates_pruned"] += max(0, len(order) - examined)

            if best is None:
                candidate_stats["points_without_label"] += 1
                continue

            _, cand, sol, report = best
            a_cells, b_cells, sin_t, cos_t, _phi = ellipse_axes(sol.L)
            if cand.offset_direction == UNBIASED_DIRECTION:
                candidate_stats["unbiased_labels"] += 1
            else:
                candidate_stats["offset_labels"] += 1
            if not sol.converged:
                candidate_stats["unconverged_labels"] += 1

            a_m = float(a_cells) * resolution
            b_m = float(b_cells) * resolution
            angle_valid = 1 if (b_cells > 0 and a_cells / max(b_cells, 1e-9)
                                >= float(lab["angle_axis_ratio_for_valid"])) else 0
            idx = cell_to_index.get((gx, gy))
            if idx is None:
                idx = int(np.argmin(np.linalg.norm(
                    np.asarray(path, dtype=float) - np.asarray([gx, gy]), axis=1)))
            tangent = tangent_by_cell[idx]

            samples["map_patch"].append(feats.patch)
            samples["map_patch_raw"].append(raw_patch)
            samples["ray_distances"].append(feats.ray_distances.astype(np.float32))
            samples["ray_distances_raw"].append(feats.ray_distances_raw.astype(np.float32))
            samples["ray_hits"].append(feats.ray_hits.astype(np.uint8))
            samples["path_tangent"].append(np.asarray(tangent, dtype=np.float32))
            samples["offset_direction"].append(int(cand.offset_direction))
            samples["offset_ratio"].append(float(cand.offset_ratio))
            samples["ellipse_ab"].append(np.asarray([a_m, b_m], dtype=np.float32))
            samples["ellipse_angle"].append(np.asarray([sin_t, cos_t], dtype=np.float32))
            samples["ellipse_center_offset"].append(
                np.asarray([(cand.center_xy[0] - point_xy[0]) * resolution,
                            (cand.center_xy[1] - point_xy[1]) * resolution], dtype=np.float32))
            samples["ellipse_area"].append(float(np.pi * a_m * b_m))
            samples["point_inside_metric"].append(float(report.point_metric))
            samples["angle_valid"].append(int(angle_valid))
            samples["label_valid"].append(1)
            samples["point_xy"].append(np.asarray([gx, gy], dtype=np.int32))
            samples["path_id"].append(int(path_id))
            samples["point_id"].append(int(point_id))
            local_clearance = float(inflated.clearance[gy, gx])
            samples["clearance"].append(local_clearance)
            samples["n_candidates"].append(int(len(cands)))
            samples["n_candidates_valid"].append(int(n_valid))
            samples["ellipse_iters"].append(int(sol.iters))
            samples["ellipse_converged"].append(int(bool(sol.converged)))
            samples["clearance_class"].append(
                0 if local_clearance <= 3.0 else (1 if local_clearance <= 8.0 else 2))

    candidate_stats["solver_calls"] = solver.stats["calls"]
    candidate_stats["barrier_solves"] = solver.stats["barrier"]
    candidate_stats["cvxpy_solves"] = solver.stats["cvxpy"]
    candidate_stats["cvxpy_fallbacks"] = solver.stats["cvxpy_fallbacks"]
    candidate_stats["candidate_pruning_saved"] = max(
        0, (candidate_stats["points"] * 0) + 0)
    return {k: np.asarray(v) for k, v in samples.items()}, {
        "reasons": reasons,
        "candidate_stats": candidate_stats,
    }


def _extract_raw_patch(raw_map: np.ndarray, point_xy: Tuple[int, int], patch_size: int) -> np.ndarray:
    from .local_features import extract_patch
    return extract_patch(raw_map, point_xy, patch_size)


# --------------------------------------------------------------------------- #
# Worker entry points (module level so Windows spawn can pickle them)
# --------------------------------------------------------------------------- #
_SOLVER_CACHE: Dict[str, FixedCenterEllipseSolver] = {}


def get_solver(cfg: Dict) -> FixedCenterEllipseSolver:
    es = cfg["ellipse_solver"]
    key = json.dumps({k: es[k] for k in ("max_iters", "angle_bins", "min_eigenvalue",
                                         "convergence_volume_change", "init_radius_scale",
                                         "max_cut_rounds", "backend")}, sort_keys=True)
    solver = _SOLVER_CACHE.get(key)
    if solver is None:
        solver = FixedCenterEllipseSolver(
            max_iters=es["max_iters"], angle_bins=es["angle_bins"],
            min_eigenvalue=es["min_eigenvalue"],
            convergence_volume_change=es["convergence_volume_change"],
            init_radius_scale=es["init_radius_scale"],
            max_cut_rounds=es["max_cut_rounds"], backend=es["backend"],
            solver_primary=es["solver_primary"], solver_backup=es["solver_backup"],
            separator=es.get("separator", "midpoint"),
        )
        _SOLVER_CACHE[key] = solver
    return solver


def load_inflated(record: MapRecord, cfg: Dict, paths_dict: Dict) -> Tuple[InflatedMap, np.ndarray]:
    robot = cfg["robot"]
    inflated = preprocess_map(
        record, paths_dict["map_dir"], paths_dict["inflated_dir"],
        robot_radius=float(robot["radius"]),
        kernel_radius=int(robot["inflation_kernel"]),
        iterations=int(robot["inflation_iterations"]),
    )
    return inflated, inflated.occupancy_raw


def paths_worker(payload: Dict) -> Dict:
    """Generate and cache A* paths for one map."""
    record = payload["record"]
    cfg = payload["cfg"]
    paths_dict = payload["dirs"]
    inflated, _ = load_inflated(record, cfg, paths_dict)
    seed = int(payload["seed"])
    result = generate_paths_for_map(inflated, cfg, seed)
    key = map_key(record)
    out_path = os.path.join(paths_dict["paths_dir"], f"{key}.npz")
    lengths = [len(p) for p in result["paths"]]
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    flat = (np.concatenate(result["paths"], axis=0) if result["paths"]
            else np.zeros((0, 2), dtype=np.int16))
    np.savez_compressed(
        out_path,
        flat_paths=flat,
        path_offsets=offsets,
        starts=result["starts"],
        goals=result["goals"],
        arc_length=np.asarray([s["arc_length_cells"] for s in result["stats"]], dtype=np.float32),
        tortuosity=np.asarray([s["tortuosity"] for s in result["stats"]], dtype=np.float32),
        clearance_min=np.asarray([s["clearance_min_cells"] for s in result["stats"]], dtype=np.float32),
        clearance_mean=np.asarray([s["clearance_mean_cells"] for s in result["stats"]], dtype=np.float32),
    )
    return {
        "map_key": key,
        "n_paths": len(result["paths"]),
        "counters": result["counters"],
        "stats": result["stats"],
        "file": out_path,
    }


def samples_worker(payload: Dict) -> Dict:
    """Build labels for one map from its cached paths."""
    record = payload["record"]
    cfg = payload["cfg"]
    paths_dict = payload["dirs"]
    key = map_key(record)
    inflated, raw = load_inflated(record, cfg, paths_dict)
    paths_npz = np.load(os.path.join(paths_dict["paths_dir"], f"{key}.npz"))
    flat = paths_npz["flat_paths"]
    offsets = paths_npz["path_offsets"]
    paths = [flat[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]

    solver = get_solver(cfg)
    before = dict(solver.stats)
    arrays, info = build_samples_for_map(
        inflated, raw, {"paths": paths}, cfg, solver, seed=int(payload["seed"]),
    )
    after = solver.stats
    info["candidate_stats"]["solver_calls"] = after["calls"] - before["calls"]
    info["candidate_stats"]["barrier_solves"] = after["barrier"] - before["barrier"]
    info["candidate_stats"]["cvxpy_solves"] = after["cvxpy"] - before["cvxpy"]
    info["candidate_stats"]["cvxpy_fallbacks"] = after["cvxpy_fallbacks"] - before["cvxpy_fallbacks"]

    out_path = os.path.join(paths_dict["samples_dir"], f"{key}.npz")
    np.savez_compressed(out_path, **arrays)
    info.update({"map_key": key, "n_samples": int(len(arrays["label_valid"])),
                 "n_paths": len(paths), "file": out_path})
    # per-map sidecar: every map's counters stay available to the reports
    # (a single per-shard log would only survive for the last shard to finish)
    with open(os.path.join(paths_dict["samples_dir"], f"{key}.json"), "w",
              encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)
    return info
