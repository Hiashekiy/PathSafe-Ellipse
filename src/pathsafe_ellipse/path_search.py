"""Random free-space start/goal sampling and 8-neighbour A* search."""
from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Coord = Tuple[int, int]  # (x, y) == (col, row)

NEIGHBOURS: Sequence[Tuple[int, int, float]] = (
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)), (-1, -1, math.sqrt(2.0)),
)


def astar(
    occupancy: np.ndarray,
    start: Coord,
    goal: Coord,
    forbid_corner_cutting: bool = True,
    max_expansions: int = 2_000_000,
    extra_cost: Optional[np.ndarray] = None,
) -> Optional[List[Coord]]:
    """8-neighbour A* on ``occupancy`` (0 free / 1 blocked).

    Diagonal moves are forbidden when either orthogonal neighbour is blocked
    (no corner cutting), matching the implementation requirement.

    ``extra_cost`` (same shape as ``occupancy``) is added to the cost of
    *entering* a cell, which lets callers bias the search (e.g. a large
    ``extra_cost`` on cells with big clearance forces a wall-hugging path).
    The Euclidean heuristic stays admissible because all costs are >= 1.
    """
    h, w = occupancy.shape
    sx, sy = start
    gx, gy = goal
    if occupancy[sy, sx] or occupancy[gy, gx]:
        return None

    def heuristic(x: int, y: int) -> float:
        return math.hypot(x - gx, y - gy)

    def enter_cost(x: int, y: int) -> float:
        return 0.0 if extra_cost is None else float(extra_cost[y, x])

    open_heap: List[Tuple[float, float, Coord]] = [(heuristic(sx, sy), 0.0, (sx, sy))]
    g_score: Dict[Coord, float] = {(sx, sy): 0.0}
    came_from: Dict[Coord, Coord] = {}
    closed = set()
    expansions = 0

    while open_heap:
        _, g_cur, (x, y) = heapq.heappop(open_heap)
        node = (x, y)
        if node in closed:
            continue
        closed.add(node)
        if node == (gx, gy):
            path = [node]
            while node in came_from:
                node = came_from[node]
                path.append(node)
            path.reverse()
            return path
        expansions += 1
        if expansions > max_expansions:
            return None
        for dx, dy, cost in NEIGHBOURS:
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= w or ny >= h:
                continue
            if occupancy[ny, nx]:
                continue
            if forbid_corner_cutting and dx and dy:
                if occupancy[y, nx] or occupancy[ny, x]:
                    continue
            tentative = g_cur + cost + enter_cost(nx, ny)
            neighbour = (nx, ny)
            if tentative < g_score.get(neighbour, math.inf):
                g_score[neighbour] = tentative
                came_from[neighbour] = node
                heapq.heappush(open_heap, (tentative + heuristic(nx, ny), tentative, neighbour))
    return None


def sample_free_targets(
    rng: np.random.Generator,
    occupancy_inflated: np.ndarray,
    clearance: np.ndarray,
    count: int,
    boundary_margin: int,
    min_clearance: float = 1.0,
) -> List[Coord]:
    """Sample ``count`` free planning cells away from the map border."""
    h, w = occupancy_inflated.shape
    valid = (
        (occupancy_inflated == 0)
        & (clearance >= min_clearance)
    )
    valid[:boundary_margin, :] = False
    valid[-boundary_margin:, :] = False
    valid[:, :boundary_margin] = False
    valid[:, -boundary_margin:] = False
    ys, xs = np.nonzero(valid)
    if xs.size == 0:
        return []
    idx = rng.integers(0, xs.size, size=count)
    return [(int(xs[i]), int(ys[i])) for i in idx]


def path_geometry_stats(
    path: Sequence[Coord],
    clearance: np.ndarray,
    resolution: float,
) -> Dict[str, float]:
    """Arc length, straight-line distance, tortuosity and clearance statistics."""
    pts = np.asarray(path, dtype=float)
    if len(pts) < 2:
        return {
            "arc_length_cells": 0.0,
            "arc_length_m": 0.0,
            "straight_distance_cells": 0.0,
            "tortuosity": 1.0,
            "clearance_min_cells": 0.0,
            "clearance_mean_cells": 0.0,
            "clearance_min_m": 0.0,
        }
    deltas = np.diff(pts, axis=0)
    seg = np.linalg.norm(deltas, axis=1)
    arc = float(seg.sum())
    straight = float(np.linalg.norm(pts[-1] - pts[0]))
    clr = clearance[pts[:, 1].astype(int), pts[:, 0].astype(int)]
    return {
        "arc_length_cells": arc,
        "arc_length_m": arc * resolution,
        "straight_distance_cells": straight,
        "tortuosity": float(arc / straight) if straight > 1e-9 else 1.0,
        "clearance_min_cells": float(clr.min()),
        "clearance_mean_cells": float(clr.mean()),
        "clearance_min_m": float(clr.min()) * resolution,
    }


def clearance_class(distance_cells: float) -> str:
    """Compact / medium / open stratification from section 4.4."""
    if distance_cells <= 3.0:
        return "compact"
    if distance_cells <= 8.0:
        return "medium"
    return "open"
