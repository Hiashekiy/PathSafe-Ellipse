"""Arc-length sparse sampling of A* paths, corner retention, path tangents."""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np


def _cumulative_arclength(pts: np.ndarray) -> np.ndarray:
    deltas = np.diff(pts, axis=0)
    seg = np.linalg.norm(deltas, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _interp_at(pts: np.ndarray, s_cum: np.ndarray, s: float) -> np.ndarray:
    idx = int(np.searchsorted(s_cum, s, side="right") - 1)
    idx = max(0, min(idx, len(s_cum) - 2))
    span = s_cum[idx + 1] - s_cum[idx]
    t = 0.0 if span <= 1e-12 else (s - s_cum[idx]) / span
    return pts[idx] + t * (pts[idx + 1] - pts[idx])


def corner_indices(path: Sequence[Tuple[int, int]], angle_threshold_deg: float) -> List[int]:
    """Indices where the path direction changes by more than ``angle_threshold_deg``."""
    pts = np.asarray(path, dtype=float)
    if len(pts) < 3:
        return []
    deltas = np.diff(pts, axis=0)
    norms = np.linalg.norm(deltas, axis=1)
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])
    cos_thr = math.cos(math.radians(angle_threshold_deg))
    out: List[int] = []
    for i in range(1, len(deltas)):
        if norms[i - 1] < 1e-9 or norms[i] < 1e-9:
            continue
        u0 = deltas[i - 1] / norms[i - 1]
        u1 = deltas[i] / norms[i]
        if float(np.dot(u0, u1)) < cos_thr:
            out.append(i)  # corner between cell i-1 -> i -> i+1; representative cell = i
    return out


def sample_path(
    path: Sequence[Tuple[int, int]],
    spacing: float,
    angle_threshold_deg: float,
    corner_spacing_min: float,
    dedupe_distance: float,
    include_endpoints: bool = True,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Sample a grid path along its arc length while keeping sharp corners.

    Returns ``(samples, stats)`` where each sample holds the interpolated
    continuous coordinate, the nearest grid cell, the arc length and tangent.
    """
    pts = np.asarray(path, dtype=float)
    if len(pts) < 2:
        return [], {"n_samples": 0, "n_corner_samples": 0, "arc_length": 0.0}

    s_cum = _cumulative_arclength(pts)
    total = float(s_cum[-1])
    if total <= 1e-9:
        return [], {"n_samples": 0, "n_corner_samples": 0, "arc_length": 0.0}

    # 1) uniform arc-length stations
    n_stations = int(math.floor(total / spacing))
    stations = [i * spacing for i in range(n_stations + 1)]
    station_meta = [{"source": "uniform"} for _ in stations]
    if include_endpoints and (total - stations[-1]) > 0.25 * spacing:
        stations.append(total)
        station_meta.append({"source": "uniform"})

    # 2) corner stations (retain turning points)
    corner_idx = corner_indices(path, angle_threshold_deg)
    corner_samples = 0
    for ci in corner_idx:
        s_corner = float(s_cum[ci])
        stations.append(s_corner)
        station_meta.append({"source": "corner"})
        corner_samples += 1

    order = np.argsort(np.asarray(stations, dtype=float), kind="stable")
    stations_sorted = [stations[i] for i in order]
    meta_sorted = [station_meta[i] for i in order]

    # 3) dedupe stations that are too close (corner spacing, then generic dedupe)
    kept_pos: List[float] = []
    kept_meta: List[Dict[str, object]] = []
    for s, meta in zip(stations_sorted, meta_sorted):
        if not kept_pos:
            kept_pos.append(s)
            kept_meta.append(meta)
            continue
        min_dist = corner_spacing_min if (meta["source"] == "corner" or kept_meta[-1]["source"] == "corner") else dedupe_distance
        if s - kept_pos[-1] >= min_dist or s >= total - 1e-9:
            kept_pos.append(s)
            kept_meta.append(meta)

    samples: List[Dict[str, object]] = []
    for s, meta in zip(kept_pos, kept_meta):
        xy = _interp_at(pts, s_cum, s)
        grid_xy = (int(round(xy[0])), int(round(xy[1])))
        samples.append(
            {
                "xy": (float(xy[0]), float(xy[1])),
                "grid_xy": grid_xy,
                "arc_length": float(s),
                "is_corner": meta["source"] == "corner",
            }
        )

    stats = {
        "n_samples": len(samples),
        "n_corner_samples": int(sum(1 for s in samples if s["is_corner"])),
        "arc_length": total,
        "raw_cells": int(len(pts)),
    }
    return samples, stats


def path_tangents(path: Sequence[Tuple[int, int]]) -> List[Tuple[float, float]]:
    """Central-difference unit tangents per grid cell (one-sided at the ends)."""
    pts = np.asarray(path, dtype=float)
    n = len(pts)
    tangents: List[Tuple[float, float]] = []
    for i in range(n):
        if n == 1:
            tangents.append((1.0, 0.0))
            continue
        if i == 0:
            d = pts[1] - pts[0]
        elif i == n - 1:
            d = pts[-1] - pts[-2]
        else:
            d = pts[i + 1] - pts[i - 1]
        norm = float(np.linalg.norm(d))
        tangents.append((float(d[0] / norm), float(d[1] / norm)) if norm > 1e-9 else (1.0, 0.0))
    return tangents


def tangent_at_grid(path: Sequence[Tuple[int, int]], cell: Tuple[int, int]) -> Tuple[float, float]:
    """Tangent at the grid cell of a path closest to ``cell``."""
    tangents = path_tangents(path)
    pts = np.asarray(path, dtype=float)
    target = np.asarray(cell, dtype=float)
    idx = int(np.argmin(np.linalg.norm(pts - target, axis=1)))
    return tangents[idx]
