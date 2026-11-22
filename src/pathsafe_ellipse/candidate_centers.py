"""Ray-based candidate offset centres (section 7)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .local_features import PatchFeatures, supercover_line_cells

UNBIASED_DIRECTION = 16


@dataclass
class CandidateCenter:
    center_xy: Tuple[float, float]
    offset_direction: int      # 0..15 ray index, 16 = unbiased
    offset_ratio: float
    ray_distance_cells: float
    segment_clearance: float
    center_clearance: float

    @property
    def is_unbiased(self) -> bool:
        return self.offset_direction == UNBIASED_DIRECTION


@dataclass
class CandidateSet:
    candidates: List[CandidateCenter]
    rejected: Dict[str, int] = field(default_factory=dict)


def _nearest_cell_clearance(clearance: np.ndarray, xy: Tuple[float, float]) -> float:
    h, w = clearance.shape
    ix = int(round(xy[0]))
    iy = int(round(xy[1]))
    if ix < 0 or iy < 0 or ix >= w or iy >= h:
        return -1.0
    return float(clearance[iy, ix])


def _segment_ok(patch: np.ndarray, clearance: np.ndarray, p0, p1, seg_clearance: float) -> Tuple[bool, float]:
    cells = supercover_line_cells(p0, p1)
    h, w = patch.shape
    min_clr = float("inf")
    for (ix, iy) in cells:
        if ix < 0 or iy < 0 or ix >= w or iy >= h:
            return False, -1.0
        if patch[iy, ix]:
            return False, -1.0
        min_clr = min(min_clr, float(clearance[iy, ix]))
    if min_clr < seg_clearance:
        return False, min_clr
    return True, min_clr


def generate_candidates(
    features: PatchFeatures,
    offset_ratios: List[float],
    ray_margin: float,
    dedupe_distance: float,
    min_center_clearance: float,
    segment_clearance: float,
    patch_size: int,
    ray_max_range: float = 64.0,
) -> CandidateSet:
    """Build offset candidate centres and drop geometrically invalid ones."""
    patch = features.patch
    clearance = features.clearance
    p0 = features.local_xy
    lo, hi = -0.5 + 1.0, patch_size - 0.5 - 1.0
    rejected: Dict[str, int] = {
        "patch_invalid": 0,
        "candidate_center_occupied": 0,
        "candidate_segment_collision": 0,
        "candidate_duplicate": 0,
    }

    raw_candidates: List[CandidateCenter] = []

    # Unbiased candidate is always kept (it is the fallback label).
    unbiased = CandidateCenter(
        center_xy=p0,
        offset_direction=UNBIASED_DIRECTION,
        offset_ratio=0.0,
        ray_distance_cells=0.0,
        segment_clearance=float(clearance[int(p0[1]), int(p0[0])]),
        center_clearance=float(clearance[int(p0[1]), int(p0[0])]),
    )
    raw_candidates.append(unbiased)

    for k in range(len(features.ray_distances)):
        raw_d = float(features.ray_distances_raw[k])
        usable = max(0.0, min(raw_d, ray_max_range) - ray_margin)
        if usable <= 0.5:
            continue
        u = features.ray_directions[k].astype(float)
        for lam in offset_ratios:
            if lam <= 0.0:
                continue
            xy = (p0[0] + lam * usable * u[0], p0[1] + lam * usable * u[1])
            if not (lo <= xy[0] <= hi and lo <= xy[1] <= hi):
                rejected["patch_invalid"] += 1
                continue
            center_clr = _nearest_cell_clearance(clearance, xy)
            if center_clr < min_center_clearance:
                rejected["candidate_center_occupied"] += 1
                continue
            ok, seg_clr = _segment_ok(patch, clearance, p0, xy, segment_clearance)
            if not ok:
                rejected["candidate_segment_collision"] += 1
                continue
            raw_candidates.append(
                CandidateCenter(
                    center_xy=xy,
                    offset_direction=int(k),
                    offset_ratio=float(lam),
                    ray_distance_cells=raw_d,
                    segment_clearance=float(seg_clr),
                    center_clearance=float(center_clr),
                )
            )

    # Deduplicate by spatial distance, keeping the smaller offset ratio / ray index
    # (and always keeping the unbiased candidate).
    raw_candidates.sort(key=lambda c: (c.offset_ratio, c.offset_direction))
    kept: List[CandidateCenter] = []
    for cand in raw_candidates:
        duplicate = False
        for other in kept:
            if cand.is_unbiased and other.is_unbiased:
                duplicate = True
                break
            dist = float(np.hypot(cand.center_xy[0] - other.center_xy[0],
                                  cand.center_xy[1] - other.center_xy[1]))
            if dist < dedupe_distance:
                duplicate = True
                break
        if duplicate:
            rejected["candidate_duplicate"] += 1
            continue
        kept.append(cand)

    # Keep unbiased first for deterministic ordering.
    kept.sort(key=lambda c: (0 if c.is_unbiased else 1, c.offset_ratio, c.offset_direction))
    return CandidateSet(candidates=kept, rejected=rejected)
