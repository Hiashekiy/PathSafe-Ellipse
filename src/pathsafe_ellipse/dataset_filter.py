"""Dataset filtering (section 10.1) with per-reason accounting."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

REASON_ORDER = (
    "map_unparsed",
    "point_not_free",
    "patch_invalid",
    "ray_features_invalid",
    "candidate_center_occupied",
    "candidate_segment_collision",
    "iris_solver_failed",
    "ellipse_degenerate",
    "path_point_outside_ellipse",
    "ellipse_collision",
    "non_finite_values",
)


def filter_arrays(arrays: Dict[str, np.ndarray], cfg: Dict) -> Tuple[Dict[str, np.ndarray], Dict[str, int], int]:
    """Apply the acceptance rules to a per-map sample dict.

    Returns ``(filtered, mask_by_reason, n_input)``.  The builder already drops
    rejected candidates; this function re-validates the stored labels so a bad
    record can never reach the splits.  Size validity is the single fixed rule:
    ``min_semi_axis <= short semi-axis`` and ``long semi-axis <= max_semi_axis``.
    """
    n = int(len(arrays["label_valid"]))
    reasons = {key: 0 for key in REASON_ORDER}

    finite_fields = ("ray_distances", "ray_distances_raw", "path_tangent", "ellipse_ab",
                     "ellipse_angle", "ellipse_center_offset", "ellipse_area",
                     "point_inside_metric", "offset_ratio")
    finite_ok = np.ones(n, dtype=bool)
    for field in finite_fields:
        finite_ok &= np.all(np.isfinite(arrays[field].reshape(n, -1)), axis=1)

    ab = arrays["ellipse_ab"].reshape(n, 2)
    angle = arrays["ellipse_angle"].reshape(n, 2)
    min_axis_m = float(cfg["label"]["min_semi_axis"])
    max_axis_m = float(cfg["label"]["max_semi_axis"])
    eta = float(cfg["label"]["point_inside_eta"])

    checks: List[Tuple[str, np.ndarray]] = [
        ("non_finite_values", finite_ok),
        ("ellipse_degenerate",
         (ab[:, 1] >= min_axis_m - 1e-9) & (ab[:, 0] <= max_axis_m * (1.0 + 1e-9))
         & (ab[:, 0] >= ab[:, 1] - 1e-9)),
        ("path_point_outside_ellipse", arrays["point_inside_metric"] <= eta),
        ("ray_features_invalid",
         (np.all(arrays["ray_distances"] >= 0.0, axis=1))
         & (np.all(arrays["ray_distances"] <= 1.0, axis=1))
         & (np.all(np.abs(np.sum(angle ** 2, axis=1) - 1.0) < 1e-4))),
        ("patch_invalid",
         arrays["map_patch"].reshape(n, -1)[:, 64 * 128 + 64] == 0 if n else np.zeros(0, dtype=bool)),
        ("label_valid", arrays["label_valid"].astype(bool)),
    ]

    mask = np.ones(n, dtype=bool)
    for name, passing in checks:
        newly_failed = mask & (~passing)
        reasons[name] = reasons.get(name, 0) + int(np.count_nonzero(newly_failed))
        mask &= passing

    filtered = {key: value[mask] for key, value in arrays.items()}
    return filtered, reasons, n
