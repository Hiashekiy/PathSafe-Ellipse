"""Shared evaluation loop: raw predictions -> metrics + exact geometry checks."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from .exact_safety import (border_violation_rate, continuity_pass_rate,
                           exact_safety_stats)
from .metrics import summarize_predictions


@torch.no_grad()
def run_evaluation(model, loader, device: torch.device, rho_max: float,
                   max_eval_trajectories: int = 120,
                   max_points_metrics: int = 200000) -> Dict[str, object]:
    """Metrics on raw predictions plus the exact (NumPy) safety verdict."""
    model.eval()
    buckets = {k: [] for k in ("center", "axes", "angle", "S", "onehot", "path",
                               "gt_center", "gt_ab", "gt_angle", "gt_S", "valid", "lidar")}
    exact_pool = []
    for batch in loader:
        b = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
             for k, v in batch.items()}
        pred = model(b["occupancy"], b["path_xy"], b["lidar"] / model.max_range_m,
                     b["geometry"], b["r_scene"], b["mask"], lidar_m=b["lidar"], tau=0.3,
                     map_index=b.get("map_index"))
        m = b["mask"].cpu().numpy()
        for name, tensor in (("center", pred["center"]), ("axes", pred["axes"]),
                             ("angle", pred["angle"]), ("S", pred["shape_matrix"]),
                             ("onehot", pred["direction_onehot"]), ("path", b["path_xy"]),
                             ("gt_ab", b["ab"]), ("gt_angle", b["angle"]), ("gt_S", b["S"]),
                             ("valid", b["valid"]), ("lidar", b["lidar"])):
            buckets[name].append(tensor.detach().cpu().numpy()[m])
        buckets["gt_center"].append(
            (b["path_xy"] + b["center_offset"]).detach().cpu().numpy()[m])
        if len(exact_pool) < max_eval_trajectories:
            n_i = b["mask"].sum(1).cpu().numpy()
            for i in range(b["occupancy"].shape[0]):
                k = int(n_i[i])
                exact_pool.append((
                    b["occupancy"][i, 0].cpu().numpy().astype(np.uint8),
                    float(b["r_scene"][i]),
                    pred["center"][i].detach().cpu().numpy()[:k],
                    pred["shape_matrix"][i].detach().cpu().numpy()[:k],
                    b["valid"][i].detach().cpu().numpy()[:k],
                    tuple(b["meta"][i]["extent"]),
                ))

    cat = lambda key: (np.concatenate(buckets[key]) if buckets[key]
                       else np.zeros((0,)))
    center, gt_center = cat("center"), cat("gt_center")
    if center.shape[0] > max_points_metrics:
        sel = np.random.default_rng(0).choice(center.shape[0], max_points_metrics,
                                             replace=False)
        sel.sort()
        pick = lambda a: a[sel]
    else:
        pick = lambda a: a
    metrics = summarize_predictions(
        pick(center), pick(cat("axes")), pick(cat("angle")), pick(cat("S")),
        pick(gt_center), pick(cat("gt_ab")), pick(cat("gt_angle")), pick(cat("gt_S")),
        pick(cat("valid")), path_xy=pick(cat("path")), ray_onehot=pick(cat("onehot")),
        lidar=pick(cat("lidar")), r_max_m=float(model.r_max_m),
        eps_r_m=float(model.eps_r_m))

    n_safe = n = 0
    viol, cont, center_col = [], [], []
    for occ, rs, c, s, v, extent in exact_pool:
        st = exact_safety_stats(c, s, occ, rs, v)
        n_safe += int(round(st.get("exact_safety_rate", 0.0) * st.get("n", 0)))
        n += int(st.get("n", 0))
        center_col.append(st.get("center_collision_rate", 0.0))
        viol.append(border_violation_rate(c, s, extent, v))
        cont.append(continuity_pass_rate(c, s, v, rho_max=rho_max))
    metrics["exact_safety"] = {
        "n_points": n,
        "raw_safety_rate": (n_safe / n) if n else None,
        "raw_collision_rate": (1.0 - n_safe / n) if n else None,
        "center_collision_rate": float(np.mean(center_col)) if center_col else None,
        "border_violation_rate": float(np.mean(viol)) if viol else None,
        "continuity_pair_rate": float(np.mean([c["pair_rate"] for c in cont])) if cont else None,
        "continuity_path_rate": float(np.mean([c["path_rate"] for c in cont])) if cont else None,
        "n_trajectories_checked": len(exact_pool),
    }
    return metrics
