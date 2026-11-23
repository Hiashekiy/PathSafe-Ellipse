"""Evaluation loop: argmax decoding, tensor diagnostics, exact (NumPy) verification."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from .metrics import (batch_diagnostics, ellipse_iou_np, exact_trajectory, quantiles,
                      summarize)


@torch.no_grad()
def run_evaluation(model, loss_fn, loader, device: torch.device, cfg: Dict,
                   exact: bool = True, max_exact_traj: int = 0,
                   max_iou_points: int = 1500, amp: bool = False,
                   select: str = "argmax") -> Dict[str, object]:
    model.eval()
    acc: Dict[str, List[np.ndarray]] = {}
    traj_rows: List[Dict[str, object]] = []
    n_batch_traj = 0
    n_exact = 0
    # map-sized / ragged fields are not stacked (only used per batch below)
    SKIP = {"mask", "occupancy"}
    for batch in loader:
        batch = _to_device(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            out = model(batch, tau=1.0)
        out = {k: (v.float() if torch.is_tensor(v) else v) for k, v in out.items()}
        diag = batch_diagnostics(out, batch, cfg, loss_fn, select=select)
        n_batch_traj += diag["mask"].shape[0]
        for k, v in diag.items():
            if k in SKIP or v.dtype == object:
                continue
            acc.setdefault(k, []).append(v)

        if exact:
            counts = diag["mask"].sum(axis=1)
            starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
            for i in range(diag["mask"].shape[0]):
                if max_exact_traj and n_exact >= max_exact_traj:
                    break
                s, e = int(starts[i]), int(starts[i] + counts[i])
                if e - s < 2:
                    continue
                occ = diag["occupancy"][i, 0].astype(np.uint8)
                res = float(diag["r_scene"][i])
                row = exact_trajectory(diag["center"][s:e], diag["a_raw"][s:e],
                                       diag["b_raw"][s:e], diag["theta_raw"][s:e],
                                       occ, res, rho_max=float(cfg["label"]["rho_max"]))
                row["corridor_ok"] = bool(row["safe"].all() and row["pair_ok"].all()
                                          if len(row["pair_ok"]) else row["safe"].all())
                row["gt"] = {
                    "center": diag["gt_center"][s:e], "ab": diag["gt_ab"][s:e],
                    "angle": diag["gt_angle"][s:e], "valid": diag["gt_valid"][s:e] > 0.5,
                }
                row["pred"] = {"center": diag["center"][s:e], "a": diag["a_raw"][s:e],
                               "b": diag["b_raw"][s:e], "theta": diag["theta_raw"][s:e]}
                traj_rows.append(row)
                n_exact += 1
    merged = {k: np.concatenate(v, axis=0) for k, v in acc.items()}
    merged["mask"] = np.zeros((n_batch_traj, 1), dtype=bool)

    exact_summary: Optional[Dict[str, object]] = None
    if exact and traj_rows:
        exact_summary = {
            "safe": np.concatenate([r["safe"] for r in traj_rows]),
            "center_blocked": np.concatenate([r["center_blocked"] for r in traj_rows]),
            "min_quadratic": np.concatenate([r["min_quadratic"] for r in traj_rows]),
            "pair_ok": np.concatenate([r["pair_ok"] for r in traj_rows]),
            "pair_cover": np.concatenate([r["pair_cover"] for r in traj_rows]),
            "border": np.concatenate([r["border"] for r in traj_rows]),
            "corridor_ok": np.asarray([r["corridor_ok"] for r in traj_rows], dtype=bool),
        }
    summary = summarize(merged, exact_summary, cfg)
    summary["n_trajectories"] = len(traj_rows) if exact else summary["n_trajectories"]

    # GT ellipse IoU on a bounded subset (rasterised, diagnostic only)
    ious = []
    for r in traj_rows:
        gt, pred = r["gt"], r["pred"]
        for i in np.flatnonzero(gt["valid"]):
            if len(ious) >= max_iou_points:
                break
            ious.append(ellipse_iou_np(pred["center"][i], pred["a"][i], pred["b"][i],
                                       pred["theta"][i], gt["center"][i],
                                       gt["ab"][i, 0], gt["ab"][i, 1], gt["angle"][i]))
    summary["gt_ellipse_iou"] = quantiles(ious)
    summary["trajectories"] = traj_rows
    return summary


def _to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out
