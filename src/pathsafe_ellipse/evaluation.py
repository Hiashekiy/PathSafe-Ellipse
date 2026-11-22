"""Prediction metrics for the ellipse network (geometric + regression)."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .ellipse_validation import ellipse_obstacle_intersections, point_metric
from .torch_dataset import SplitData


def rebuild_L_cells(a_m: np.ndarray, b_m: np.ndarray, angle_rad: np.ndarray,
                    resolution: float) -> np.ndarray:
    """``R diag(a, b) R^T`` in cells for a batch of ellipses."""
    ca = np.cos(angle_rad)[:, None, None]
    sa = np.sin(angle_rad)[:, None, None]
    R = np.concatenate([np.concatenate([ca, -sa], axis=2),
                        np.concatenate([sa, ca], axis=2)], axis=1)
    D = np.zeros((len(a_m), 2, 2))
    D[:, 0, 0] = a_m / resolution
    D[:, 1, 1] = b_m / resolution
    return R @ D @ np.transpose(R, (0, 2, 1))


def _inside_mask(points: np.ndarray, center_m: np.ndarray, ab_m: np.ndarray,
                 angle_rad: np.ndarray, resolution: float) -> np.ndarray:
    ca = np.cos(angle_rad)[:, None]
    sa = np.sin(angle_rad)[:, None]
    a_cell = np.maximum(ab_m[:, 0] / resolution, 1e-6)[:, None]
    b_cell = np.maximum(ab_m[:, 1] / resolution, 1e-6)[:, None]
    d = points[None, :, :] - (center_m / resolution)[:, None, :]
    u = d[:, :, 0] * ca + d[:, :, 1] * sa
    v = -d[:, :, 0] * sa + d[:, :, 1] * ca
    return (u / a_cell) ** 2 + (v / b_cell) ** 2 <= 1.0


def ellipse_iou_grid(pred, gt, resolution: float, size: int = 128) -> np.ndarray:
    """IoU of predicted vs ground-truth ellipses rasterised on the patch grid."""
    ys, xs = np.mgrid[0:size, 0:size]
    pts = np.stack([xs.ravel() - 64.0, ys.ravel() - 64.0], axis=1)
    m_pred = _inside_mask(pts, pred[0], pred[1], pred[2], resolution)
    m_gt = _inside_mask(pts, gt[0], gt[1], gt[2], resolution)
    inter = np.sum(m_pred & m_gt, axis=1)
    union = np.sum(m_pred | m_gt, axis=1)
    return inter / np.maximum(union, 1)


def angle_error_deg(pred_angle: np.ndarray, gt_angle: np.ndarray) -> np.ndarray:
    """Orientation error folded into [0, 90] deg (ellipse pi-periodicity)."""
    diff = np.mod(np.abs(pred_angle - gt_angle), np.pi)
    return np.degrees(np.minimum(diff, np.pi - diff))


def _patch_of(data: SplitData, i: int) -> np.ndarray:
    return np.unpackbits(data.packed[i], axis=1).reshape(128, 128)


@torch.no_grad()
def evaluate_model(
    model,
    data: SplitData,
    device: torch.device,
    resolution: float,
    batch_size: int = 256,
    max_samples: Optional[int] = None,
    eta: float = 0.9,
    safety_check: bool = True,
    max_safety_samples: int = 1200,
    criterion=None,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    n = data.n if max_samples is None else min(data.n, max_samples)
    idx_all = np.arange(n)

    sums = {k: 0.0 for k in ("center_err_m", "a_err_m", "b_err_m", "iou", "rho_pred", "loss")}
    angle_err_sum = 0.0
    n_angle = 0
    n_total = 0
    rho_ok = 0
    safety_checked = 0
    safety_ok = 0
    both_ok = 0

    for start in range(0, n, batch_size):
        idx = idx_all[start:start + batch_size]
        batch = data.batch(idx)
        tensors = {k: torch.from_numpy(batch[k]).to(device)
                   for k in ("patch", "rays", "hits", "tangent")}
        center, ab, angle, sincos, logit = model.predict_ellipse(tensors)
        center = center.cpu().numpy()
        ab = np.sort(ab.cpu().numpy(), axis=1)[:, ::-1]
        angle = angle.cpu().numpy()
        sincos_np = sincos.cpu().numpy()
        logit_np = logit.cpu().numpy()

        gt_center = batch["center_offset"]
        gt_ab = np.sort(batch["ab"], axis=1)[:, ::-1]
        gt_angle = np.arctan2(batch["sincos"][:, 0], batch["sincos"][:, 1])
        gt_valid = batch["angle_valid"] > 0.5

        if criterion is not None:
            preds = {"center": torch.from_numpy(center).to(device),
                     "log_ab": torch.from_numpy(np.log(np.maximum(ab, 1e-3))).to(device),
                     "sincos": torch.from_numpy(sincos_np).to(device),
                     "angle_logit": torch.from_numpy(logit_np).to(device)}
            targets = {k: torch.from_numpy(batch[k]).to(device)
                       for k in ("center_offset", "ab", "sincos", "angle_valid")}
            loss, _ = criterion(preds, targets)
            sums["loss"] += float(loss) * len(idx)

        ang_err = angle_error_deg(angle, gt_angle)
        sums["center_err_m"] += float(np.abs(center - gt_center).sum())
        sums["a_err_m"] += float(np.abs(ab[:, 0] - gt_ab[:, 0]).sum())
        sums["b_err_m"] += float(np.abs(ab[:, 1] - gt_ab[:, 1]).sum())
        angle_err_sum += float(ang_err[gt_valid].sum())
        n_angle += int(gt_valid.sum())
        sums["iou"] += float(ellipse_iou_grid((center, ab, angle),
                                             (gt_center, gt_ab, gt_angle), resolution).sum())

        L_pred = rebuild_L_cells(ab[:, 0], ab[:, 1], angle, resolution)
        centers_cells = np.stack([64.0 + center[:, 0] / resolution,
                                  64.0 + center[:, 1] / resolution], axis=1)
        for bi in range(len(idx)):
            rho = point_metric(L_pred[bi], centers_cells[bi], np.array([64.0, 64.0]))
            sums["rho_pred"] += rho
            inside = rho <= eta
            rho_ok += int(inside)
            if safety_check and safety_checked < max_safety_samples:
                safety_checked += 1
                collides, _, _ = ellipse_obstacle_intersections(
                    L_pred[bi], centers_cells[bi], _patch_of(data, int(idx[bi])))
                safe = not collides
                safety_ok += int(safe)
                both_ok += int(safe and inside)
        n_total += len(idx)

    if was_training:
        model.train()
    out = {k: v / max(n_total, 1) for k, v in sums.items()}
    out["angle_err_deg"] = angle_err_sum / max(n_angle, 1)
    out["n_samples"] = n_total
    out["n_angle_valid"] = n_angle
    out["rho_ok_rate"] = rho_ok / max(n_total, 1)
    out["safety_checked"] = safety_checked
    out["safety_rate"] = safety_ok / max(safety_checked, 1)
    out["safe_and_contains_rate"] = both_ok / max(safety_checked, 1)
    return out


class EllipseCriterion:
    """Weighted regression loss for the ellipse parameterisation."""

    def __init__(self, w_center: float = 1.0, w_axes: float = 1.0,
                 w_angle: float = 0.5, w_angle_valid: float = 0.1):
        self.w_center = w_center
        self.w_axes = w_axes
        self.w_angle = w_angle
        self.w_angle_valid = w_angle_valid

    def __call__(self, preds: Dict[str, torch.Tensor],
                 targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        center_loss = torch.nn.functional.smooth_l1_loss(preds["center"], targets["center_offset"])
        log_ab_gt = torch.log(torch.clamp(targets["ab"], min=1e-3))
        axes_loss = torch.nn.functional.smooth_l1_loss(preds["log_ab"], log_ab_gt)
        valid = targets["angle_valid"] > 0.5
        if valid.any():
            dot = torch.sum(preds["sincos"][valid] * targets["sincos"][valid], dim=1).abs()
            angle_loss = torch.mean(1.0 - dot)
        else:
            angle_loss = torch.zeros((), device=preds["center"].device)
        av_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            preds["angle_logit"], targets["angle_valid"])
        total = (self.w_center * center_loss + self.w_axes * axes_loss
                 + self.w_angle * angle_loss + self.w_angle_valid * av_loss)
        return total, {
            "center": float(center_loss.detach()),
            "axes": float(axes_loss.detach()),
            "angle": float(angle_loss.detach()),
            "angle_valid": float(av_loss.detach()),
        }
