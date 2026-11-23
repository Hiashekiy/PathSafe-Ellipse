"""Supervised regression losses (plan section 6), V2.1-compatible definitions."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(x.dtype)
    return (x * m).sum() / m.sum().clamp_min(1.0)


def center_loss(pred_offset: torch.Tensor, gt_offset: torch.Tensor,
                mask: torch.Tensor, beta: float = 0.25) -> torch.Tensor:
    """``SmoothL1`` on the metric centre offset ``q - p``."""
    return masked_mean(F.smooth_l1_loss(pred_offset, gt_offset, beta=beta, reduction="none"),
                       mask.unsqueeze(-1).expand_as(pred_offset).bool())


def axes_loss(pred_axes: torch.Tensor, gt_axes: torch.Tensor,
              mask: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
    """``SmoothL1`` on ``log a, log b`` (V2.1 definition)."""
    pred = torch.log(pred_axes.clamp_min(1e-6))
    gt = torch.log(gt_axes.clamp_min(1e-6))
    return masked_mean(F.smooth_l1_loss(pred, gt, beta=beta, reduction="none"),
                       mask.unsqueeze(-1).expand_as(pred).bool())


def double_angle_vector(angle: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.cos(2.0 * angle), torch.sin(2.0 * angle)], dim=-1)


def angle_loss(pred_angle: torch.Tensor, gt_angle: torch.Tensor, gt_axes: torch.Tensor,
               mask: torch.Tensor, use_angle_mask: bool = True,
               aspect_min: float = 1.05) -> torch.Tensor:
    """L1 between double-angle vectors, optionally only on anisotropic ellipses."""
    v_pred = double_angle_vector(pred_angle)
    v_gt = double_angle_vector(gt_angle)
    if use_angle_mask:
        a, b = gt_axes[..., 0], gt_axes[..., 1]
        keep = mask & (a / b.clamp_min(1e-6) > aspect_min)
    else:
        keep = mask
    return masked_mean((v_pred - v_gt).abs(), keep.unsqueeze(-1).expand_as(v_pred).bool())


def regression_losses(pred: dict, gt: dict, mask: torch.Tensor, cfg: dict) -> dict:
    lc = cfg["loss"]
    return {
        "center": center_loss(pred["center_offset"], gt["center_offset"], mask,
                              beta=float(lc["center_beta"])),
        "axes": axes_loss(pred["axes"], gt["ab"], mask, beta=float(lc["axes_beta"])),
        "angle": angle_loss(pred["angle"], gt["angle"], gt["ab"], mask,
                            use_angle_mask=bool(lc.get("use_angle_mask", True)),
                            aspect_min=float(cfg["label"].get("angle_aspect_min", 1.05))),
    }
