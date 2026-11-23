"""Ellipse decoding/encoding in the label-generator's convention."""
from __future__ import annotations

import math
from typing import Dict

import torch

from .box_quadratic_torch import shape_matrix


def ray_directions(n_rays: int = 16, device=None, dtype=torch.float32) -> torch.Tensor:
    """``u_j = [cos(2 pi j / R), sin(2 pi j / R)]``, shape ``(R, 2)``."""
    j = torch.arange(n_rays, device=device, dtype=dtype)
    ang = 2.0 * math.pi * j / n_rays
    return torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)


def ray_limits(lidar_ranges: torch.Tensor, r_max_m: float, eps_r_m: float,
               max_range_m: float) -> torch.Tensor:
    """Per-ray usable distance ``r_ij = clamp(d_ij - eps_r, 0, R_max)``.

    ``lidar_ranges`` may be metres or already normalised; the caller passes metres
    (``lidar_m``) so the centre decoder stays metric.
    """
    return (lidar_ranges - eps_r_m).clamp(0.0, min(r_max_m, max_range_m))


def decode_center(path_xy: torch.Tensor, lidar_m: torch.Tensor,
                  direction_onehot: torch.Tensor, ratio: torch.Tensor,
                  r_max_m: float, eps_r_m: float, max_range_m: float) -> torch.Tensor:
    """``q_hat = p + alpha * sum_j y_j r_ij u_j`` (plan section 4.3)."""
    r = ray_limits(lidar_m, r_max_m, eps_r_m, max_range_m)          # (B,N,R)
    u = ray_directions(lidar_m.shape[-1], lidar_m.device, lidar_m.dtype)
    # one-hot selects exactly one ray: sum_j y_j r_ij u_j == r_{i,argmax} * u_{argmax}
    selected = (direction_onehot * r).sum(-1, keepdim=True)                     # (B,N,1)
    direction = (direction_onehot.unsqueeze(-1) * u.view(1, 1, -1, 2)).sum(2)   # (B,N,2)
    return path_xy + ratio.unsqueeze(-1) * selected * direction


def decode_axes(z_shape: torch.Tensor, a_min: float, a_max: float):
    """Bounded semi-axes: ``a in [a_min, a_max]``, ``b in [a_min, a]``."""
    za = z_shape[..., 0]
    zb = z_shape[..., 1]
    a = a_min + (a_max - a_min) * torch.sigmoid(za)
    b = a_min + (a - a_min) * torch.sigmoid(zb)
    return torch.stack([a, b], dim=-1)


def decode_angle(z_shape: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Double-angle vector normalised then halved (plan section 5.2)."""
    v = z_shape[..., 2:4]
    v = v / (v.norm(dim=-1, keepdim=True) + eps)
    return 0.5 * torch.atan2(v[..., 1], v[..., 0])


def decode_ellipse(z_shape: torch.Tensor, a_min: float, a_max: float) -> Dict[str, torch.Tensor]:
    axes = decode_axes(z_shape, a_min, a_max)
    theta = decode_angle(z_shape)
    S = shape_matrix(axes, theta)
    return {"axes": axes, "angle": theta, "shape_matrix": S}
