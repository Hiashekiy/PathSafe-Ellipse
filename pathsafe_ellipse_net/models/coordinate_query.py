"""Coordinate query: sample the multi-scale map features at metric path points.

Grid convention (must match the label generator, ``scene.py``):

* cell ``(row=i, col=j)`` has its centre at ``x = j * r_scene``, ``y = i * r_scene``;
* array indexing is ``[row=y, col=x]``.

With ``align_corners=False`` the pixel ``j`` centre sits at normalized coordinate
``2 * (j + 0.5) / W - 1``, so a metric point is mapped with the extra half-cell
shift below.  Training and inference both go through :func:`world_to_grid`.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F


def world_to_grid(points: torch.Tensor, r_scene: torch.Tensor, size) -> torch.Tensor:
    """Metric ``(x, y)`` -> ``grid_sample`` coordinates in ``[-1, 1]``.

    ``points`` must be ``(B, N, 2)`` and ``r_scene`` a scalar or ``(B,)``.
    """
    H, W = size
    if points.dim() == 2:
        points = points.unsqueeze(0)
    rs = r_scene.reshape(-1)
    rs = rs.reshape(-1, 1, 1) if rs.numel() > 1 else rs.reshape(1, 1, 1)
    gx = 2.0 * (points[..., 0:1] / rs + 0.5) / W - 1.0
    gy = 2.0 * (points[..., 1:2] / rs + 0.5) / H - 1.0
    return torch.cat([gx, gy], dim=-1)


def sample_features(feats: List[torch.Tensor], points: torch.Tensor,
                    r_scene: torch.Tensor, map_size) -> torch.Tensor:
    """Bilinear sample each scale and concatenate -> ``(B,N,sum(C))``."""
    grid = world_to_grid(points, r_scene, map_size)
    outs = []
    for f in feats:
        sampled = F.grid_sample(f, grid.unsqueeze(1), mode="bilinear",
                                padding_mode="border", align_corners=False)
        outs.append(sampled.squeeze(2).transpose(1, 2))
    return torch.cat(outs, dim=-1)
