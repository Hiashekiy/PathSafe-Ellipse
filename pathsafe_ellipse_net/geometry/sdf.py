"""Signed distance field of the inflated occupancy map (centre-free auxiliary term).

Convention: positive in free space, negative inside an obstacle cell, ``0`` on the
cell-centre boundary.  It only provides a gradient direction for the predicted
centre; the final safety verdict stays with the exact box quadratic test.
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F


def signed_distance_field(occupancy: np.ndarray, r_scene: float) -> np.ndarray:
    """``(H,W)`` float32 SDF in metres for a ``0/1`` occupancy grid."""
    occ = np.asarray(occupancy).astype(bool)
    inside = ndimage.distance_transform_edt(occ) * r_scene
    outside = ndimage.distance_transform_edt(~occ) * r_scene
    return (outside - inside).astype(np.float32)


def sample_sdf(sdf: torch.Tensor, points: torch.Tensor, map_size,
               r_scene: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample ``sdf (B,1,H,W)`` at metric ``points (B,N,2)``.

    Uses the same cell-aligned mapping as the map feature query, so both terms see
    the identical grid convention.
    """
    from ..models.coordinate_query import world_to_grid

    grid = world_to_grid(points, r_scene, sdf.shape[-2:])
    sampled = F.grid_sample(sdf, grid.unsqueeze(1), mode="bilinear",
                            padding_mode="border", align_corners=False)
    return sampled.squeeze(1).squeeze(1)                     # (B,N)
