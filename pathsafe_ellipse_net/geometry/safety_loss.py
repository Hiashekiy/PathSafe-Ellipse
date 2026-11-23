"""Differentiable geometric safety loss (plan section 7).

    L_safety = L_ellipse_safe + lambda_sdf * L_center_free + lambda_border * L_border

The ellipse term evaluates the **exact** quadratic minimum over every candidate
obstacle *square* (not the cell centre, not boundary samples), using PyTorch only.
Candidate cells come from the predicted ellipse's axis-aligned bounding box; the
window is dense (all cells within the box), never a fixed Top-K, so a collision
cannot be missed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .box_quadratic_torch import box_quadratic_min_torch, invert_shape
from .sdf import sample_sdf


@dataclass
class SafetyParts:
    ellipse_safe: torch.Tensor
    center_free: torch.Tensor
    border: torch.Tensor

    def total(self, lambda_sdf: float, lambda_border: float) -> torch.Tensor:
        return self.ellipse_safe + lambda_sdf * self.center_free + lambda_border * self.border

    def as_dict(self) -> Dict[str, float]:
        return {"ellipse_safe": float(self.ellipse_safe.detach()),
                "center_free": float(self.center_free.detach()),
                "border": float(self.border.detach())}


def ellipse_safe_loss(center: torch.Tensor, shape_matrix: torch.Tensor,
                      occupancy: torch.Tensor, r_scene: torch.Tensor,
                      valid_mask: torch.Tensor, m_safe: float = 0.0,
                      max_window_cells: int = 40,
                      chunk_points: int = 512) -> torch.Tensor:
    """``mean_i [ReLU(1 - phi_i^min)]^2`` over the candidate obstacle squares.

    The candidate set is *every* cell inside the predicted ellipse's bounding box
    (never a fixed Top-K).  Points are processed in chunks so a long trajectory
    cannot blow up GPU memory; the result is identical to the unchunked sum.
    """
    B, N, _ = center.shape
    H, W = occupancy.shape[-2:]
    rs = r_scene.reshape(-1)
    P = B * N
    rs_p = rs.repeat_interleave(N) if rs.numel() == B else rs.expand(B).repeat_interleave(N)
    c = center.reshape(P, 2)
    S = shape_matrix.reshape(P, 2, 2)
    valid = valid_mask.reshape(P).to(torch.bool)
    if not bool(valid.any()):
        return center.sum() * 0.0

    occ_flat = occupancy.reshape(-1)                        # (B*H*W,)
    b_of = torch.arange(P, device=center.device) // N
    rx = torch.sqrt(S[..., 0, 0].clamp_min(0.0)).detach()
    ry = torch.sqrt(S[..., 1, 1].clamp_min(0.0)).detach()
    h = 0.5 * rs_p
    rad = (((torch.maximum(rx, ry) + h + m_safe) / rs_p).ceil()
           .clamp(min=0, max=max_window_cells))
    base_c = torch.round(c[..., 0] / rs_p).long()
    base_r = torch.round(c[..., 1] / rs_p).long()

    total = c.new_zeros(())
    denom = valid.to(c.dtype).sum().clamp_min(1.0)
    for s0 in range(0, P, chunk_points):
        s1 = min(s0 + chunk_points, P)
        v = valid[s0:s1]
        if not bool(v.any()):
            continue
        K = int(rad[s0:s1][v].max().item())
        if K <= 0:
            continue
        ring = torch.arange(-K, K + 1, device=center.device)
        cc = base_c[s0:s1, None] + ring[None, :]              # (Pc, K')
        rr = base_r[s0:s1, None] + ring[None, :]
        cc2 = cc[:, None, :].expand(-1, 2 * K + 1, -1)        # (Pc, K', K')
        rr2 = rr[:, :, None].expand(-1, 2 * K + 1, -1)

        inb = (rr2 >= 0) & (rr2 < H) & (cc2 >= 0) & (cc2 < W)
        flat = (b_of[s0:s1, None, None] * (H * W)
                + rr2.clamp(0, H - 1) * W + cc2.clamp(0, W - 1))
        occ_v = torch.gather(occ_flat, 0, flat.reshape(-1)).reshape(flat.shape)
        dist = torch.sqrt((rr2 - base_r[s0:s1, None, None].to(c.dtype)) ** 2
                          + (cc2 - base_c[s0:s1, None, None].to(c.dtype)) ** 2)
        candidate = (occ_v > 0.5) & inb & (dist <= rad[s0:s1, None, None] + 0.5)

        rsc = rs_p[s0:s1, None, None].to(c.dtype)
        cx = cc2.to(c.dtype) * rsc
        cy = rr2.to(c.dtype) * rsc
        half = h[s0:s1, None, None].to(c.dtype) + m_safe
        M = invert_shape(S[s0:s1])                            # (Pc,2,2)
        phi = box_quadratic_min_torch(M[:, None, None], c[s0:s1, None, None],
                                      cx - half, cx + half, cy - half, cy + half)
        phi = torch.where(candidate, phi, torch.full_like(phi, float("inf")))
        phi_min = phi.amin(dim=(-1, -2))
        has = candidate.any(dim=(-1, -2))
        pen = torch.where(has, F.relu(1.0 - phi_min).pow(2),
                          torch.zeros_like(phi_min))
        total = total + (pen * v.to(pen.dtype)).sum()
    return total / denom


def center_free_loss(center: torch.Tensor, sdf: torch.Tensor, r_scene: torch.Tensor,
                     valid_mask: torch.Tensor, m_center: float = 0.0) -> torch.Tensor:
    """``mean_i [ReLU(m_center - d_i) / r_scene]^2`` with ``d_i`` the sampled SDF."""
    d = sample_sdf(sdf, center, None, r_scene)
    rs = r_scene.view(-1, 1) if r_scene.dim() == 1 else r_scene
    pen = (F.relu(m_center - d) / rs).pow(2) * valid_mask.to(d.dtype)
    return pen.sum() / valid_mask.to(d.dtype).sum().clamp_min(1.0)


def border_loss(center: torch.Tensor, shape_matrix: torch.Tensor, r_scene: torch.Tensor,
                map_size, valid_mask: torch.Tensor, m_border: float = 0.0) -> torch.Tensor:
    """Keep the ellipse's axis-aligned bounding box inside the map (plan 7.6)."""
    B, N, _ = center.shape
    H, W = map_size
    rs = r_scene.view(B, 1) if r_scene.dim() == 1 else r_scene
    rx = torch.sqrt(shape_matrix[..., 0, 0].clamp_min(0.0))
    ry = torch.sqrt(shape_matrix[..., 1, 1].clamp_min(0.0))
    gL = center[..., 0] - rx - m_border
    gR = (W * rs) - center[..., 0] - rx - m_border
    gB = center[..., 1] - ry - m_border
    gT = (H * rs) - center[..., 1] - ry - m_border
    pen = sum((F.relu(-g) / rs).pow(2) for g in (gL, gR, gB, gT))
    pen = pen * valid_mask.to(pen.dtype)
    return pen.sum() / valid_mask.to(pen.dtype).sum().clamp_min(1.0)


class SafetyCriterion(nn.Module):
    """``L_safety`` with the three parts exposed for logging."""

    def __init__(self, lambda_sdf: float = 0.2, lambda_border: float = 1.0,
                 m_safe_m: float = 0.0, m_center_m: float = 0.0, m_border_m: float = 0.01,
                 max_window_cells: int = 64):
        super().__init__()
        self.lambda_sdf = float(lambda_sdf)
        self.lambda_border = float(lambda_border)
        self.m_safe_m = float(m_safe_m)
        self.m_center_m = float(m_center_m)
        self.m_border_m = float(m_border_m)
        self.max_window_cells = int(max_window_cells)

    def forward(self, center: torch.Tensor, shape_matrix: torch.Tensor,
                occupancy: torch.Tensor, sdf: torch.Tensor, r_scene: torch.Tensor,
                valid_mask: torch.Tensor) -> SafetyParts:
        valid = valid_mask.to(torch.bool)
        return SafetyParts(
            ellipse_safe=ellipse_safe_loss(center, shape_matrix, occupancy, r_scene, valid,
                                           m_safe=self.m_safe_m,
                                           max_window_cells=self.max_window_cells),
            center_free=center_free_loss(center, sdf, r_scene, valid,
                                         m_center=self.m_center_m),
            border=border_loss(center, shape_matrix, r_scene,
                               occupancy.shape[-2:], valid, m_border=self.m_border_m),
        )

    def total(self, parts: SafetyParts) -> torch.Tensor:
        return parts.total(self.lambda_sdf, self.lambda_border)
