"""Coordinate queries: metric (x, y) -> ``grid_sample`` coordinates and bilinear SDF.

Conventions (identical everywhere in ``ellipsenet`` and to the data generator):

* cell ``(row=i, col=j)`` centre sits at metric ``(x, y) = (j*r, i*r)``;
* arrays are indexed ``[row=y, col=x]``;
* ``align_corners=False``: pixel ``j`` centre is at normalized ``2*(j+0.5)/W - 1``;
* ``r_scene`` may differ per sample of the batch, so it is passed as ``(B,)``.

``sample_sdf`` additionally enforces "outside the map == blocked" instead of
letting the border padding report free space (plan section 3).
"""
from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn.functional as F

from ..data.sdf import SDF_OUTSIDE_M


def world_to_grid(points: torch.Tensor, r_scene: torch.Tensor, size: Tuple[int, int]
                  ) -> torch.Tensor:
    """Metric ``(x, y)`` -> normalized grid coordinates.

    ``points`` is ``(B, ..., 2)``; ``r_scene`` a scalar or ``(B,)`` tensor.
    """
    H, W = int(size[0]), int(size[1])
    if points.dim() < 2:
        raise ValueError(f"points must be (B, ..., 2), got {tuple(points.shape)}")
    rs = torch.as_tensor(r_scene, dtype=points.dtype, device=points.device).reshape(-1)
    extra = points.dim() - 1                     # one singleton per trailing dim
    rs = rs.reshape(-1, *([1] * extra)) if rs.numel() > 1 else rs.reshape(1, *([1] * extra))
    gx = 2.0 * (points[..., 0:1] / rs + 0.5) / W - 1.0
    gy = 2.0 * (points[..., 1:2] / rs + 0.5) / H - 1.0
    return torch.cat([gx, gy], dim=-1)


def inside_map(points: torch.Tensor, r_scene: torch.Tensor, size: Tuple[int, int]
               ) -> torch.Tensor:
    """Bool mask: the point lies within the map's cell-centre bounds (plan 3)."""
    H, W = int(size[0]), int(size[1])
    rs = torch.as_tensor(r_scene, dtype=points.dtype, device=points.device).reshape(-1)
    extra = points.dim() - 1
    rs = rs.reshape(-1, *([1] * extra)) if rs.numel() > 1 else rs.reshape(1, *([1] * extra))
    gx = points[..., 0:1] / rs
    gy = points[..., 1:2] / rs
    m = (gx >= 0.0) & (gx <= W - 1.0) & (gy >= 0.0) & (gy <= H - 1.0)
    return m.squeeze(-1)


def sample_feature(feat: torch.Tensor, points: torch.Tensor, r_scene: torch.Tensor
                   ) -> torch.Tensor:
    """Bilinear sample ``feat (B,C,H,W)`` at ``points (B, ..., 2)`` -> ``(B, ..., C)``."""
    shape = points.shape
    flat = points.reshape(shape[0], -1, 2)
    grid = world_to_grid(flat, r_scene, (feat.shape[-2], feat.shape[-1]))
    # grid_sample wants (B, 1, P, 2) -> (B, C, 1, P)
    out = F.grid_sample(feat, grid.unsqueeze(2), mode="bilinear",
                        padding_mode="border", align_corners=False)
    out = out.squeeze(2).transpose(1, 2)                     # (B, P, C)
    return out.reshape(*shape[:-1], feat.shape[1])


def sample_sdf(sdf: torch.Tensor, points: torch.Tensor, r_scene: torch.Tensor
               ) -> torch.Tensor:
    """Bilinear SDF in metres; positions outside the map return ``SDF_OUTSIDE_M``.

    Outside positions must not inherit the clamped border value, otherwise the
    network could hide an ellipse outside the map for free.
    """
    shape = points.shape
    flat = points.reshape(shape[0], -1, 2)
    inside = inside_map(flat, r_scene, (sdf.shape[-2], sdf.shape[-1]))
    val = sample_feature(sdf, flat, r_scene).squeeze(-1)     # (B, P)
    out_val = torch.where(inside, val, torch.full_like(val, float(SDF_OUTSIDE_M)))
    return out_val.reshape(shape[:-1])


def ellipse_rotation(theta: torch.Tensor) -> torch.Tensor:
    """``(B, ..., 2, 2)`` rotation matrices ``R(theta)`` (columns = ellipse axes)."""
    c, s = torch.cos(theta), torch.sin(theta)
    row0 = torch.stack([c, -s], dim=-1)
    row1 = torch.stack([s, c], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def unit_ellipse_template(radii: Sequence[float], counts: Sequence[int],
                          device=None, dtype=torch.float32) -> torch.Tensor:
    """Fixed sample template inside/on the unit circle, ``(K, 2)``.

    ``radii`` are absolute radii in (0, 1]; a radius of ``0`` contributes the
    centre only.  Sample order is deterministic (radii ascending, then angle), which
    keeps the loss reproducible.
    """
    pts = []
    for r, n in zip(radii, counts):
        r = float(r)
        n = int(n)
        if r <= 0.0 or n <= 1:
            pts.append([[0.0, 0.0]])
            continue
        ang = 2.0 * torch.pi * torch.arange(n, device=device, dtype=dtype) / n
        pts.append(torch.stack([r * torch.cos(ang), r * torch.sin(ang)], dim=-1))
    return torch.cat([torch.as_tensor(p, dtype=dtype, device=device).reshape(-1, 2)
                      for p in pts], dim=0)


def ellipse_points(center: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                   theta: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
    """Map ``template (K,2)`` into ellipses -> ``(B, N, R, K, 2)``.

    ``center`` is ``(B,N,R,2)``, ``a``/``b``/``theta`` are ``(B,N)``.
    """
    B, N, R, _ = center.shape
    K = template.shape[0]
    scale = torch.stack([a, b], dim=-1)[:, :, None, None, :]          # (B,N,1,1,2)
    local = template.view(1, 1, 1, K, 2) * scale                     # (B,N,1,K,2)
    rot = ellipse_rotation(theta)[:, :, None, None]                  # (B,N,1,1,2,2)
    rot = rot.expand(B, N, R, 1, 2, 2)
    pts = torch.matmul(rot, local.unsqueeze(-1)).squeeze(-1)         # (B,N,R,K,2)
    return pts + center.unsqueeze(3)


def quadratic_form(x: torch.Tensor, center: torch.Tensor, a: torch.Tensor,
                   b: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Normalized quadratic form ``rho_E(x)``: ``<= 1`` inside the ellipse.

    ``x (..., 2)``, ``center (..., 2)``, ``a,b,theta`` broadcastable.
    """
    d = x - center
    c, s = torch.cos(theta), torch.sin(theta)
    dx = c * d[..., 0] + s * d[..., 1]
    dy = -s * d[..., 0] + c * d[..., 1]
    return (dx / a) ** 2 + (dy / b) ** 2


def sinusoidal_pe(s: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal positional encoding of normalized arclength ``s in [0,1]``."""
    half = dim // 2
    freqs = torch.exp(-torch.arange(half, device=s.device, dtype=s.dtype)
                      * (torch.log(torch.tensor(10000.0, device=s.device, dtype=s.dtype))
                         / max(half - 1, 1)))
    args = s.unsqueeze(-1) * freqs
    pe = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if pe.shape[-1] < dim:                                   # odd dim -> pad
        pe = torch.cat([pe, torch.zeros_like(pe[..., :1])], dim=-1)
    return pe[..., :dim]


def normalized_arclength(path: torch.Tensor,
                         mask: torch.Tensor | None = None) -> torch.Tensor:
    """``s_i`` in ``[0,1]`` from the cumulative chord length (plan section 8).

    ``mask (B,N)`` excludes padded points: a step that touches a padded point is
    dropped, so padding never stretches the arc-length parametrisation.
    """
    d = torch.linalg.norm(path[:, 1:] - path[:, :-1], dim=-1)
    if mask is not None:
        d = d * (mask[:, 1:] & mask[:, :-1]).to(d.dtype)
    cum = torch.cumsum(d, dim=1)
    cum = torch.cat([torch.zeros_like(cum[:, :1]), cum], dim=1)
    total = cum[:, -1:].clamp_min(1e-6)
    return cum / total
