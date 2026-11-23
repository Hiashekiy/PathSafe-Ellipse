"""LiDAR circular encoder: shared 1D CNN with circular padding.

Ray 0 and ray 15 are neighbours in direction space, so the convolution wraps
around; the ray order itself is never permuted.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CircularConv1d(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 3):
        super().__init__()
        self.pad = k // 2
        self.conv = nn.Conv1d(cin, cout, k, padding=0, bias=False)
        self.norm = nn.GroupNorm(min(8, cout), cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, self.pad), mode="circular")
        return F.silu(self.norm(self.conv(x)))


class LidarEncoder(nn.Module):
    """``(B,N,R)`` normalised ranges -> ``(B,N,ch)`` features."""

    def __init__(self, out_ch: int = 64, rays: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            CircularConv1d(1, 32),
            CircularConv1d(32, out_ch),
            CircularConv1d(out_ch, out_ch),
        )
        self.out_ch = out_ch

    def forward(self, lidar_norm: torch.Tensor) -> torch.Tensor:
        B, N, R = lidar_norm.shape
        x = lidar_norm.reshape(B * N, 1, R)
        x = self.net(x)                                        # (B*N, C, R)
        pooled = torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=-1)
        return pooled.reshape(B, N, -1)
