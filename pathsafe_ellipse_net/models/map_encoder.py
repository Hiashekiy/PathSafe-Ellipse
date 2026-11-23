"""Global map encoder: residual CNN + FPN at strides 4 / 8 / 16.

Map size is never hard-coded: the encoder accepts any ``(H, W)`` and only the
down-sampling factor (4/8/16) matters, which is enforced by strided convolutions.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def norm_layer(kind: str, ch: int) -> nn.Module:
    if kind == "layer":
        return nn.GroupNorm(1, ch)          # equivalent to LayerNorm over channels
    return nn.GroupNorm(min(8, ch), ch)


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1, norm: str = "group"):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.norm = norm_layer(norm, cout)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, ch: int, norm: str = "group"):
        super().__init__()
        self.c1 = ConvBlock(ch, ch, norm=norm)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.n2 = norm_layer(norm, ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.n2(self.c2(self.c1(x))))


class MapEncoder(nn.Module):
    """``(B,1,H,W) -> [F4, F8, F16]`` = ``H/4, H/8, H/16`` (plan section 3.1).

    The stride-2 stem is what makes the first feature map sit at ``/4``: without it
    stage 1 would run at full resolution, costing ~4x the FLOPs for features at the
    wrong scale.
    """

    def __init__(self, in_ch: int = 1, width: int = 32, out_ch: int = 64,
                 strides: List[int] = (4, 8, 16), blocks: int = 2, norm: str = "group"):
        super().__init__()
        self.stem = ConvBlock(in_ch, width, stride=2, norm=norm)      # /2
        self.stage1 = nn.Sequential(*[ResidualBlock(width, norm) for _ in range(blocks)])
        self.down2 = ConvBlock(width, width * 2, stride=2, norm=norm)  # /4
        self.stage2 = nn.Sequential(*[ResidualBlock(width * 2, norm) for _ in range(blocks)])
        self.down3 = ConvBlock(width * 2, width * 4, stride=2, norm=norm)  # /8
        self.stage3 = nn.Sequential(*[ResidualBlock(width * 4, norm) for _ in range(blocks)])
        # channel counts: /2 -> width, /4 -> width*2, /8 -> width*4
        self.proj4 = nn.Conv2d(width * 2, out_ch, 1)
        self.proj8 = nn.Conv2d(width * 4, out_ch, 1)
        self.proj16 = nn.Conv2d(width * 4, out_ch, 1)
        self.out_ch = out_ch
        self.strides = tuple(strides)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        f2 = self.stage1(self.stem(x))                  # /2
        f4 = self.stage2(self.down2(f2))                # /4
        f8 = self.stage3(self.down3(f4))                # /8
        f16 = F.avg_pool2d(f8, 2, ceil_mode=True)       # /16
        return [self.proj4(f4), self.proj8(f8), self.proj16(f16)]


def encode_map_multi(encoder: MapEncoder, occupancy: torch.Tensor) -> List[torch.Tensor]:
    return encoder(occupancy)
