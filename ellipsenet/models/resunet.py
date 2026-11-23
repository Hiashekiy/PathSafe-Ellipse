"""ResUNet: encoder-decoder with same-scale skip connections, full-resolution output.

One instance per map channel (occupancy, signed distance); the two branches never
share weights (plan section 3).  The decoder output has exactly the input
resolution, so a feature map cell ``(i, j)`` is aligned with pixel ``(i, j)`` of
the input map (verified by ``tests/test_alignment.py``).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def norm_layer(ch: int, kind: str = "group", groups: int = 8) -> nn.Module:
    if kind == "group":
        g = groups if ch % groups == 0 else 1
        return nn.GroupNorm(g, ch)
    if kind == "batch":
        return nn.BatchNorm2d(ch)
    if kind == "instance":
        return nn.InstanceNorm2d(ch, affine=True)
    raise ValueError(f"unknown norm '{kind}'")


class ResBlock(nn.Module):
    """Two 3x3 convolutions with a residual ("full pre-activation" style)."""

    def __init__(self, in_ch: int, out_ch: int, norm: str = "group"):
        super().__init__()
        self.n1 = norm_layer(in_ch, norm)
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.n2 = norm_layer(out_ch, norm)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = (nn.Identity() if in_ch == out_ch
                     else nn.Conv2d(in_ch, out_ch, 1))
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.c1(self.act(self.n1(x)))
        h = self.c2(self.act(self.n2(h)))
        return self.act(h + self.skip(x))


class Down(nn.Module):
    """Strided 3x3 convolution (learned downsampling, no pooling)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Up(nn.Module):
    """Bilinear upsample + skip concat + residual block."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, norm: str = "group"):
        super().__init__()
        self.block = ResBlock(in_ch + skip_ch, out_ch, norm)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class ResUNet(nn.Module):
    """Full-resolution feature extractor.

    ``depths`` counts residual blocks per level (stem, 3 encoder levels, bottleneck);
    ``base`` is the channel width at full resolution.  The output is
    ``(B, out_ch, H, W)`` with ``H, W`` identical to the input.
    """

    def __init__(self, in_ch: int = 1, out_ch: int = 128, base: int = 24,
                 depths=(1, 1, 1, 1), norm: str = "group"):
        super().__init__()
        d0, d1, d2, d3 = [int(d) for d in depths]
        c0 = base
        c1, c2, c3 = base * 2, base * 4, base * 8
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, c0, 3, padding=1), norm_layer(c0, norm), nn.SiLU(inplace=True),
            *[ResBlock(c0, c0, norm) for _ in range(d0)],
        )
        self.down1 = Down(c0, c1)
        self.enc1 = nn.Sequential(*[ResBlock(c1, c1, norm) for _ in range(d1)])
        self.down2 = Down(c1, c2)
        self.enc2 = nn.Sequential(*[ResBlock(c2, c2, norm) for _ in range(d2)])
        self.down3 = Down(c2, c3)
        self.enc3 = nn.Sequential(*[ResBlock(c3, c3, norm) for _ in range(d3)])
        self.up3 = Up(c3, c2, c2, norm)
        self.up2 = Up(c2, c1, c1, norm)
        self.up1 = Up(c1, c0, c0, norm)
        self.head = nn.Conv2d(c0, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)
        s1 = self.enc1(self.down1(s0))
        s2 = self.enc2(self.down2(s1))
        s3 = self.enc3(self.down3(s2))
        u = self.up3(s3, s2)
        u = self.up2(u, s1)
        u = self.up1(u, s0)
        return self.head(u)
