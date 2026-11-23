"""Path-level backbone: residual dilated 1D convolutions, padding-mask aware."""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(kind: str, ch: int) -> nn.Module:
    if kind == "layer":
        return nn.GroupNorm(1, ch)
    return nn.GroupNorm(min(8, ch), ch)


class ResDilatedBlock(nn.Module):
    """``x + Conv(k=3, d=dilation)(x)`` with GroupNorm + SiLU + dropout."""

    def __init__(self, ch: int, dilation: int, dropout: float = 0.1, norm: str = "group"):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation, bias=False)
        self.norm = _norm(norm, ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B,C,N)
        h = self.drop(F.silu(self.norm(self.conv(x))))
        return x + h


class PathBackbone(nn.Module):
    """Four residual dilated blocks with growing receptive field on the path."""

    def __init__(self, ch: int = 128, dilations: List[int] = (1, 2, 4, 8),
                 dropout: float = 0.1, norm: str = "group"):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ResDilatedBlock(ch, d, dropout, norm) for d in dilations])
        self.ch = ch

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``x (B,C,N)``, ``mask (B,N)``; padded positions never leak into features."""
        m = mask.unsqueeze(1).to(x.dtype)
        h = x * m
        for blk in self.blocks:
            # zero-pad the convolution input so padding cannot act as context, and
            # re-apply the mask so padded outputs stay exactly zero
            h = blk(h * m) * m
        return h
