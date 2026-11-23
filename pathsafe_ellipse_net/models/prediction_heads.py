"""Prediction heads: 16-way direction, 1 ratio, 4 shape parameters."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, cin: int, hidden: int, cout: int, dropout: float = 0.0):
        super().__init__()
        layers = [nn.Linear(cin, hidden), nn.SiLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, cout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PredictionHeads(nn.Module):
    """Direction logits (16), ratio logit (1) and shape parameters (4)."""

    def __init__(self, cin: int = 128, hidden: int = 128, rays: int = 16,
                 dropout: float = 0.1):
        super().__init__()
        self.dir_head = MLP(cin, hidden, rays, dropout)
        self.ratio_head = MLP(cin, hidden, 1, dropout)
        self.shape_head = MLP(cin, hidden, 4, dropout)
        self.rays = rays

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        direction_logits = self.dir_head(h)
        ratio_logit = self.ratio_head(h).squeeze(-1)
        shape = self.shape_head(h)
        # padded positions must not contribute random logits downstream
        m = mask.unsqueeze(-1)
        direction_logits = direction_logits.masked_fill(~m, 0.0)
        ratio_logit = ratio_logit.masked_fill(~mask, 0.0)
        shape = shape.masked_fill(~m, 0.0)
        return direction_logits, ratio_logit, shape


def sample_direction(logits: torch.Tensor, tau: float, hard: bool, training: bool
                     ) -> torch.Tensor:
    """Gumbel-softmax (hard, straight-through) in training, argmax at inference."""
    if training:
        return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
    idx = logits.argmax(dim=-1, keepdim=True)
    return torch.zeros_like(logits).scatter_(-1, idx, 1.0)
