"""Path-conditioned ellipse regression network (phase 2: training / testing).

Input  : 128x128 inflated local occupancy patch + 16 range rays + path tangent.
Output : ellipse centre offset (m), semi-axes (m), orientation (sin, cos) and an
         ``angle_valid`` logit that mirrors the label's near-circular flag.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class EllipseNet(nn.Module):
    def __init__(self, ray_count: int = 16, width: int = 32, dropout: float = 0.05):
        super().__init__()
        w = width
        self.stem = nn.Sequential(
            ConvBlock(1, w, 2),        # 64
            ConvBlock(w, w * 2, 2),    # 32
            ConvBlock(w * 2, w * 3, 2),   # 16
            ConvBlock(w * 3, w * 4, 2),   # 8
            ConvBlock(w * 4, w * 4, 2),    # 4
        )
        self.img_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(w * 4 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.ray_mlp = nn.Sequential(
            nn.Linear(ray_count * 2 + 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Linear(256 + 64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(256, 2 + 2 + 2 + 1)

    def forward(self, patch, rays, hits, tangent):
        img = self.stem(patch).flatten(1)
        img = self.img_head[1](self.img_head[0](img))
        img = self.img_head[2](img)
        ray_feat = self.ray_mlp(torch.cat([rays, hits, tangent], dim=1))
        feat = self.fuse(torch.cat([img, ray_feat], dim=1))
        out = self.head(feat)
        center = torch.tanh(out[:, 0:2]) * 8.0            # metres, |offset| <= 8 m
        log_ab = out[:, 2:4]
        sincos = F.normalize(out[:, 4:6], dim=1)
        angle_logit = out[:, 6]
        return {"center": center, "log_ab": log_ab, "sincos": sincos,
                "angle_logit": angle_logit}

    @torch.no_grad()
    def predict_ellipse(self, batch):
        """Return ``(center_m, semi_axes_m, angle_rad)`` for a batch of tensors."""
        out = self.forward(batch["patch"], batch["rays"], batch["hits"], batch["tangent"])
        ab = torch.exp(out["log_ab"])
        angle = torch.atan2(out["sincos"][:, 0], out["sincos"][:, 1])
        return out["center"], ab, angle, out["sincos"], out["angle_logit"]
