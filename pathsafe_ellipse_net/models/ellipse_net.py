"""EllipseNet: map encoder + coordinate query + LiDAR/geometry encoders +
path backbone + three heads, decoded into ray-constrained centres and ellipses.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..geometry.ellipse import decode_ellipse, decode_center
from .coordinate_query import sample_features
from .lidar_encoder import LidarEncoder
from .map_encoder import MapEncoder
from .path_backbone import PathBackbone
from .prediction_heads import MLP, PredictionHeads, sample_direction


class EllipseNet(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        mc = cfg["model"]
        self.map_encoder = MapEncoder(in_ch=1, width=int(mc["map_width"]),
                                      out_ch=int(mc["map_out_ch"]), norm=mc.get("norm", "group"))
        self.lidar_encoder = LidarEncoder(out_ch=int(mc["lidar_ch"]),
                                          rays=int(cfg["label"]["lidar_rays"]))
        geom_ch = int(mc["geom_ch"])
        self.geom_mlp = MLP(int(mc["geom_in"]), 64, geom_ch, dropout=0.0)
        fusion_in = (self.map_encoder.out_ch * len(mc["map_strides"])
                     + self.lidar_encoder.out_ch * 2 + geom_ch)
        self.fusion = MLP(fusion_in, int(mc["fusion_ch"]), int(mc["fusion_ch"]),
                          dropout=float(mc["dropout"]))
        self.backbone = PathBackbone(ch=int(mc["fusion_ch"]),
                                     dilations=tuple(mc["backbone_dilations"]),
                                     dropout=float(mc["dropout"]))
        self.heads = PredictionHeads(cin=int(mc["fusion_ch"]),
                                     hidden=int(mc["fusion_ch"]),
                                     rays=int(cfg["label"]["lidar_rays"]),
                                     dropout=float(mc["dropout"]))
        lab = cfg["label"]
        self.r_max_m = float(lab["r_max_m"])
        self.eps_r_m = float(lab["eps_r_m"])
        self.max_range_m = float(lab["lidar_max_range_m"])
        self.a_min = float(lab["min_eig_m2"]) ** 0.5
        self.a_max = float(lab["r_axis_max_m"])

    def forward(self, occupancy: torch.Tensor, path_xy: torch.Tensor,
                lidar_ranges: torch.Tensor, path_geometry: torch.Tensor,
                map_resolution: torch.Tensor, valid_mask: torch.Tensor,
                lidar_m: Optional[torch.Tensor] = None,
                tau: float = 1.0, hard: bool = True,
                map_index: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """``lidar_ranges`` may be normalised; ``lidar_m`` must be metres.

        ``occupancy`` holds **unique** maps when ``map_index`` is given (one map per
        scene), so the encoder runs once per scene and its features are gathered for
        every trajectory of that scene (plan section 9).

        Returns direction logits/onehot, ratio, centre, axes, angle, shape matrix.
        """
        B, N, _ = path_xy.shape
        map_size = occupancy.shape[-2:]
        feats = self.map_encoder(occupancy)
        if map_index is not None and len(feats[0]) != B:
            feats = [f[map_index] for f in feats]
        f_map = sample_features(feats, path_xy, map_resolution, map_size)   # (B,N,3C)
        f_lidar = self.lidar_encoder(lidar_ranges)                          # (B,N,2C)
        f_geom = self.geom_mlp(path_geometry)                               # (B,N,G)
        h0 = self.fusion(torch.cat([f_map, f_lidar, f_geom], dim=-1))       # (B,N,Cf)
        h = self.backbone(h0.transpose(1, 2), valid_mask).transpose(1, 2)   # (B,N,Cf)

        logits, ratio_logit, shape = self.heads(h, valid_mask)
        onehot = sample_direction(logits, tau=tau, hard=hard, training=self.training)
        ratio = torch.sigmoid(ratio_logit)

        lidar_metric = lidar_m if lidar_m is not None else lidar_ranges * self.max_range_m
        center = decode_center(path_xy, lidar_metric, onehot, ratio,
                               self.r_max_m, self.eps_r_m, self.max_range_m)
        ell = decode_ellipse(shape, self.a_min, self.a_max)
        return {
            "direction_logits": logits,
            "direction_onehot": onehot,
            "ratio": ratio,
            "center": center,
            "center_offset": center - path_xy,
            "axes": ell["axes"],
            "angle": ell["angle"],
            "shape_matrix": ell["shape_matrix"],
            "features": h,
        }
