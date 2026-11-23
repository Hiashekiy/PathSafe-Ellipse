"""PathSafe-EllipseNet (spec V3): two ResUNets + coordinate queries + two-level attention.

Data flow (plan sections 2-9, no deviation):

    occupancy (B,1,H,W) -> ResUNet_occ  -> F_occ (B,C,H,W)
    signed distance     -> ResUNet_dist -> F_dist (B,C,H,W)
    9 occupancy tokens around each path point (+ relative position before flatten)
    3 distance samples per LiDAR ray (+ SDF scalar, lambda, range, direction, valid)
    ray MLP with the broadcast occupancy token  -> R (B,N,16,C)
    per-point 16-ray self-attention             -> R_hat
    MLP([c_occ, Flatten(R_hat)])                -> H_local
    path self-attention (+ arclength PE)        -> H_global
    Direction (16 logits) / Ratio (1) / Shape (a,b,theta) heads
    16 candidate ellipses, each centre exactly on its own ray

There is no path tangent input, no clearance scalar, no hand-made ray index or
offset-ratio label and no point-ray cross-attention.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from ..geometry.sampling import (sample_feature, sample_sdf, sinusoidal_pe,
                                 normalized_arclength)
from .resunet import ResUNet


def mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2,
        dropout: float = 0.0) -> nn.Sequential:
    """``layers``-deep GELU MLP (``layers >= 1``)."""
    mods: list = []
    d = in_dim
    for _ in range(max(layers - 1, 0)):
        mods += [nn.Linear(d, hidden), nn.GELU()]
        if dropout > 0:
            mods.append(nn.Dropout(dropout))
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


def transformer_encoder(layers: int, d_model: int, heads: int, ff_mult: int,
                        dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads,
                                       dim_feedforward=int(d_model * ff_mult),
                                       dropout=dropout, activation="gelu",
                                       batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)


class PSENet(nn.Module):
    """Spec-conformant network. ``cfg`` is the yaml config dict."""

    def __init__(self, cfg: Dict):
        super().__init__()
        mc = cfg["model"]
        dc = cfg["data"]
        self.cfg = cfg
        C = int(mc["ch"])
        self.C = C
        self.rays = int(dc["lidar_rays"])
        self.d_clip = float(dc["d_clip"])
        self.occ_spacing_m = float(mc["occ_token_spacing_m"])
        self.ray_lambdas = tuple(float(v) for v in mc["ray_sample_lambdas"])
        self.a_min = float(mc["shape_a_min_m"])
        self.a_max = float(mc["shape_a_max_m"])
        self.b_min = float(mc["shape_b_min_m"])
        self.ray_margin_m = float(mc["candidate_ray_margin_m"])
        self.pos_norm_m = float(mc["occ_pos_encoding_norm_m"])
        ablate = mc.get("ablate", {}) or {}
        self.ab_occ_point = bool(ablate.get("occ_point", False))
        self.ab_ray_attn = bool(ablate.get("ray_attn", False))
        self.ab_path_attn = bool(ablate.get("path_attn", False))
        self.ab_pos = bool(ablate.get("occ_pos_encoding", False))

        base = int(mc["resunet_base"])
        depths = mc.get("resunet_depths", [1, 1, 1, 1])
        norm = mc.get("norm", "group")
        self.occ_unet = ResUNet(1, C, base, depths, norm)
        self.dist_unet = ResUNet(1, C, base, depths, norm)

        drop = float(mc.get("dropout", 0.0))
        hid = int(mc.get("mlp_hidden", 2 * C))
        n_tok = 9
        self.pos_mlp = mlp(2, hid, C, layers=2, dropout=drop)
        self.token_mlp = mlp(2 * C, hid, C, layers=2, dropout=drop)
        self.occ_mlp = mlp(n_tok * C, hid, C, layers=2, dropout=drop)

        self.dir_mlp = mlp(2, hid, C, layers=2, dropout=drop)
        self.slot_embed = nn.Parameter(torch.zeros(self.rays, C))
        n_lam = len(self.ray_lambdas)
        ray_in = C * n_lam + n_lam + n_lam + 1 + C + 1
        self.ray_mlp = mlp(ray_in, hid, C, layers=2, dropout=drop)
        self.fusion_mlp = mlp(2 * C, hid, C, layers=2, dropout=drop)
        self.point_mlp = mlp(C + self.rays * C, hid, C, layers=2, dropout=drop)

        self.ray_attn = transformer_encoder(int(mc["ray_attn_layers"]), C,
                                            int(mc["attn_heads"]),
                                            int(mc["ff_mult"]), float(mc["attn_dropout"]))
        self.path_attn = transformer_encoder(int(mc["path_attn_layers"]), C,
                                             int(mc["attn_heads"]),
                                             int(mc["ff_mult"]), float(mc["attn_dropout"]))

        self.dir_head = nn.Linear(C, self.rays)
        self.ratio_head = nn.Linear(C, 1)
        self.shape_head = nn.Linear(C, 4)

        # 3x3 occupancy token offsets in metres (fixed, in plan section 4 order:
        # row-major over (dy, dx) from -spacing..+spacing)
        s = self.occ_spacing_m
        vals = [-s, 0.0, s]
        offs = torch.tensor([[dx, dy] for dy in vals for dx in vals], dtype=torch.float32)
        self.register_buffer("pos_offsets", offs, persistent=False)
        self._init_heads(float(mc.get("init_a_m", 1.5)), float(mc.get("init_b_m", 1.0)),
                         float(mc.get("init_ratio", 0.3)))

    def _init_heads(self, a0: float, b0: float, ratio0: float) -> None:
        """Start from a small, valid ellipse instead of an arbitrary one.

        With a default initialisation the shape head produces random semi-axes down
        to ``b_min``, and two neighbouring ``b = 5 cm`` ellipses make the boundary
        coverage loss explode (order 1e2) before any learning happens.
        """
        p_a = (a0 - self.a_min) / max(self.a_max - self.a_min, 1e-6)
        p_b = (b0 - self.b_min) / max(a0 - self.b_min, 1e-6)
        logit = lambda p: float(torch.log(torch.tensor(p / (1.0 - p))))  # noqa: E731
        with torch.no_grad():
            self.shape_head.weight.normal_(0.0, 0.01)
            self.shape_head.bias.copy_(torch.tensor([logit(p_a), logit(p_b), 1.0, 0.0]))
            self.ratio_head.weight.normal_(0.0, 0.01)
            self.ratio_head.bias.fill_(logit(ratio0))

    # ------------------------------------------------------------------ helpers
    def _occupancy_tokens(self, F_occ: torch.Tensor, path: torch.Tensor,
                          r_scene: torch.Tensor) -> torch.Tensor:
        B, N, _ = path.shape
        offs = self.pos_offsets.to(path.dtype)                            # (9,2)
        coords = path[:, :, None, :] + offs.view(1, 1, 9, 2)              # (B,N,9,2)
        m = sample_feature(F_occ, coords, r_scene)                        # (B,N,9,C)
        if self.ab_pos:
            z = self.token_mlp(torch.cat([m, torch.zeros_like(m)], dim=-1))
        else:
            e = self.pos_mlp(offs / self.pos_norm_m)                      # (9,C)
            e = e.view(1, 1, 9, self.C).expand(B, N, 9, self.C)
            z = self.token_mlp(torch.cat([m, e], dim=-1))
        return self.occ_mlp(z.reshape(B, N, 9 * self.C))                  # (B,N,C)

    def _ray_tokens(self, F_dist: torch.Tensor, sdf: torch.Tensor, path: torch.Tensor,
                    lidar: torch.Tensor, ray_dir: torch.Tensor, ray_valid: torch.Tensor,
                    r_scene: torch.Tensor, c_occ: torch.Tensor) -> torch.Tensor:
        B, N, _ = path.shape
        R = self.rays
        lam = torch.tensor(self.ray_lambdas, dtype=path.dtype, device=path.device)
        K = lam.numel()
        u = ray_dir[:, None, :, None, :]                                  # (B,1,R,1,2)
        step = lam.view(1, 1, 1, K, 1) * lidar[:, :, :, None, None]
        pos = path[:, :, None, None, :] + step * u                        # (B,N,R,K,2)
        m_dist = sample_feature(F_dist, pos, r_scene)                     # (B,N,R,K,C)
        s_val = sample_sdf(sdf, pos, r_scene)                             # (B,N,R,K)

        e_dir = self.dir_mlp(ray_dir) + self.slot_embed.to(path.dtype)    # (B,R,C)
        e_dir = e_dir[:, None].expand(B, N, R, self.C)
        lam_b = lam.view(1, 1, 1, K).expand(B, N, R, K)
        parts = [
            m_dist.reshape(B, N, R, K * self.C),
            s_val,
            lam_b,
            lidar[:, :, :, None],
            e_dir,
            ray_valid[:, :, :, None],
        ]
        c_ray = self.ray_mlp(torch.cat(parts, dim=-1))                    # (B,N,R,C)
        return self.fusion_mlp(torch.cat([c_occ[:, :, None].expand(B, N, R, self.C),
                                          c_ray], dim=-1))

    # ------------------------------------------------------------------ forward
    def forward(self, batch: Dict[str, torch.Tensor], tau: float = 1.0) -> Dict:
        occ = batch["occupancy"]
        sdf = batch["sdf"]
        path = batch["path_xy"]
        lidar = batch["lidar"]
        ray_dir = batch["ray_dir"]
        ray_valid = batch["ray_valid"]
        mask = batch["mask"]
        r_scene = batch["r_scene"]

        B, N, _ = path.shape
        R = self.rays
        F_occ = self.occ_unet(occ)
        sdf_in = torch.clamp(sdf / self.d_clip, -1.0, 1.0)
        F_dist = self.dist_unet(sdf_in)

        c_occ = self._occupancy_tokens(F_occ, path, r_scene)              # (B,N,C)
        c_occ_f = c_occ if not self.ab_occ_point else torch.zeros_like(c_occ)
        R_tok = self._ray_tokens(F_dist, sdf, path, lidar, ray_dir, ray_valid,
                                 r_scene, c_occ_f)                        # (B,N,R,C)

        if self.ab_ray_attn:
            R_hat = R_tok
        else:
            R_hat = self.ray_attn(R_tok.reshape(B * N, R, self.C)).reshape(B, N, R, self.C)

        h_local = self.point_mlp(torch.cat([c_occ_f, R_hat.reshape(B, N, R * self.C)],
                                           dim=-1))                       # (B,N,C)

        s = normalized_arclength(path, mask)
        h = h_local + sinusoidal_pe(s, self.C)
        if self.ab_path_attn:
            H_out = h
        else:
            H_out = self.path_attn(h, src_key_padding_mask=~mask)

        logits = self.dir_head(H_out)                                     # (B,N,16)
        pi = torch.softmax(logits / max(float(tau), 1e-6), dim=-1)
        alpha = torch.sigmoid(self.ratio_head(H_out).squeeze(-1))         # (B,N)
        raw = self.shape_head(H_out)                                      # (B,N,4)

        a = self.a_min + (self.a_max - self.a_min) * torch.sigmoid(raw[..., 0])
        b = self.b_min + (a - self.b_min) * torch.sigmoid(raw[..., 1])
        theta = torch.atan2(raw[..., 3], raw[..., 2])

        d_eff = torch.clamp(lidar - self.ray_margin_m, min=0.0)
        q = path[:, :, None, :] + (alpha[:, :, None, None] * d_eff[..., None]) \
            * ray_dir[:, None, :, :]                                      # (B,N,R,2)

        return {
            "logits": logits, "pi": pi, "alpha": alpha, "a": a, "b": b, "theta": theta,
            "q": q, "h_local": h_local, "h_global": H_out, "c_occ": c_occ,
            "R_hat": R_hat, "path_s": s,
        }

    # ------------------------------------------------------------------ decode
    @torch.no_grad()
    def decode_argmax(self, out: Dict) -> Dict[str, torch.Tensor]:
        """Inference rule (plan 13): pick the most probable ray, never an average."""
        j = out["pi"].argmax(dim=-1)                                      # (B,N)
        q = torch.gather(out["q"], 2, j[..., None, None].expand(-1, -1, 1, 2)).squeeze(2)
        return {"index": j, "center": q, "a": out["a"], "b": out["b"],
                "theta": out["theta"], "prob": out["pi"].max(dim=-1).values,
                "alpha": out["alpha"]}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
