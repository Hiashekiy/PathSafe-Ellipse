"""Geometric-expected losses (plan section 10). No centre / direction / shape labels.

For every path point ``i`` and candidate ray ``j``::

    L_ij      = w_d * (-min(max(S(q_ij), 0), D_max))                  # centre clearance
              + w_s * mean_k [max(0, m - S(x_ijk))]^2                 # ellipse safety
              + w_a * (-log(pi * a_i * b_i + eps))                    # area (shape only)
    L_choice  = mean_i sum_j pi_ij L_ij                               # direction training
    L_conn    = mean_i sum_jk pi_ij pi_i+1,k L_conn_ij,k              # boundary coverage
    L_total   = L_choice + lambda_c * L_conn

``S`` is the metric signed distance field sampled bilinearly from the data version
(positive in free space); positions outside the map read ``SDF_OUTSIDE_M`` (blocked),
so an ellipse pushed off the map is penalised instead of being clamped free.

Raw component magnitudes are logged separately; ``loss.scales`` in the config
divides them before the weights are applied (plan 10.7: record, then normalise).
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .sampling import ellipse_points, quadratic_form, sample_sdf, unit_ellipse_template


class GeometricLoss(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        lc = cfg["loss"]
        mc = cfg["model"]
        self.w_dist = float(lc["w_dist"])
        self.w_safe = float(lc["w_safe"])
        self.w_area = float(lc["w_area"])
        self.w_conn = float(lc["w_conn"])
        self.d_max = float(lc["d_max_m"])
        self.d_margin = float(lc["d_margin_m"])
        self.area_eps = float(lc.get("area_eps", 1e-6))
        scales = lc.get("scales", {}) or {}
        self.s_dist = float(scales.get("dist", 1.0))
        self.s_safe = float(scales.get("safe", 1.0))
        self.s_area = float(scales.get("area", 1.0))
        self.s_conn = float(scales.get("conn", 1.0))
        self.register_buffer(
            "safe_template",
            unit_ellipse_template(mc["ellipse_samples_radii"], mc["ellipse_samples_counts"]),
            persistent=False)
        self.register_buffer(
            "boundary_template",
            unit_ellipse_template([1.0], [int(mc["conn_boundary_samples"])]),
            persistent=False)

    # -------------------------------------------------------------- components
    def local_terms(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        """Per-candidate local losses, ``(B,N,R)`` each plus the area term ``(B,N)``."""
        sdf = batch["sdf"]
        r_scene = batch["r_scene"]
        q = out["q"]
        a, b, theta = out["a"], out["b"], out["theta"]

        s_center = sample_sdf(sdf, q, r_scene)                        # (B,N,R)
        l_dist = -torch.clamp(s_center, min=0.0, max=self.d_max)

        pts = ellipse_points(q, a, b, theta, self.safe_template)      # (B,N,R,K,2)
        s_ell = sample_sdf(sdf, pts, r_scene)                         # (B,N,R,K)
        pen = torch.clamp(self.d_margin - s_ell, min=0.0) ** 2
        l_safe = pen.mean(dim=-1)

        area = torch.pi * a * b
        l_area = -torch.log(area + self.area_eps)                     # (B,N)
        return {"dist": l_dist, "safe": l_safe, "area": l_area,
                "s_center": s_center, "area_m2": area}

    def connection_terms(self, out: Dict) -> torch.Tensor:
        """Bidirectional boundary-coverage loss for every candidate pair, ``(B,N-1,R,R)``."""
        q = out["q"]
        a, b, theta = out["a"], out["b"], out["theta"]
        bnd = ellipse_points(q, a, b, theta, self.boundary_template)   # (B,N,R,Kb,2)

        # boundary samples of candidate j at point i against every candidate k at i+1
        x = bnd[:, :-1][:, :, :, None]                                 # (B,N-1,R,1,Kb,2)
        cen_next = q[:, 1:][:, :, None, :, None, :]                    # (B,N-1,1,R,1,2)
        a_next = a[:, 1:][:, :, None, None, None]                      # (B,N-1,1,1,1)
        b_next = b[:, 1:][:, :, None, None, None]
        th_next = theta[:, 1:][:, :, None, None, None]
        rho_next = quadratic_form(x, cen_next, a_next, b_next, th_next)
        pen_next = torch.clamp(rho_next - 1.0, min=0.0).pow(2).mean(dim=-1)

        # boundary samples of candidate k at point i+1 against every candidate j at i
        y = bnd[:, 1:][:, :, None]                                     # (B,N-1,1,R,Kb,2)
        cen_prev = q[:, :-1][:, :, :, None, None]                      # (B,N-1,R,1,1,2)
        a_prev = a[:, :-1][:, :, None, None, None]                     # (B,N-1,1,1,1)
        b_prev = b[:, :-1][:, :, None, None, None]
        th_prev = theta[:, :-1][:, :, None, None, None]
        rho_prev = quadratic_form(y, cen_prev, a_prev, b_prev, th_prev)
        pen_prev = torch.clamp(rho_prev - 1.0, min=0.0).pow(2).mean(dim=-1)
        return pen_next + pen_prev

    # -------------------------------------------------------------- full loss
    def forward(self, out: Dict, batch: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        mask = batch["mask"]                                           # (B,N) bool
        pi = out["pi"]
        terms = self.local_terms(out, batch)
        l_local = (self.w_dist * terms["dist"] / self.s_dist
                   + self.w_safe * terms["safe"] / self.s_safe
                   + self.w_area * terms["area"][:, :, None] / self.s_area)   # (B,N,R)

        m = mask.to(l_local.dtype)
        denom = m.sum().clamp_min(1.0)
        l_choice_pt = (pi * l_local).sum(dim=-1)                       # (B,N)
        l_choice = (l_choice_pt * m).sum() / denom

        conn_pair = self.connection_terms(out)                          # (B,N-1,R,R)
        pi_next = pi[:, 1:]
        l_conn_pt = torch.einsum("bnj,bnk,bnjk->bn", pi[:, :-1], pi_next, conn_pair)
        pair_mask = (mask[:, :-1] & mask[:, 1:]).to(l_conn_pt.dtype)
        l_conn = (l_conn_pt * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)

        total = l_choice + self.w_conn * l_conn / self.s_conn

        with torch.no_grad():
            ent = -(pi.clamp_min(1e-9).log() * pi).sum(dim=-1)         # (B,N)
            safe_mean = (self.w_safe * terms["safe"] / self.s_safe)
            logs = {
                "total": float(total.detach()),
                "choice": float(l_choice.detach()),
                "conn": float(l_conn.detach()),
                # raw magnitudes (plan 10.7) - never used for gradients
                "raw_dist": float((-(terms["dist"]) * m[:, :, None]).sum()
                                  / (m.sum() * pi.shape[-1]).clamp_min(1.0)),
                "raw_safe": float((safe_mean * m[:, :, None]).sum()
                                  / (m.sum() * pi.shape[-1]).clamp_min(1.0)),
                "raw_area": float((terms["area"] * m).sum() / denom),
                "raw_conn": float((l_conn_pt * pair_mask).sum()
                                  / pair_mask.sum().clamp_min(1.0)),
                "s_center_mean": float((terms["s_center"] * m[:, :, None]).sum()
                                       / (m.sum() * pi.shape[-1]).clamp_min(1.0)),
                "area_mean": float((terms["area_m2"] * m).sum() / denom),
                "pi_entropy": float((ent * m).sum() / denom),
                "pi_max": float((pi.max(dim=-1).values * m).sum() / denom),
                "pi_uniform": float(torch.log(torch.tensor(float(pi.shape[-1])))),
            }
        return total, logs
