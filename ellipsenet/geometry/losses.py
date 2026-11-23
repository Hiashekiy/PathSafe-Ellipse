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
        self.w_span = float(lc.get("w_span", 0.0))
        self.w_smooth = float(lc.get("w_smooth", 0.0))
        self.span_rho_target = float(lc.get("span_rho_target", 0.85))
        self.d_max = float(lc["d_max_m"])
        self.d_margin = float(lc["d_margin_m"])
        self.area_eps = float(lc.get("area_eps", 1e-6))
        scales = lc.get("scales", {}) or {}
        self.s_dist = float(scales.get("dist", 1.0))
        self.s_safe = float(scales.get("safe", 1.0))
        self.s_area = float(scales.get("area", 1.0))
        self.s_conn = float(scales.get("conn", 1.0))
        self.s_span = float(scales.get("span", 1.0))
        self.s_smooth = float(scales.get("smooth", 1.0))
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

    def span_terms(self, out: Dict, batch: Dict):
        """Arc-gap coverage: ``(pen_prev, pen_next)``, each ``(B,N-1,R)``.

        The midpoint of two consecutive stations must lie *well* inside both selected
        ellipses (plan 10.4: "路径点间距不同的情况应按实际弧长间隔调整连接要求").

        The boundary-sample term above is a *distance* penalty (zero gradient at
        ``rho = 1``), so it cannot remove small gaps; this term uses the generator's
        own continuity criterion ``rho <= rho_target (0.85)`` evaluated at the chord
        midpoint, which is also the reported "midpoint continuity" metric.
        ``pen_prev`` is indexed by the candidate at point ``i`` and ``pen_next`` by the
        candidate at ``i+1``, so the 16x16 joint expectation collapses exactly into two
        probability-weighted 16-vectors (no approximation).
        """
        path = batch["path_xy"]
        q = out["q"]
        a, b, theta = out["a"], out["b"], out["theta"]
        mid = 0.5 * (path[:, :-1] + path[:, 1:])                       # (B,N-1,2)
        x = mid[:, :, None, None, :]                                   # (B,N-1,1,1,2)
        cen_p = q[:, :-1][:, :, None, :]                               # (B,N-1,1,R,2)
        rho_p = quadratic_form(x, cen_p, a[:, :-1][:, :, None, None],
                               b[:, :-1][:, :, None, None],
                               theta[:, :-1][:, :, None, None]).squeeze(2)   # (B,N-1,R)
        cen_n = q[:, 1:][:, :, None, :]
        rho_n = quadratic_form(x, cen_n, a[:, 1:][:, :, None, None],
                               b[:, 1:][:, :, None, None],
                               theta[:, 1:][:, :, None, None]).squeeze(2)
        pen_p = torch.clamp(rho_p - self.span_rho_target, min=0.0).pow(2)
        pen_n = torch.clamp(rho_n - self.span_rho_target, min=0.0).pow(2)
        return pen_p, pen_n

    def offset_smooth_terms(self, out: Dict, batch: Dict) -> torch.Tensor:
        """Adjacent-station offset coherence, ``(B,N-1)``.

        ``Delta_i = q_i - p_i`` is the probability-weighted lateral/forward shift the
        network chose at station ``i``.  Two neighbouring ellipses keep a continuous
        corridor only if their shifts are similar -- the measured failure mode is
        neighbouring stations choosing *opposite* rays, which tears the chain apart and
        (measured) cannot be repaired by the arc-gap term without shrinking the
        clearance.  This term penalises ``||Delta_i - Delta_{i+1}||^2`` and therefore
        acts on the direction distribution and the ratio only; it never rewards a
        bigger ellipse, so it does not trade safety away.

        Uses the A* path (input) and the prediction only -- no labels.
        """
        path = batch["path_xy"]
        delta = out["q"] - path[:, :, None, :]                         # (B,N,R,2)
        mbar = (out["pi"][..., None] * delta).sum(dim=2)               # (B,N,2)
        d = mbar[:, 1:] - mbar[:, :-1]
        return (d * d).sum(dim=-1)

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

        pen_p, pen_n = self.span_terms(out, batch)                      # (B,N-1,R) each
        l_span_pt = ((pi[:, :-1] * pen_p).sum(-1) + (pi[:, 1:] * pen_n).sum(-1))
        l_span = (l_span_pt * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)

        l_smooth_pt = self.offset_smooth_terms(out, batch)              # (B,N-1)
        l_smooth = (l_smooth_pt * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)

        total = (l_choice + self.w_conn * l_conn / self.s_conn
                 + self.w_span * l_span / self.s_span
                 + self.w_smooth * l_smooth / self.s_smooth)

        with torch.no_grad():
            ent = -(pi.clamp_min(1e-9).log() * pi).sum(dim=-1)         # (B,N)
            safe_mean = (self.w_safe * terms["safe"] / self.s_safe)
            logs = {
                "total": float(total.detach()),
                "choice": float(l_choice.detach()),
                "conn": float(l_conn.detach()),
                "span": float(l_span.detach()),
                "smooth": float(l_smooth.detach()),
                # raw magnitudes (plan 10.7) - never used for gradients
                "raw_dist": float((-(terms["dist"]) * m[:, :, None]).sum()
                                  / (m.sum() * pi.shape[-1]).clamp_min(1.0)),
                "raw_safe": float((safe_mean * m[:, :, None]).sum()
                                  / (m.sum() * pi.shape[-1]).clamp_min(1.0)),
                "raw_area": float((terms["area"] * m).sum() / denom),
                "raw_conn": float((l_conn_pt * pair_mask).sum()
                                  / pair_mask.sum().clamp_min(1.0)),
                "raw_span": float(l_span.detach()),
                "raw_smooth": float(l_smooth.detach()),
                "s_center_mean": float((terms["s_center"] * m[:, :, None]).sum()
                                       / (m.sum() * pi.shape[-1]).clamp_min(1.0)),
                "area_mean": float((terms["area_m2"] * m).sum() / denom),
                "pi_entropy": float((ent * m).sum() / denom),
                "pi_max": float((pi.max(dim=-1).values * m).sum() / denom),
                "pi_uniform": float(torch.log(torch.tensor(float(pi.shape[-1])))),
            }
        return total, logs
