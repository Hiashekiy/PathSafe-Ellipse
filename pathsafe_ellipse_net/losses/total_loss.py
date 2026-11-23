"""Total training loss with the plan's weights and the safety-weight warm-up."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from ..geometry.safety_loss import SafetyCriterion
from .continuity_loss import continuity_loss
from .regression_loss import regression_losses


class TotalLoss(nn.Module):
    """``L_center + L_axes + 0.5 L_angle + lambda_s L_safety + 0.1 L_continuity``.

    The external safety weight ramps linearly from 0 to ``w_safety`` over the first
    ``safety_warmup_frac`` of the training steps (never fully disabled afterwards).
    """

    def __init__(self, cfg: Dict, total_steps: int):
        super().__init__()
        lc = cfg["loss"]
        self.cfg = cfg
        self.w = {"center": float(lc["w_center"]), "axes": float(lc["w_axes"]),
                  "angle": float(lc["w_angle"]), "safety": float(lc["w_safety"]),
                  "continuity": float(lc["w_continuity"])}
        self.safety = SafetyCriterion(
            lambda_sdf=float(lc["w_lambda_sdf"]),
            lambda_border=float(lc["w_lambda_border"]),
            m_safe_m=float(lc.get("m_safe_m", 0.0)),
            m_center_m=float(lc.get("m_center_m", 0.0)),
            m_border_m=float(cfg["label"].get("epsilon_m", 0.01)))
        self.rho_max = float(cfg["label"]["rho_max"])
        self.total_steps = max(int(total_steps), 1)
        self.warmup_steps = int(self.total_steps * float(lc.get("safety_warmup_frac", 0.08)))

    def safety_scale(self, step: int) -> float:
        if self.warmup_steps <= 0:
            return 1.0
        return min(1.0, step / self.warmup_steps)

    def forward(self, pred: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor],
                step: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        valid = batch["valid"] & batch["mask"]
        reg = regression_losses(pred, batch, valid, self.cfg)
        occ, sdf = batch["occupancy"], batch["sdf"]
        # collate stores unique scene maps; expand them to per-trajectory views
        if "map_index" in batch and occ.shape[0] != pred["center"].shape[0]:
            occ = occ[batch["map_index"]]
            sdf = sdf[batch["map_index"]]
        parts = self.safety(center=pred["center"], shape_matrix=pred["shape_matrix"],
                            occupancy=occ, sdf=sdf,
                            r_scene=batch["r_scene"], valid_mask=valid)
        cont = continuity_loss(pred["center"], pred["shape_matrix"],
                               batch["pair_mask"] & valid[:, :-1] & valid[:, 1:],
                               rho_max=self.rho_max)
        scale = self.safety_scale(step)
        total = (self.w["center"] * reg["center"] + self.w["axes"] * reg["axes"]
                 + self.w["angle"] * reg["angle"]
                 + scale * self.w["safety"] * self.safety.total(parts)
                 + self.w["continuity"] * cont)
        logs = {"center": float(reg["center"].detach()), "axes": float(reg["axes"].detach()),
                "angle": float(reg["angle"].detach()),
                "continuity": float(cont.detach()), "safety_scale": scale}
        logs.update(parts.as_dict())
        logs["total"] = float(total.detach())
        return total, logs
