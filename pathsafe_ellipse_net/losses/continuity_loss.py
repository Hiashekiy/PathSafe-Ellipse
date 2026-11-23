"""Path continuity loss (plan section 8) with the generator's ``rho_max``."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..geometry.box_quadratic_torch import midpoint_metric


def continuity_loss(center: torch.Tensor, shape_matrix: torch.Tensor,
                    pair_mask: torch.Tensor, rho_max: float = 0.85) -> torch.Tensor:
    """Penalise adjacent midpoints falling outside either ellipse by more than rho_max.

    ``center (B,N,2)``, ``shape_matrix (B,N,2,2)``, ``pair_mask (B,N-1)``.
    """
    if center.shape[1] < 2:
        return center.sum() * 0.0
    q0, q1 = center[:, :-1], center[:, 1:]
    S0, S1 = shape_matrix[:, :-1], shape_matrix[:, 1:]
    mid = 0.5 * (q0 + q1)
    rho0 = midpoint_metric(S0, q0, mid)
    rho1 = midpoint_metric(S1, q1, mid)
    pen = (F.relu(rho0 - rho_max).pow(2) + F.relu(rho1 - rho_max).pow(2))
    m = pair_mask.to(pen.dtype)
    return (pen * m).sum() / m.sum().clamp_min(1.0)
