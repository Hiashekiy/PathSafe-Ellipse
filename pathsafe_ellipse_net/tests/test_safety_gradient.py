"""S1 acceptance for the safety loss (plan section 7.7 checks 1-6).

Run: ``python pathsafe_ellipse_net/tests/test_safety_gradient.py``
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from pathsafe_ellipse_net.geometry import (SafetyCriterion, border_loss,  # noqa: E402
                                           center_free_loss, ellipse_safe_loss,
                                           signed_distance_field)


def make_map(H=32, W=32, blocks=()):
    occ = np.zeros((H, W), np.float32)
    for (r0, r1, c0, c1) in blocks:
        occ[r0:r1, c0:c1] = 1.0
    return torch.from_numpy(occ)[None, None]


def test_empty_map_zero():
    occ = make_map()
    B, N = 1, 3
    center = torch.tensor([[[5.0, 5.0], [10.0, 10.0], [15.0, 15.0]]])
    S = torch.tensor(np.tile(np.diag([1.0, 0.5]), (1, 3, 1, 1)), dtype=torch.float32)
    rs = torch.tensor([0.25])
    mask = torch.ones(B, N, dtype=torch.bool)
    loss = ellipse_safe_loss(center, S, occ, rs, mask)
    assert float(loss) == 0.0, loss
    print("safety: empty map -> ellipse_safe == 0")


def test_intersection_positive():
    occ = make_map(blocks=[(10, 14, 10, 14)])          # obstacle square around cell (11,11)
    center = torch.tensor([[[11.5 * 0.25, 11.5 * 0.25]]])   # right on the obstacle
    S = torch.tensor(np.diag([1.0, 1.0]), dtype=torch.float32)[None, None].repeat(1, 1, 1, 1)
    rs = torch.tensor([0.25])
    mask = torch.ones(1, 1, dtype=torch.bool)
    loss = ellipse_safe_loss(center, S, occ, rs, mask)
    assert float(loss) > 0.0, loss
    print(f"safety: intersecting ellipse -> ellipse_safe = {float(loss):.4f} > 0")


def test_center_free_inside_obstacle():
    occ_np = np.zeros((32, 32), np.float32)
    occ_np[10:14, 10:14] = 1.0
    sdf = torch.from_numpy(signed_distance_field(occ_np, 0.25))[None, None]
    inside = torch.tensor([[[11.5 * 0.25, 11.5 * 0.25]]])
    outside = torch.tensor([[[2.0, 2.0]]])
    rs = torch.tensor([0.25])
    mask = torch.ones(1, 1, dtype=torch.bool)
    l_in = center_free_loss(inside, sdf, rs, mask)
    l_out = center_free_loss(outside, sdf, rs, mask)
    assert float(l_in) > 0.0 and float(l_out) == 0.0, (l_in, l_out)
    print(f"safety: centre inside obstacle -> {float(l_in):.3f}, outside -> {float(l_out):.3f}")


def test_border_positive():
    center = torch.tensor([[[-0.5, 5.0]]])              # left of the map
    S = torch.tensor(np.diag([1.0, 1.0]), dtype=torch.float32)[None, None]
    rs = torch.tensor([0.25])
    mask = torch.ones(1, 1, dtype=torch.bool)
    loss = border_loss(center, S, rs, (32, 32), mask, m_border=0.01)
    assert float(loss) > 0.0, loss
    print(f"safety: ellipse outside the map -> border = {float(loss):.3f} > 0")


def test_gradients_finite():
    occ_np = np.zeros((48, 48), np.float32)
    occ_np[20:26, 20:26] = 1.0
    occ = torch.from_numpy(occ_np)[None, None]
    sdf = torch.from_numpy(signed_distance_field(occ_np, 0.25))[None, None]
    crit = SafetyCriterion(lambda_sdf=0.2, lambda_border=1.0, m_border_m=0.01)
    center = torch.tensor([[[5.4, 5.2], [6.1, 5.9]]], requires_grad=True)
    axes = torch.tensor([[[1.2, 0.9], [1.4, 1.0]]], requires_grad=True)
    theta = torch.tensor([[0.3, -0.4]], requires_grad=True)

    def S_of(a, t):
        c, s = torch.cos(t), torch.sin(t)
        R = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
        D = torch.zeros_like(R)
        D[..., 0, 0] = a[..., 0] ** 2
        D[..., 1, 1] = a[..., 1] ** 2
        return R @ D @ R.transpose(-1, -2)

    S = S_of(axes, theta)
    parts = crit(center, S, occ, sdf, torch.tensor([0.25]), torch.ones(1, 2, dtype=torch.bool))
    crit.total(parts).backward()
    for name, t in (("center", center), ("axes", axes), ("theta", theta)):
        assert t.grad is not None, name
        assert torch.isfinite(t.grad).all(), f"{name} grad not finite"
    print("safety: gradients finite wrt centre, axes and angle")


def test_checklist_numbers():
    """Plan 7.7 items 1-5 in one summary line."""
    occ = make_map(blocks=[(10, 14, 10, 14)])
    rs = torch.tensor([0.25])
    mask = torch.ones(1, 1, dtype=torch.bool)
    far = torch.tensor([[[8.0, 8.0]]])          # ellipse (r=1 m) still clear of the square
    near = torch.tensor([[[11.5 * 0.25, 11.5 * 0.25]]])
    S = torch.tensor(np.diag([1.0, 1.0]), dtype=torch.float32)[None, None]
    assert float(ellipse_safe_loss(far, S, occ, rs, mask)) == 0.0
    assert float(ellipse_safe_loss(near, S, occ, rs, mask)) > 0.0
    print("safety: checklist 1-5 satisfied")


if __name__ == "__main__":
    test_empty_map_zero()
    test_intersection_positive()
    test_center_free_inside_obstacle()
    test_border_positive()
    test_gradients_finite()
    test_checklist_numbers()
    print("test_safety_gradient: OK")
