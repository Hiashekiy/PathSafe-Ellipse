"""S2 acceptance: coordinate query mapping and variable-size / masked forward.

Run: ``python pathsafe_ellipse_net/tests/test_coordinate_query.py``
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from pathsafe_ellipse_net.models import EllipseNet  # noqa: E402
from pathsafe_ellipse_net.models.coordinate_query import sample_features, world_to_grid  # noqa: E402


def test_world_to_grid_centre():
    """Cell ``j`` must map onto pixel ``j`` for ``align_corners=False``."""
    rs = torch.tensor([0.25])
    H = W = 64
    j = 20
    pts = torch.tensor([[[j * 0.25, 5 * 0.25]]])            # metric point of cell (5, 20)
    grid = world_to_grid(pts, rs, (H, W))
    expected_x = 2.0 * (j + 0.5) / W - 1.0
    expected_y = 2.0 * (5 + 0.5) / H - 1.0
    assert abs(float(grid[0, 0, 0]) - expected_x) < 1e-6
    assert abs(float(grid[0, 0, 1]) - expected_y) < 1e-6
    print(f"coordinate query: cell (5,{j}) -> grid ({expected_x:.4f}, {expected_y:.4f})")


def test_sample_ramp():
    """Sampling a column-index ramp reproduces the value of the queried cell."""
    H = W = 64
    rs = torch.tensor([0.5])
    ramp = (torch.arange(W, dtype=torch.float32) / (W - 1)).view(1, 1, 1, W).repeat(1, 1, H, 1)
    j = 20
    pts = torch.tensor([[[j * 0.5, 5 * 0.5]]])
    got = float(sample_features([ramp], pts, rs, (H, W))[0, 0, 0])
    expected = j / (W - 1)
    assert abs(got - expected) < 1e-4, (got, expected)
    print(f"coordinate query: ramp sampled {got:.5f} (expected {expected:.5f})")


def _cfg():
    return {
        "label": {"lidar_rays": 16, "lidar_max_range_m": 9.0, "r_max_m": 3.0,
                  "eps_r_m": 0.05, "min_eig_m2": 1e-6, "r_axis_max_m": 6.0,
                  "rho_max": 0.85, "epsilon_m": 0.01, "angle_aspect_min": 1.05},
        "model": {"map_width": 8, "map_out_ch": 16, "map_strides": [4, 8, 16],
                  "lidar_ch": 16, "geom_ch": 8, "geom_in": 7, "fusion_ch": 32,
                  "backbone_blocks": 4, "backbone_dilations": [1, 2, 4, 8],
                  "dropout": 0.0, "norm": "group"},
        "loss": {"center_beta": 0.25, "axes_beta": 0.1},
    }


def test_forward_variable_sizes_and_mask():
    torch.manual_seed(0)
    cfg = _cfg()
    model = EllipseNet(cfg).eval()
    for (H, W, N, rs) in ((128, 128, 12, 0.25), (256, 208, 20, 0.3125), (96, 160, 5, 0.1875)):
        occ = torch.zeros(2, 1, H, W)
        occ[:, :, : H // 4, : W // 4] = 1.0
        path = torch.rand(2, N, 2) * torch.tensor([W * rs, H * rs])
        lidar = torch.rand(2, N, 16) * 6.0
        geom = torch.rand(2, N, 7) * 2 - 1
        mask = torch.zeros(2, N, dtype=torch.bool)
        mask[0] = True
        mask[1, : N - 3] = True
        r_scene = torch.full((2,), rs)
        out = model(occ, path, lidar / 9.0, geom, r_scene, mask, lidar_m=lidar, tau=0.3)
        assert out["center"].shape == (2, N, 2)
        assert out["shape_matrix"].shape == (2, N, 2, 2)
        assert torch.allclose(out["direction_onehot"].sum(-1), torch.ones(2, N), atol=1e-5)
        off = (out["center"] - path).norm(dim=-1)
        assert float(off.max()) <= 3.0 + 1e-5, float(off.max())
        assert float(out["axes"][..., 1].min()) >= 1e-3 - 1e-6
        assert torch.all(out["axes"][..., 0] >= out["axes"][..., 1] - 1e-6)
    print("forward: variable map sizes (128/256x208/96x160), padding mask, "
          "one-hot rays, offset <= R_max, a >= b >= a_min")


def test_padded_positions_zero():
    torch.manual_seed(1)
    cfg = _cfg()
    model = EllipseNet(cfg).eval()
    H = W = 128
    N = 10
    occ = torch.zeros(2, 1, H, W)
    path = torch.rand(2, N, 2) * 20
    lidar = torch.rand(2, N, 16) * 5
    geom = torch.zeros(2, N, 7)
    mask = torch.zeros(2, N, dtype=torch.bool)
    mask[0, :4] = True
    mask[1, :7] = True
    out = model(occ, path, lidar / 9.0, geom, torch.full((2,), 0.25), mask, lidar_m=lidar)
    feats = out["features"]
    assert torch.allclose(feats[~mask], torch.zeros_like(feats[~mask])), "padding leaked"
    print("forward: padded positions stay exactly zero through the backbone")


if __name__ == "__main__":
    test_world_to_grid_centre()
    test_sample_ramp()
    test_forward_variable_sizes_and_mask()
    test_padded_positions_zero()
    print("test_coordinate_query: OK")
