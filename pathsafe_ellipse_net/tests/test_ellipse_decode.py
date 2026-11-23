"""S2 acceptance: ellipse/centre decoding (plan sections 4 and 5).

Run: ``python pathsafe_ellipse_net/tests/test_ellipse_decode.py``
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from pathsafe_ellipse_net.geometry import (decode_center, decode_ellipse, ray_directions,
                                           shape_axes_angle)
from pathsafe_ellipse_net.geometry.ellipse import ray_limits


def test_axes_bounds():
    z = torch.randn(2000, 4) * 6
    ell = decode_ellipse(z, a_min=1e-3, a_max=6.0)
    a, b = ell["axes"][..., 0], ell["axes"][..., 1]
    assert torch.all(a <= 6.0 + 1e-5) and torch.all(a >= 1e-3 - 1e-6)
    assert torch.all(b <= a + 1e-6) and torch.all(b >= 1e-3 - 1e-6)
    print(f"axes: a in [{float(a.min()):.4f}, {float(a.max()):.4f}], b <= a always")


def test_angle_and_shape_roundtrip():
    z = torch.randn(500, 4)
    ell = decode_ellipse(z, 1e-3, 6.0)
    axes2, theta2 = shape_axes_angle(ell["shape_matrix"])
    assert torch.allclose(axes2, ell["axes"], atol=1e-4)
    d = torch.remainder(theta2 - ell["angle"], np.pi)
    d = torch.minimum(d, np.pi - d)
    assert float(d.max()) < 1e-3, float(d.max())
    print(f"shape: S -> (axes, angle) round-trip, max angle err {float(d.max()):.2e} rad")


def test_ray_decode_constraint():
    torch.manual_seed(0)
    B, N, R = 2, 5, 16
    path = torch.zeros(B, N, 2)
    lidar = torch.rand(B, N, R) * 12.0
    idx = torch.randint(0, R, (B, N))
    onehot = torch.zeros(B, N, R).scatter_(-1, idx.unsqueeze(-1), 1.0)
    ratio = torch.rand(B, N)
    center = decode_center(path, lidar, onehot, ratio, r_max_m=3.0, eps_r_m=0.05,
                           max_range_m=9.0)
    off = center - path
    u = ray_directions(R)
    chosen = u[idx]                                     # (B,N,2)
    step = (off * chosen).sum(-1)
    lateral = (off - step.unsqueeze(-1) * chosen).norm(dim=-1)
    usable = ray_limits(lidar, 3.0, 0.05, 9.0).gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    assert float(lateral.max()) < 1e-5, float(lateral.max())
    assert torch.all(step <= usable + 1e-5)
    assert float(step.max()) <= 3.0 + 1e-5
    print(f"decode: offsets stay on the chosen ray (lateral {float(lateral.max()):.1e}), "
          f"step <= min(lidar-eps, R_max)")


def test_ratio_is_bounded():
    lidar = torch.zeros(4, 6, 16)
    path = torch.zeros(4, 6, 2)
    onehot = torch.zeros(4, 6, 16)
    onehot[..., 0] = 1.0
    ratio = torch.rand(4, 6)
    center = decode_center(path, lidar, onehot, ratio, 3.0, 0.05, 9.0)
    assert torch.allclose(center, path), "zero lidar must forbid any offset"
    print("decode: zero-range rays pin the centre to the path point")


if __name__ == "__main__":
    test_axes_bounds()
    test_angle_and_shape_roundtrip()
    test_ray_decode_constraint()
    test_ratio_is_bounded()
    print("test_ellipse_decode: OK")
