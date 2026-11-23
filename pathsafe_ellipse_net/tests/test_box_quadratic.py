"""S1 acceptance: torch box quadratic minimum == NumPy reference.

Run: ``python pathsafe_ellipse_net/tests/test_box_quadratic.py``
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from pathsafe_ellipse.v2.ellipse import box_quadratic_min as box_min_np  # noqa: E402
from pathsafe_ellipse_net.geometry import box_quadratic_min_torch  # noqa: E402


def random_pd(rng, n, scale=2.0):
    A = rng.normal(size=(n, 2, 2)) * scale
    return np.einsum("nij,nkj->nik", A, A) + 0.05 * np.eye(2)


def test_matches_numpy():
    """The NumPy reference is vectorised over *boxes* and single-matrix, so compare
    one (M, q) at a time against a scalar box."""
    rng = np.random.default_rng(0)
    for trial in range(20):
        n = 64
        M = random_pd(rng, n)
        q = rng.normal(size=(n, 2)) * 1.5
        centre = rng.normal(size=(n, 2)) * 2.0
        half = 0.05 + rng.random(n) * 0.3
        ref = np.array([
            float(box_min_np(M[i], q[i], centre[i, 0] - half[i], centre[i, 0] + half[i],
                             centre[i, 1] - half[i], centre[i, 1] + half[i]))
            for i in range(n)])
        got = box_quadratic_min_torch(
            torch.tensor(M), torch.tensor(q),
            torch.tensor(centre[:, 0] - half), torch.tensor(centre[:, 0] + half),
            torch.tensor(centre[:, 1] - half), torch.tensor(centre[:, 1] + half))
        err = np.max(np.abs(got.detach().numpy() - ref))
        assert err < 1e-8, f"trial {trial}: max abs err {err}"
    print("box quadratic: torch == numpy on 20 x 64 random cases")


def test_center_inside_box_is_zero():
    M = torch.eye(2).double()[None]
    q = torch.tensor([[0.0, 0.0]], dtype=torch.double)
    zero = torch.zeros(1, dtype=torch.double)
    one = torch.ones(1, dtype=torch.double)
    got = box_quadratic_min_torch(M, q, -one, one, -one, one)
    assert torch.allclose(got, zero), got
    print("box quadratic: centre inside the square gives exactly 0")


def test_gradient_flow():
    rng = np.random.default_rng(1)
    M = random_pd(rng, 4)
    q = rng.normal(size=(4, 2))
    half = np.full(4, 0.1)
    Mt = torch.tensor(M, requires_grad=True)
    qt = torch.tensor(q, requires_grad=True)
    ht = torch.tensor(half)
    centre = torch.tensor(rng.normal(size=(4, 2)))
    out = box_quadratic_min_torch(Mt, qt, centre[:, 0] - ht, centre[:, 0] + ht,
                                 centre[:, 1] - ht, centre[:, 1] + ht)
    out.sum().backward()
    assert Mt.grad is not None and torch.isfinite(Mt.grad).all()
    assert qt.grad is not None and torch.isfinite(qt.grad).all()
    print("box quadratic: finite gradients wrt M and q")


def test_finite_difference():
    """Check d(phi)/dq against central differences away from the piecewise seams."""
    rng = np.random.default_rng(3)
    M = random_pd(rng, 1)
    q = rng.normal(size=(1, 2)) * 0.5
    centre = rng.normal(size=(1, 2))
    half = 0.15
    eps = 1e-6

    def f(qq):
        return float(box_min_np(M[0], qq[0], float(centre[0, 0] - half),
                                float(centre[0, 0] + half),
                                float(centre[0, 1] - half),
                                float(centre[0, 1] + half)))

    qt = torch.tensor(q, requires_grad=True)
    ht = torch.tensor([half])
    out = box_quadratic_min_torch(torch.tensor(M), qt,
                                 torch.tensor(centre[:, 0]) - ht,
                                 torch.tensor(centre[:, 0]) + ht,
                                 torch.tensor(centre[:, 1]) - ht,
                                 torch.tensor(centre[:, 1]) + ht)
    out.sum().backward()
    grad = qt.grad.detach().numpy()[0]
    num = np.zeros(2)
    for k in range(2):
        dq = np.zeros(2)
        dq[k] = eps
        num[k] = (f(q + dq[None]) - f(q - dq[None])) / (2 * eps)
    err = np.max(np.abs(grad - num))
    assert err < 1e-4, f"gradient mismatch {grad} vs {num} (err {err})"
    print(f"box quadratic: finite-difference gradient agrees (err {err:.2e})")


if __name__ == "__main__":
    test_matches_numpy()
    test_center_inside_box_is_zero()
    test_gradient_flow()
    test_finite_difference()
    print("test_box_quadratic: OK")
