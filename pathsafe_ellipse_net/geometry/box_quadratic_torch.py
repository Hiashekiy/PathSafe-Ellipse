"""PyTorch geometry: exact quadratic minimum over a rectangle, ellipse algebra.

``box_quadratic_min_torch`` reproduces the NumPy reference
(:func:`pathsafe_ellipse.v2.ellipse.box_quadratic_min`) with PyTorch ops only,
so the gradient flows into the shape matrix ``M = S^{-1}`` and the centre ``q``.
"""
from __future__ import annotations

import torch


def box_quadratic_min_torch(M: torch.Tensor, q: torch.Tensor, x0: torch.Tensor,
                            x1: torch.Tensor, y0: torch.Tensor,
                            y1: torch.Tensor) -> torch.Tensor:
    """Exact ``min_{x in [x0,x1]x[y0,y1]} (x-q)^T M (x-q)`` (differentiable).

    Parameters are broadcastable: ``M (...,2,2)``, ``q (...,2)``, boxes ``(...)``.
    The minimum of a convex quadratic over a rectangle lies inside, on an edge or
    in a corner; clamping the analytic edge minimiser covers all edge cases and
    the four corners are added explicitly so the result can be cross-checked
    against the NumPy implementation.
    """
    m11, m12, m22 = M[..., 0, 0], M[..., 0, 1], M[..., 1, 1]
    cx, cy = q[..., 0], q[..., 1]
    eps = 1e-12

    def evaluate(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dx = x - cx
        dy = y - cy
        return m11 * dx * dx + 2.0 * m12 * dx * dy + m22 * dy * dy

    inside = (cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1)
    best = torch.where(inside, evaluate(cx, cy), torch.full_like(evaluate(cx, cy), float("inf")))

    for xf in (x0, x1):
        y_star = torch.clamp(cy - m12 * (xf - cx) / m22.clamp_min(eps), y0, y1)
        best = torch.minimum(best, evaluate(xf, y_star))
        best = torch.minimum(best, evaluate(xf, y0))
        best = torch.minimum(best, evaluate(xf, y1))
    for yf in (y0, y1):
        x_star = torch.clamp(cx - m12 * (yf - cy) / m11.clamp_min(eps), x0, x1)
        best = torch.minimum(best, evaluate(x_star, yf))
        best = torch.minimum(best, evaluate(x0, yf))
        best = torch.minimum(best, evaluate(x1, yf))
    return best


# --------------------------------------------------------------------------- #
# Ellipse <-> shape matrix
# --------------------------------------------------------------------------- #
def rotation(theta: torch.Tensor) -> torch.Tensor:
    """``R(theta)`` with shape ``(..., 2, 2)``."""
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack([torch.stack([c, -s], dim=-1),
                        torch.stack([s, c], dim=-1)], dim=-2)


def shape_matrix(axes: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """``S = R diag(a^2, b^2) R^T`` for ``axes = [a, b]`` (a >= b)."""
    a2, b2 = axes[..., 0] ** 2, axes[..., 1] ** 2
    R = rotation(theta)
    D = torch.zeros_like(R)
    D[..., 0, 0] = a2
    D[..., 1, 1] = b2
    return R @ D @ R.transpose(-1, -2)


def shape_axes_angle(S: torch.Tensor):
    """``(axes [a,b], theta)`` of a 2x2 shape matrix (a >= b)."""
    S = 0.5 * (S + S.transpose(-1, -2))
    # closed-form 2x2 eigendecomposition (stable, differentiable)
    tr = S[..., 0, 0] + S[..., 1, 1]
    diff = S[..., 0, 0] - S[..., 1, 1]
    off = S[..., 0, 1]
    rad = torch.sqrt(0.25 * diff * diff + off * off)
    lmax = 0.5 * tr + rad
    lmin = 0.5 * tr - rad
    theta = 0.5 * torch.atan2(2.0 * off, diff)
    axes = torch.stack([torch.sqrt(lmax.clamp_min(0.0)),
                        torch.sqrt(lmin.clamp_min(0.0))], dim=-1)
    return axes, theta


def invert_shape(S: torch.Tensor) -> torch.Tensor:
    """``S^{-1}`` for symmetric positive definite 2x2 matrices."""
    a, b, c = S[..., 0, 0], S[..., 0, 1], S[..., 1, 1]
    det = (a * c - b * b).clamp_min(1e-18)
    inv = torch.empty_like(S)
    inv[..., 0, 0] = c / det
    inv[..., 0, 1] = -b / det
    inv[..., 1, 0] = -b / det
    inv[..., 1, 1] = a / det
    return inv


def ellipse_quadratic(S: torch.Tensor, q: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """``(x - q)^T S^{-1} (x - q)`` for ``points (..., 2)``."""
    d = points - q
    inv = invert_shape(S)
    return torch.einsum("...i,...ij,...j->...", d, inv, d)


def midpoint_metric(S: torch.Tensor, q: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """``rho = sqrt((m-q)^T S^{-1} (m-q))`` (contiguity metric of the generator)."""
    return torch.sqrt(ellipse_quadratic(S, q, m).clamp_min(0.0))
