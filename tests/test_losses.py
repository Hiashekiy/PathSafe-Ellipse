"""Stage-A test: geometric losses, their gradients and the softmax direction gradient.

Plan 12.3 items covered here:

* SDF query gradient (bilinear) is the analytic one;
* ``L_dist`` / ``L_safe`` / ``L_area`` / ``L_conn`` respond correctly to centre,
  axes, angle and relative position;
* intersecting vs separated ellipses give the expected connection loss;
* the probability-weighted choice yields ``dL/dl_ij = pi_ij/tau (L_ij - <L>_i)``
  (plan section 11), i.e. lower-cost rays get a positive gradient;
* masked (padded) points and padded adjacent edges never contribute.

    python tests/test_losses.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ellipsenet.config import load_config  # noqa: E402
from ellipsenet.geometry.losses import GeometricLoss  # noqa: E402
from ellipsenet.geometry.sampling import (ellipse_points, quadratic_form,  # noqa: E402
                                       sample_sdf, unit_ellipse_template)

FAILS = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}{(' :: ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def ramp_sdf(H=128, W=128, res=0.25, x0=None):
    """``S(x, y) = x0 - x`` (a wall normal to x), sampled on the cell grid."""
    x0 = (W // 2) * res if x0 is None else x0
    c0 = x0 / res
    cols = np.arange(W, dtype=np.float32)
    sdf = (c0 - cols)[None, :].repeat(H, 0) * res
    return torch.from_numpy(sdf)[None, None], x0


def base_cfg(rays=8):
    cfg = load_config()
    cfg["data"]["lidar_rays"] = rays
    cfg["model"]["ellipse_samples_radii"] = [0.0, 0.5, 1.0]
    cfg["model"]["ellipse_samples_counts"] = [1, 4, 8]
    cfg["model"]["conn_boundary_samples"] = 12
    cfg["loss"]["w_area"] = 0.0
    return cfg


def one_candidate_out(B=1, N=1, R=8, q=None, a=None, b=None, theta=None, pi=None):
    q = torch.tensor([[[[0.0, 0.0]] * R]]) if q is None else q
    a = torch.full((B, N), 1.0) if a is None else a
    b = torch.full((B, N), 0.5) if b is None else b
    theta = torch.zeros(B, N) if theta is None else theta
    pi = torch.full((B, N, R), 1.0 / R) if pi is None else pi
    return {"q": q, "a": a, "b": b, "theta": theta, "pi": pi}


def batch_of(sdf, res=0.25, N=1, R=8, mask=None):
    H, W = sdf.shape[-2:]
    m = torch.ones(1, N, dtype=torch.bool) if mask is None else mask
    return {"sdf": sdf, "r_scene": torch.tensor([res]), "mask": m,
            "lidar": torch.zeros(1, N, R), "path_xy": torch.zeros(1, N, 2)}


def test_sdf_gradient_and_dist() -> None:
    sdf, x0 = ramp_sdf()                     # x0 = 16 m, map spans x in [0, 31.75]
    crit = GeometricLoss(base_cfg())
    qx, qy = 10.0, 16.0                      # inside the map, free side
    q = torch.zeros(1, 1, 8, 2, requires_grad=False)
    q[..., 0] = qx
    q[..., 1] = qy
    q.requires_grad_(True)
    out = one_candidate_out(q=q)
    terms = crit.local_terms(out, batch_of(sdf))
    l_dist = terms["dist"][0, 0, 0]
    expect = -min(max(x0 - qx, 0.0), crit.d_max)
    check("L_dist = -min(max(S,0),D)", abs(float(l_dist.detach()) - expect) < 1e-5,
          f"{float(l_dist):.4f} vs {expect:.4f}")
    l_dist.backward()
    g = float(q.grad[0, 0, 0, 0])
    check("dL_dist/dq_x == +1 (gradient descent moves away from the wall)",
          abs(g - 1.0) < 1e-4, f"{g:.5f}")

    # far from the wall the clearance term saturates at -D_max
    q2 = torch.zeros(1, 1, 8, 2)
    q2[..., 0] = 1.0                         # deep in free space: S = 15 m > D_max
    q2[..., 1] = qy
    t2 = crit.local_terms(one_candidate_out(q=q2), batch_of(sdf))
    check("clearance clamped at -D_max far from the wall",
          abs(float(t2["dist"].max()) + crit.d_max) < 1e-5,
          f"{float(t2['dist'].max()):.2f} vs {-crit.d_max:.2f}")


def test_safe_gradient_fd() -> None:
    """Finite differences vs autograd for centre / axes / angle of the safety term."""
    sdf, _ = ramp_sdf()                           # S(x) = 16 - x, wall at x = 16 m
    crit = GeometricLoss(base_cfg())

    def loss_at(qx, a, b, th):
        q = torch.tensor([[[[qx, 16.0]] * 8]], requires_grad=False)
        out = one_candidate_out(q=q, a=torch.full((1, 1), a), b=torch.full((1, 1), b),
                                theta=torch.full((1, 1), th))
        return float(crit.local_terms(out, batch_of(sdf))["safe"][0, 0, 0])

    eps = 1e-4
    for name, kw, idx in (("q_x", "qx", 0), ("a", "a", 1), ("b", "b", 2), ("theta", "th", 3)):
        vals = [15.4, 1.2, 0.6, 0.3]
        vals[idx] += eps
        lp = loss_at(*vals)
        vals[idx] -= 2 * eps
        lm = loss_at(*vals)
        fd = (lp - lm) / (2 * eps)

        vals[idx] += eps
        q = torch.tensor([[[[vals[0], 16.0]] * 8]])
        q.requires_grad_(True)
        a = torch.full((1, 1), vals[1], requires_grad=True)
        b = torch.full((1, 1), vals[2], requires_grad=True)
        th = torch.full((1, 1), vals[3], requires_grad=True)
        out = one_candidate_out(q=q, a=a, b=b, theta=th)
        l = crit.local_terms(out, batch_of(sdf))["safe"][0, 0, 0]
        l.backward()
        autograds = {"qx": q.grad[0, 0, 0, 0], "a": a.grad, "b": b.grad, "th": th.grad}[kw]
        auto = float(autograds)
        ok = abs(fd - auto) <= 2e-2 * max(1.0, abs(auto)) + 1e-4
        check(f"L_safe d/d{name}: autograd matches finite differences", ok,
              f"auto {auto:.5f} vs fd {fd:.5f}")


def test_safe_penalises_collision() -> None:
    sdf, _ = ramp_sdf()                      # wall at x = 16 m, free for x < 16
    crit = GeometricLoss(base_cfg())
    # ellipse entirely in free space -> no penalty
    q_free = torch.tensor([[[[14.0, 16.0]] * 8]])
    l_free = crit.local_terms(one_candidate_out(q=q_free, a=torch.full((1, 1), 0.5),
                                                b=torch.full((1, 1), 0.3)),
                              batch_of(sdf))["safe"][0, 0, 0]
    # ellipse pushed into the blocked half -> positive penalty
    q_bad = torch.tensor([[[[16.5, 16.0]] * 8]])
    l_bad = crit.local_terms(one_candidate_out(q=q_bad, a=torch.full((1, 1), 1.0),
                                               b=torch.full((1, 1), 0.6)),
                             batch_of(sdf))["safe"][0, 0, 0]
    check("free ellipse has zero safety penalty", float(l_free.detach()) < 1e-6,
          f"{float(l_free):.2e}")
    check("colliding ellipse has a large penalty", float(l_bad.detach()) > 1e-2,
          f"{float(l_bad):.4f}")


def test_area_term() -> None:
    sdf, _ = ramp_sdf()
    cfg = base_cfg()
    cfg["loss"]["w_area"] = 1.0
    crit = GeometricLoss(cfg)
    small = crit.local_terms(one_candidate_out(a=torch.full((1, 1), 0.5),
                                               b=torch.full((1, 1), 0.3)), batch_of(sdf))["area"]
    big = crit.local_terms(one_candidate_out(a=torch.full((1, 1), 2.0),
                                             b=torch.full((1, 1), 1.0)), batch_of(sdf))["area"]
    check("area loss decreases as the ellipse grows", float(small) > float(big),
          f"{float(small):.3f} > {float(big):.3f}")
    check("area loss == -log(pi a b)",
          abs(float(small) + np.log(np.pi * 0.5 * 0.3)) < 1e-4,
          f"{float(small):.4f} vs {-np.log(np.pi * 0.15):.4f}")


def test_connection_loss() -> None:
    cfg = base_cfg()
    crit = GeometricLoss(cfg)
    R = 8
    # identical, coincident ellipses: every boundary point is on the other boundary
    q = torch.zeros(1, 2, R, 2)
    q[..., 1] = 16.0
    a = torch.full((1, 2), 1.0)
    b = torch.full((1, 2), 0.5)
    th = torch.zeros(1, 2)
    out = {"q": q, "a": a, "b": b, "theta": th, "pi": torch.full((1, 2, R), 1.0 / R)}
    same = crit.connection_terms(out)                             # (1,1,R,R)
    check("coincident ellipses -> ~0 connection loss", float(same.max().detach()) < 1e-5,
          f"{float(same.max()):.2e}")

    # neighbours 5 m apart with 1 m axes: far outside each other
    q2 = torch.zeros(1, 2, R, 2)
    q2[..., 1] = 16.0
    q2[0, 1, :, 0] = 5.0
    out2 = dict(out, q=q2)
    far = crit.connection_terms(out2)
    check("separated ellipses -> positive connection loss", float(far.min().detach()) > 1.0,
          f"min {float(far.min()):.3f}")

    # monotone in separation
    vals = []
    for d in (0.0, 0.5, 1.0, 2.0, 4.0):
        q3 = torch.zeros(1, 2, R, 2)
        q3[..., 1] = 16.0
        q3[0, 1, :, 0] = d
        vals.append(float(crit.connection_terms(dict(out, q=q3)).mean().detach()))
    check("connection loss is monotone in separation",
          all(vals[i] <= vals[i + 1] + 1e-6 for i in range(len(vals) - 1)),
          str([round(v, 3) for v in vals]))


def test_direction_gradient_identity() -> None:
    """``dL/dl_ij = pi_ij/tau [L_ij - sum_k pi_ik L_ik]`` (plan section 11)."""
    tau = 0.7
    B, N, R = 2, 3, 8
    logits = torch.randn(B, N, R, requires_grad=True)
    pi = torch.softmax(logits / tau, dim=-1)
    l_local = torch.tensor(np.random.default_rng(0).normal(size=(B, N, R)), dtype=torch.float32)
    loss = (pi * l_local).sum(-1).mean()
    loss.backward()
    mean_l = (pi * l_local).sum(-1, keepdim=True)
    expect = pi / tau * (l_local - mean_l) / (B * N)
    err = float((logits.grad - expect).abs().max())
    check("softmax-choice gradient identity", err < 1e-6, f"max err {err:.2e}")
    sign_ok = bool(((logits.grad * (l_local - mean_l) > 0).all()))
    check("lower-cost rays get a positive (loss-reducing) logit gradient", sign_ok)


def test_full_loss_masking() -> None:
    """Values at a masked point must not influence the loss (same mask on both runs)."""
    sdf, _ = ramp_sdf()
    cfg = base_cfg()
    crit = GeometricLoss(cfg)
    R = 8
    B, N = 1, 4
    g = torch.Generator().manual_seed(0)
    out = {"q": torch.rand(B, N, R, 2, generator=g) * 2 - 1,
           "a": torch.rand(B, N, generator=g) * 1.5 + 0.5,
           "b": torch.rand(B, N, generator=g) * 0.5 + 0.2,
           "theta": torch.rand(B, N, generator=g) * 3.0,
           "pi": torch.softmax(torch.randn(B, N, R, generator=g), -1)}
    out["q"][..., 1] += 16.0
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[0, -1] = False
    l_ref, _ = crit({k: v.clone() for k, v in out.items()},
                    batch_of(sdf, N=N, R=R, mask=mask))

    out_absurd = {k: v.clone() for k, v in out.items()}
    out_absurd["a"][0, -1] = cfg["model"]["shape_a_max_m"]
    out_absurd["b"][0, -1] = cfg["model"]["shape_a_max_m"]
    out_absurd["q"][0, -1] = 5.0
    out_absurd["pi"][0, -1] = torch.eye(R)[0]
    l_absurd, _ = crit(out_absurd, batch_of(sdf, N=N, R=R, mask=mask))
    check("padded point values do not contribute to the loss",
          abs(float(l_ref.detach()) - float(l_absurd.detach())) < 1e-6,
          f"{float(l_ref):.5f} vs {float(l_absurd):.5f}")

    l_full, _ = crit({k: v.clone() for k, v in out.items()},
                     batch_of(sdf, N=N, R=R, mask=torch.ones(B, N, dtype=torch.bool)))
    check("unmasking the point changes the loss (mask is effective)",
          abs(float(l_full.detach()) - float(l_ref.detach())) > 1e-6,
          f"{float(l_full):.5f} vs {float(l_ref):.5f}")


def main() -> int:
    for fn in (test_sdf_gradient_and_dist, test_safe_gradient_fd, test_safe_penalises_collision,
               test_area_term, test_connection_loss, test_direction_gradient_identity,
               test_full_loss_masking):
        print(fn.__name__)
        fn()
    print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
