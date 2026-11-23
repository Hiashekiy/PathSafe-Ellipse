"""Stage-A test: candidate geometry (plan 12.3 - candidates sit exactly on their rays) 
plus a full forward/backward smoke test of the network.

    python psen_v3/tests/test_candidates.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from psen_v3.config import load_config  # noqa: E402
from psen_v3.geometry.sampling import (ellipse_rotation, quadratic_form,  # noqa: E402
                                       unit_ellipse_template)
from psen_v3.models.psenet import PSENet, count_parameters  # noqa: E402

FAILS = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}{(' :: ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def tiny_cfg(H=64, W=64, rays=16) -> dict:
    cfg = load_config()
    cfg["model"].update({"ch": 32, "resunet_base": 8, "resunet_depths": [1, 1, 1, 1],
                         "mlp_hidden": 64, "ray_attn_layers": 1, "path_attn_layers": 1,
                         "attn_heads": 4, "ff_mult": 2})
    cfg["data"]["lidar_rays"] = rays
    return cfg


def make_batch(B=2, N=5, H=64, W=64, res=0.25, rays=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    ang = 2.0 * np.pi * np.arange(rays) / rays
    ray_dir = torch.tensor(np.stack([np.cos(ang), np.sin(ang)], -1), dtype=torch.float32)
    path = torch.rand(B, N, 2, generator=g) * 8.0 + 2.0
    lidar = torch.rand(B, N, rays, generator=g) * 9.0
    occ = (torch.rand(B, 1, H, W, generator=g) > 0.9).float()
    sdf = torch.rand(B, 1, H, W, generator=g) * 4.0 - 1.0
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[1, -2:] = False                                  # variable-length example
    return {"occupancy": occ, "sdf": sdf, "path_xy": path, "lidar": lidar,
            "ray_dir": ray_dir.expand(B, rays, 2).contiguous(),
            "ray_valid": (lidar < 8.9).float(), "mask": mask,
            "r_scene": torch.full((B,), float(res))}


def main() -> int:
    cfg = tiny_cfg()
    torch.manual_seed(0)
    model = PSENet(cfg)
    print(f"model params {count_parameters(model) / 1e6:.2f} M (tiny config)")
    batch = make_batch()
    out = model(batch, tau=1.0)

    q, path, lidar = out["q"], batch["path_xy"], batch["lidar"]
    rays = q.shape[2]
    ang = 2.0 * np.pi * np.arange(rays) / rays
    u = torch.tensor(np.stack([np.cos(ang), np.sin(ang)], -1), dtype=torch.float32)

    delta = q - path[:, :, None, :]                            # (B,N,R,2)
    ux = u[:, 0].view(1, 1, rays)
    uy = u[:, 1].view(1, 1, rays)
    cross = delta[..., 0] * uy - delta[..., 1] * ux
    check("candidate centres lie on their own ray (cross ~ 0)",
          float(cross.abs().max()) < 1e-5, f"max |cross| {float(cross.abs().max()):.2e}")
    along = delta[..., 0] * ux + delta[..., 1] * uy            # projection distance
    check("candidate centres move forward along the ray only",
          float(along.min()) >= -1e-6, f"min {float(along.min()):.3e}")
    expect = out["alpha"][:, :, None] * torch.clamp(lidar - model.ray_margin_m, min=0.0)
    check("offset length == alpha * max(d - m, 0)",
          torch.allclose(along, expect, atol=1e-5), f"max err {float((along - expect).abs().max()):.2e}")

    check("alpha in (0,1)", bool(((out["alpha"] > 0) & (out["alpha"] < 1)).all()),
          f"[{float(out['alpha'].min()):.3f}, {float(out['alpha'].max()):.3f}]")
    check("a >= b > 0", bool(((out["a"] >= out["b"] - 1e-6) & (out["b"] > 0)).all()))
    check("a <= a_max", bool((out["a"] <= cfg["model"]["shape_a_max_m"] + 1e-5).all()))
    check("pi sums to 1", torch.allclose(out["pi"].sum(-1),
                                         torch.ones_like(out["pi"].sum(-1)), atol=1e-5))
    check("padded points stay finite", bool(torch.isfinite(out["pi"][~batch["mask"]]).all()))

    # ellipse template points are inside/on the ellipse by construction
    tpl = unit_ellipse_template(cfg["model"]["ellipse_samples_radii"],
                                cfg["model"]["ellipse_samples_counts"])
    scale = tpl.view(1, 1, 1, -1, 2) * torch.stack([out["a"], out["b"]], -1)[:, :, None, None, :]
    rot = ellipse_rotation(out["theta"])[:, :, None, None]
    pts = torch.matmul(rot, scale.unsqueeze(-1)).squeeze(-1) + q.unsqueeze(3)
    rho = quadratic_form(pts, q.unsqueeze(3), out["a"][:, :, None, None],
                         out["b"][:, :, None, None], out["theta"][:, :, None, None])
    check("template points satisfy rho <= 1", float(rho.max()) <= 1.0 + 1e-5,
          f"max rho {float(rho.max()):.4f}")
    check("template covers the boundary", abs(float(rho.max()) - 1.0) < 1e-4,
          f"max rho {float(rho.max()):.6f}")

    # encode/decode round trip through the argmax rule
    dec = model.decode_argmax(out)
    check("argmax centre equals its own candidate",
          torch.allclose(dec["center"], torch.gather(q, 2, dec["index"][..., None, None]
                                                     .expand(-1, -1, 1, 2)).squeeze(2)),
          "")

    # backward: every head must receive a finite gradient
    out = model(batch, tau=1.0)
    loss = (out["pi"] * torch.log(out["pi"] + 1e-9)).sum() * -1e-3 \
        + out["q"].pow(2).mean() + out["a"].mean() + out["b"].mean() + out["theta"].sin().abs().mean()
    loss.backward()
    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    finite = all(torch.isfinite(g).all() for g in grads.values())
    check("all parameter gradients finite", finite, f"{len(grads)} tensors")
    for name in ("dir_head.weight", "ratio_head.weight", "shape_head.weight",
                 "occ_unet.head.bias", "dist_unet.head.bias", "ray_mlp.0.weight",
                 "point_mlp.0.weight", "path_attn.layers.0.self_attn.in_proj_weight"):
        g = dict(model.named_parameters()).get(name)
        got = (g.grad is not None and float(g.grad.abs().sum()) > 0)
        check(f"gradient reaches {name}", got)

    # ablation switches must not crash and must change the output
    base = out["pi"].detach().clone()
    for flag in ("occ_point", "ray_attn", "path_attn", "occ_pos_encoding"):
        cfg2 = tiny_cfg()
        cfg2["model"]["ablate"] = {flag: True}
        torch.manual_seed(0)
        m2 = PSENet(cfg2)
        m2.load_state_dict(model.state_dict(), strict=True)
        o2 = m2(batch, tau=1.0)
        check(f"ablation '{flag}' runs and changes the prediction",
              float((o2["pi"] - base).abs().max()) > 0.0,
              f"max delta {float((o2['pi'] - base).abs().max()):.3e}")

    print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
