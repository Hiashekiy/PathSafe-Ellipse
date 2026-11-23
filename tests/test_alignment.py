"""Stage-A test: full-resolution map alignment and coordinate-query correctness.

Verifies (plan section 3, acceptance list):
1. both ResUNets return exactly the input resolution;
2. a feature cell ``(i, j)`` queried at its own metric centre returns that cell
   (no half-cell drift after the encoder/decoder round trip, ``align_corners``
   convention fixed project-wide);
3. bilinear interpolation halfway between two cells returns the mean;
4. points outside the map are reported as such and the SDF sampler returns the
   "outside == blocked" constant instead of the clamped border value.

    python tests/test_alignment.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ellipsenet.data.sdf import SDF_OUTSIDE_M  # noqa: E402
from ellipsenet.geometry.sampling import (inside_map, sample_feature, sample_sdf,  # noqa: E402
                                       world_to_grid)
from ellipsenet.models.resunet import ResUNet  # noqa: E402

FAILS = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}{(' :: ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def test_output_resolution() -> None:
    torch.manual_seed(0)
    for hw in ((256, 256), (256, 208), (96, 160)):
        net = ResUNet(1, 64, base=8, depths=[1, 1, 1, 1])
        net.eval()
        x = torch.zeros(2, 1, *hw)
        with torch.no_grad():
            y = net(x)
        check(f"out shape {hw} -> {(py := tuple(y.shape))[2:]}", py == (2, 64, *hw), str(py))


def test_cell_centre_alignment() -> None:
    """A spike map sampled at the spike's own centre must return the spike value."""
    H = W = 64
    res = 0.25
    feat = torch.zeros(1, 1, H, W)
    r, c = 31, 17
    feat[0, 0, r, c] = 5.0
    pt = torch.tensor([[[c * res, r * res]]])                     # metric centre
    got = sample_feature(feat, pt, torch.tensor([res]))
    check("cell centre maps exactly onto its pixel", abs(float(got) - 5.0) < 1e-5,
          f"{float(got):.6f} vs 5.0")

    # one full cell away must be exactly zero (no leakage past the bilinear kernel)
    pt2 = torch.tensor([[[(c + 1) * res, r * res]]])
    got2 = sample_feature(feat, pt2, torch.tensor([res]))
    check("one cell away is exactly 0", abs(float(got2)) < 1e-5, f"{float(got2):.2e}")

    # half a cell away is the bilinear mean
    pt3 = torch.tensor([[[(c + 0.5) * res, r * res]]])
    got3 = sample_feature(feat, pt3, torch.tensor([res]))
    check("half cell away == 2.5 (bilinear mean)", abs(float(got3) - 2.5) < 1e-4,
          f"{float(got3):.5f}")

    g = world_to_grid(torch.tensor([[[0.0, 0.0]]]), torch.tensor([res]), (H, W))
    expect = 2.0 * (0.0 / res + 0.5) / W - 1.0
    check("normalized coordinate of the first cell centre",
          abs(float(g[0, 0, 0]) - expect) < 1e-6, f"{float(g[0, 0, 0]):.6f} vs {expect:.6f}")


def test_encoder_decoder_alignment() -> None:
    """A single bright input pixel must stay centred on the same output pixel."""
    torch.manual_seed(1)
    net = ResUNet(1, 8, base=8, depths=[1, 1, 1, 1])
    net.eval()
    H = W = 64
    res = 0.25
    x = torch.zeros(1, 1, H, W)
    r, c = 32, 32
    x[0, 0, r, c] = 1.0
    with torch.no_grad():
        y = net(x)
    energy = y[0].abs().sum(dim=0)                                # (H,W)
    peak = int(torch.argmax(energy))
    pr, pc = peak // W, peak % W
    check("output energy peaks at the input pixel", (pr, pc) == (r, c), f"peak {(pr, pc)}")
    # centroid of the response (gravity centre) must stay within a quarter cell
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    wsum = energy.sum()
    cy = float((energy * yy).sum() / wsum)
    cx = float((energy * xx).sum() / wsum)
    check("response centroid centred", abs(cy - r) < 5.0 and abs(cx - c) < 5.0,
          f"centroid ({cy:.2f}, {cx:.2f}) vs ({r}, {c})")


def test_sdf_outside() -> None:
    H = W = 32
    res = 0.5
    sdf = torch.full((1, 1, H, W), 3.0)                            # pretend all free
    inside_pt = torch.tensor([[[5.0, 5.0]]])
    outside_pt = torch.tensor([[[-2.0, 5.0]]])
    far_pt = torch.tensor([[[(W + 4) * res, 5.0]]])
    v_in = sample_sdf(sdf, inside_pt, torch.tensor([res]))
    v_out = sample_sdf(sdf, outside_pt, torch.tensor([res]))
    v_far = sample_sdf(sdf, far_pt, torch.tensor([res]))
    check("inside keeps the field value", abs(float(v_in) - 3.0) < 1e-6, f"{float(v_in):.3f}")
    check("left of the map reads the blocked constant",
          abs(float(v_out) - SDF_OUTSIDE_M) < 1e-6, f"{float(v_out):.3f}")
    check("right of the map reads the blocked constant",
          abs(float(v_far) - SDF_OUTSIDE_M) < 1e-6, f"{float(v_far):.3f}")
    m = inside_map(torch.tensor([[[0.0, 0.0], [(W - 1) * res, 0.0], [-1e-3, 0.0]]]),
                   torch.tensor([res]), (H, W))
    check("inside_map corners", m.tolist() == [[True, True, False]],
          str(m.shape) + " " + str(m.tolist()))

def test_batch_varying_resolution() -> None:
    """``r_scene`` differs per sample in the dataset, so the sampler must respect it."""
    H = W = 32
    feat = torch.zeros(2, 1, H, W)
    feat[0, 0, 10, 5] = 1.0
    feat[1, 0, 2, 20] = 1.0
    res = torch.tensor([0.25, 0.5])
    pts = torch.tensor([[[5 * 0.25, 10 * 0.25]], [[20 * 0.5, 2 * 0.5]]])
    got = sample_feature(feat, pts, res).reshape(2)
    check("per-sample resolution honoured", torch.allclose(got, torch.ones(2), atol=1e-6),
          str(got.tolist()))


def main() -> int:
    for fn in (test_output_resolution, test_cell_centre_alignment,
               test_encoder_decoder_alignment, test_sdf_outside,
               test_batch_varying_resolution):
        print(fn.__name__)
        fn()
    print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
