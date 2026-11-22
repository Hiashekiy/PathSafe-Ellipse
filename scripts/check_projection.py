"""Benchmark / sanity-check the inference-time safety guard."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pathsafe_ellipse.config import load_config, project_paths, setup_logging  # noqa: E402
from pathsafe_ellipse.evaluation import ellipse_iou_grid, rebuild_L_cells  # noqa: E402
from pathsafe_ellipse.model import EllipseNet  # noqa: E402
from pathsafe_ellipse.safe_projection import (  # noqa: E402
    from_parameters, project_to_safe, verify,
)
from pathsafe_ellipse.torch_dataset import SplitData  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg, "check_projection")
    pp = project_paths(cfg)
    res = float(cfg["map"]["resolution"])
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    ckpt = torch.load(args.ckpt or os.path.join(pp.outputs_dir, "checkpoints", "ellipse_net_best.pt"),
                      map_location=device, weights_only=False)
    model = EllipseNet(width=int(ckpt.get("width", 32))).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    data = SplitData(os.path.join(pp.splits_dir, f"{args.split}.npz"))
    idx = np.random.default_rng(0).permutation(data.n)[: args.n]
    batch = data.batch(idx)
    patches = batch["patch"][:, 0].astype(np.uint8)

    t0 = time.perf_counter()
    with torch.no_grad():
        tensors = {k: torch.from_numpy(batch[k]).to(device) for k in ("patch", "rays", "hits", "tangent")}
        center, ab, angle, _, _ = model.predict_ellipse(tensors)
    t_net = (time.perf_counter() - t0) / len(idx)
    center = center.cpu().numpy()
    ab = np.sort(ab.cpu().numpy(), axis=1)[:, ::-1]
    angle = angle.cpu().numpy()

    gt_center = batch["center_offset"]
    gt_ab = np.sort(batch["ab"], axis=1)[:, ::-1]
    gt_angle = np.arctan2(batch["sincos"][:, 0], batch["sincos"][:, 1])

    t_verify = t_proj = 0.0
    need = safe0 = safe1 = 0
    scales = []
    iou_before = ellipse_iou_grid((center, ab, angle), (gt_center, gt_ab, gt_angle), res)
    proj_ab = np.zeros_like(ab)
    proj_center = np.zeros_like(center)
    for i in range(len(idx)):
        L, c = from_parameters(ab[i, 0], ab[i, 1], angle[i], center[i], res)
        t = time.perf_counter()
        ok, _ = verify(L, c, patches[i])
        t_verify += time.perf_counter() - t
        safe0 += int(ok)
        t = time.perf_counter()
        sol = project_to_safe(L, c, patches[i])
        t_proj += time.perf_counter() - t
        safe1 += int(sol.safe)
        need += int(sol.scale < 0.999)
        scales.append(sol.scale)
        a_c, b_c = sol.semi_axes_cells
        ang_s = float(np.arctan2(sol.L[1, 0] - sol.L[0, 1], 0.0))
        # rebuild the axis angle from the shrunk shape matrix (orientation is preserved)
        vals, vecs = np.linalg.eigh(sol.L)
        v = vecs[:, int(np.argmax(vals))]
        if v[0] < 0:
            v = -v
        ang_s = float(np.arctan2(v[1], v[0]))
        proj_ab[i] = [a_c * res, b_c * res]
        proj_center[i] = (sol.center - np.array([64.0, 64.0])) * res
    iou_after = ellipse_iou_grid((proj_center, proj_ab, angle), (gt_center, gt_ab, gt_angle), res)

    report = {
        "n": int(len(idx)),
        "net_forward_ms_per_sample": round(t_net * 1e3, 3),
        "verify_ms_per_sample": round(t_verify / len(idx) * 1e3, 3),
        "project_ms_per_sample": round(t_proj / len(idx) * 1e3, 3),
        "collision_free_before": safe0 / len(idx),
        "collision_free_after": safe1 / len(idx),
        "shrink_needed_rate": need / len(idx),
        "mean_scale": float(np.mean(scales)),
        "mean_scale_when_needed": float(np.mean([s for s in scales if s < 0.999])) if need else 1.0,
        "iou_before": float(iou_before.mean()),
        "iou_after": float(iou_after.mean()),
    }
    print(json.dumps(report, indent=2))
    out = os.path.join(pp.reports_dir, "safe_projection_benchmark.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
