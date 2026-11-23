"""Evaluate a trained ``psen_v3`` checkpoint on a held-out split.

    python psen_v3/evaluate.py --checkpoint runs_v3sdf/full/best.pt --split test

Writes ``<run_dir>/eval_<split>.json`` with the metrics of plan section 14 and, with
``--dump``, the per-point predictions of a few trajectories for plotting.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from psen_v3.data.dataset import build_loader, load_index, split_by_group  # noqa: E402
from psen_v3.evaluation.runner import run_evaluation  # noqa: E402
from psen_v3.geometry.losses import GeometricLoss  # noqa: E402
from psen_v3.models.psenet import PSENet  # noqa: E402

KEYS = ("exact_safe_rate", "exact_collision_rate", "continuity_pair_rate",
        "boundary_cover_rate", "corridor_success_rate", "border_violation_rate",
        "sdf_safety_rate", "argmax_vs_min_loss_agreement")


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = PSENet(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=0, help="cap trajectories")
    ap.add_argument("--exact-traj", type=int, default=0, help="cap exact (NumPy) checks")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", default="")
    ap.add_argument("--dump", default="", help="npz path for per-point predictions")
    ap.add_argument("--select", default="argmax", choices=["argmax", "minloss"],
                    help="argmax = inference rule; minloss = offline oracle diagnostic")
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg, ckpt = load_model(args.checkpoint, device)
    entries = load_index(cfg["data"]["sdf_dir"])
    tr, va, te = split_by_group(entries, frac=tuple(cfg["data"]["split_frac"]),
                                seed=int(cfg["data"]["split_seed"]))
    pick = {"train": tr, "val": va, "test": te}[args.split]
    max_pts = int(cfg["data"].get("max_points", 256))
    pick = [i for i in pick if int(entries[i]["n_points"]) <= max_pts]
    if args.limit:
        pick = pick[: args.limit]
    files = [entries[i] for i in pick]
    print(f"{args.split}: {len(files)} trajectories "
          f"({len({e['map_group'] for e in files})} map groups)", flush=True)

    loader = build_loader(files, cfg["data"]["trajectory_dir"], cfg["data"]["sdf_dir"],
                          args.batch_size, cfg, augment=False, shuffle=False)
    crit = GeometricLoss(cfg).to(device)
    metrics = run_evaluation(model, crit, loader, device, cfg, exact=True,
                             max_exact_traj=args.exact_traj, select=args.select)
    traj = metrics.pop("trajectories", [])

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)),
                                   f"eval_{args.split}.json")
    payload = {"checkpoint": os.path.abspath(args.checkpoint), "split": args.split,
               "epoch": ckpt.get("epoch"), "n_trajectories": len(files),
               "metrics": metrics}
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=float)

    print(json.dumps({k: metrics.get(k) for k in KEYS}, indent=1, default=float))
    for key in ("selected_center_clearance", "ellipse_area_m2", "direction_entropy",
                "gt_center_error_m", "gt_ellipse_iou", "gt_area_ratio"):
        print(f"{key}: {json.dumps(metrics.get(key), default=float)}")

    if args.dump and traj:
        rows = {}
        for i, r in enumerate(traj):
            rows[f"pred_center_{i}"] = r["pred"]["center"]
            rows[f"pred_a_{i}"] = r["pred"]["a"]
            rows[f"pred_b_{i}"] = r["pred"]["b"]
            rows[f"pred_theta_{i}"] = r["pred"]["theta"]
        np.savez_compressed(args.dump, **rows)
        print(f"predictions -> {args.dump}")
    print(f"metrics -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
