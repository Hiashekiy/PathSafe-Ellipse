"""Evaluate a trained PathSafe-EllipseNet checkpoint on a held-out split.

    python pathsafe_ellipse_net/evaluate.py --checkpoint runs_v3/seed0/best.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from pathsafe_ellipse_net.datasets import list_trajectories, split_by_group  # noqa: E402
from pathsafe_ellipse_net.evaluation.runner import run_evaluation  # noqa: E402
from pathsafe_ellipse_net.models import EllipseNet  # noqa: E402
from pathsafe_ellipse_net.train import groups_for, make_loader, resolve_device  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--trajectory-dir", default="")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out", default="")
    ap.add_argument("--max-trajectories", type=int, default=600)
    args = ap.parse_args()

    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = blob["cfg"]
    if args.trajectory_dir:
        cfg["data"]["trajectory_dir"] = os.path.abspath(args.trajectory_dir)
    files = list_trajectories(cfg["data"]["trajectory_dir"])
    groups = groups_for(files)
    tr, va, te = split_by_group(groups, frac=tuple(cfg["data"]["split_frac"]),
                                seed=int(cfg["data"]["split_seed"]))
    idx = {"train": tr, "val": va, "test": te}[args.split]
    sel = [files[i] for i in idx][: args.max_trajectories]
    if not sel:
        print("empty split")
        return 1

    device = resolve_device(cfg["train"]["device"])
    model = EllipseNet(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    loader = make_loader(cfg, sel, augment=False, shuffle=False)
    metrics = run_evaluation(model, loader, device,
                             rho_max=float(cfg["label"]["rho_max"]),
                             max_eval_trajectories=len(sel))
    metrics["split"] = args.split
    metrics["n_trajectories"] = len(sel)
    metrics["checkpoint"] = os.path.basename(args.checkpoint)
    out = args.out or os.path.splitext(args.checkpoint)[0] + f"_{args.split}_metrics.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    print(json.dumps({k: metrics[k] for k in ("n", "center_error", "ellipse_iou",
                                              "exact_safety")}, indent=2, default=float))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
