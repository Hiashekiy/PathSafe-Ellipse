"""Ablation programme of plan section 14 (in-process, no subprocess/shell).

Each variant is trained with the same reduced budget (fixed steps, same split, same
seed) and evaluated on the validation split with the exact (NumPy) checks, so the
comparison isolates the removed component:

    1 drop the Occupancy point feature         --ablate occ_point
    2 drop the ray self-attention              --ablate ray_attn
    3 drop the path self-attention             --ablate path_attn
    4 drop the centre-clearance loss           loss.w_dist = 0
    5 drop the ellipse safety loss             loss.w_safe = 0
    6 drop the connection loss                 loss.w_conn = 0
    7 drop the 9-token relative position enc.  --ablate occ_pos_encoding
    8 weighted expectation vs offline oracle   --select argmax | minloss (diagnostic)

    python scripts/v3_ablate.py --steps 800 --limit-train 4000 --out reports_v3/ablation_v3sdf.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from psen_v3.evaluate import load_model  # noqa: E402
from psen_v3.data.dataset import build_loader, load_index, split_by_group  # noqa: E402
from psen_v3.evaluation.runner import run_evaluation  # noqa: E402
from psen_v3.geometry.losses import GeometricLoss  # noqa: E402

VARIANTS = [
    ("full", []),
    ("no_occ_point", ["--ablate", "occ_point"]),
    ("no_ray_attn", ["--ablate", "ray_attn"]),
    ("no_path_attn", ["--ablate", "path_attn"]),
    ("no_occ_pos", ["--ablate", "occ_pos_encoding"]),
    ("no_dist_loss", ["--set", "loss__w_dist=0.0"]),
    ("no_safe_loss", ["--set", "loss__w_safe=0.0"]),
    ("no_conn_loss", ["--set", "loss__w_conn=0.0"]),
]

KEYS = ("exact_safe_rate", "boundary_cover_rate", "continuity_pair_rate",
        "selected_center_clearance", "ellipse_area_m2", "direction_entropy",
        "argmax_vs_min_loss_agreement", "border_violation_rate")


def train_variant(tag: str, extra: List[str], args) -> str:
    import psen_v3.train as trainer

    argv = ["train.py", "--tag", tag, "--epochs", "1",
            "--batch-size", str(args.batch_size),
            "--limit-train", str(args.limit_train),
            "--max-steps", str(args.steps),
            "--val-limit", "8", "--exact-val-traj", "2",
            "--out-dir", args.out_dir] + extra
    old = sys.argv
    sys.argv = argv
    try:
        print(f"\n=== ablation variant '{tag}' {extra}", flush=True)
        trainer.main()
    finally:
        sys.argv = old
    return os.path.join(args.out_dir, tag)


def evaluate_variant(run_dir: str, args, select: str = "argmax") -> Dict[str, object]:
    ckpt = os.path.join(run_dir, "last.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, _ck = load_model(ckpt, device)
    entries = load_index(cfg["data"]["sdf_dir"])
    _tr, va, _te = split_by_group(entries, frac=tuple(cfg["data"]["split_frac"]),
                                  seed=int(cfg["data"]["split_seed"]))
    pick = [i for i in va
            if int(entries[i]["n_points"]) <= int(cfg["data"]["max_points"])][: args.eval_traj]
    loader = build_loader([entries[i] for i in pick], cfg["data"]["trajectory_dir"],
                          cfg["data"]["sdf_dir"], args.batch_size, cfg,
                          augment=False, shuffle=False)
    crit = GeometricLoss(cfg).to(device)
    m = run_evaluation(model, crit, loader, device, cfg, exact=True,
                       max_exact_traj=args.eval_traj, select=select)
    m.pop("trajectories", None)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--limit-train", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--eval-traj", type=int, default=192)
    ap.add_argument("--out-dir", default="runs_v3sdf/ablation")
    ap.add_argument("--out", default="reports_v3/ablation_v3sdf.json")
    ap.add_argument("--only", default="", help="comma list of variant names")
    args = ap.parse_args()

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    results: Dict[str, object] = {}
    for tag, extra in VARIANTS:
        if only and tag not in only:
            continue
        run_dir = train_variant(tag, extra, args)
        metrics = evaluate_variant(run_dir, args)
        results[tag] = {k: metrics.get(k) for k in KEYS}

    # ablation 8: same checkpoint, argmax selection vs offline min-loss oracle
    base = os.path.join(args.out_dir, "full")
    if os.path.exists(os.path.join(base, "last.pt")):
        results["full__oracle_selection"] = {
            k: v for k, v in evaluate_variant(base, args, select="minloss").items()
            if k in KEYS}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False, default=float)

    lines = ["| variant | exact safe | boundary cover | cont pair | centre clearance p50 "
             "| area p50 | direction entropy | argmax=minloss |",
             "|---|---|---|---|---|---|---|---|"]
    for tag, m in results.items():
        cc = m.get("selected_center_clearance") or {}
        ar = m.get("ellipse_area_m2") or {}
        ent = m.get("direction_entropy") or {}
        lines.append(f"| {tag} | {m.get('exact_safe_rate'):.4f} "
                     f"| {m.get('boundary_cover_rate'):.4f} "
                     f"| {m.get('continuity_pair_rate'):.4f} "
                     f"| {cc.get('sdf_p50')} | {ar.get('p50')} "
                     f"| {ent.get('mean')} | {m.get('argmax_vs_min_loss_agreement')} |")
    md = os.path.splitext(args.out)[0] + ".md"
    with open(md, "w", encoding="utf-8") as fh:
        fh.write("# 消融实验（缩减预算，仅用于相对比较）\n\n")
        fh.write(f"- steps {args.steps}，训练轨迹 {args.limit_train}，batch {args.batch_size}，"
                 f"验证轨迹 {args.eval_traj}\n\n")
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"-> {args.out}\n-> {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
