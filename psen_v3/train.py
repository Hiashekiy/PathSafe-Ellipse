"""Train PathSafe-EllipseNet (spec V3) on the V3 trajectory dataset + v3sdf version.

    python psen_v3/train.py --tag full --epochs 3 --batch-size 4
    python psen_v3/train.py --overfit 12 --tag overfit --epochs 40     # stage B

Splits are by ``map_group`` only.  Loss components are logged separately because the
plan forbids judging progress by the total loss alone; the checkpoint is selected on
the held-out geometric metrics (exact collision rate, then continuity), not on loss.

Mixed precision is used for the CNN/transformers, while the geometric tensors (SDF
queries, ellipse sampling, loss accumulation) stay in float32 (plan section 12.5).
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from psen_v3.config import load_config, git_commit  # noqa: E402
from psen_v3.data.dataset import (build_loader, load_index, split_by_group)  # noqa: E402
from psen_v3.evaluation.runner import run_evaluation  # noqa: E402
from psen_v3.geometry.losses import GeometricLoss  # noqa: E402
from psen_v3.models.psenet import PSENet, count_parameters  # noqa: E402

LOG_COLUMNS = ["step", "epoch", "lr", "tau", "total", "choice", "conn", "raw_dist",
               "raw_safe", "raw_area", "raw_conn", "s_center_mean", "area_mean",
               "pi_entropy", "pi_max"]


def lr_at(step: int, total: int, base_lr: float, warmup_frac: float) -> float:
    warm = max(int(total * warmup_frac), 1)
    if step < warm:
        return base_lr * (step + 1) / warm
    prog = (step - warm) / max(total - warm, 1)
    return base_lr * 0.5 * (1.0 + np.cos(np.pi * min(prog, 1.0)))


def move(batch: Dict, device: torch.device) -> Dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def select_split(entries: List[Dict], cfg: Dict, overfit: int = 0):
    frac = tuple(float(v) for v in cfg["data"]["split_frac"])
    tr, va, te = split_by_group(entries, frac=frac, seed=int(cfg["data"]["split_seed"]))
    return tr, va, te


def filter_by_points(entries: List[Dict], indices: List[int], max_points: int) -> List[int]:
    return [i for i in indices if max_points <= 0 or int(entries[i]["n_points"]) <= max_points]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="")
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--overfit", type=int, default=0,
                    help="stage B: train on N short trajectories from the train split")
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="cap steps per epoch")
    ap.add_argument("--val-limit", type=int, default=0)
    ap.add_argument("--exact-val-traj", type=int, default=192)
    ap.add_argument("--device", default="")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--ablate", default="", help="comma list: occ_point,ray_attn,path_attn,occ_pos_encoding")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--num-threads", type=int, default=0)
    ap.add_argument("--set", action="append", default=[],
                    help="config override section__key=value (repeatable)")
    ap.add_argument("--eval-overfit", action="store_true",
                    help="evaluate on the overfit trajectories instead of the val split")
    ap.add_argument("--resume", default="", help="checkpoint to continue from")
    args = ap.parse_args()

    cfg = load_config(args.config or None)
    for item in args.set:
        key, _, value = item.partition("=")
        section, _, name = key.partition("__")
        if not (section and name):
            raise SystemExit(f"--set needs section__key=value, got '{item}'")
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = value
        cfg[section][name] = parsed
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["train"]["lr"] = args.lr
    if args.device:
        cfg["train"]["device"] = args.device
    if args.val_limit:
        cfg["train"]["val_limit"] = args.val_limit
    if args.out_dir:
        cfg["train"]["out_dir"] = os.path.abspath(args.out_dir)
    if args.no_augment:
        cfg["train"]["augment"] = False
    if args.ablate:
        for k in args.ablate.split(","):
            cfg["model"]["ablate"][k.strip()] = True
    if args.num_threads:
        torch.set_num_threads(args.num_threads)

    seed = int(cfg["train"]["seeds"][0] if args.seed < 0 else args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    entries = load_index(cfg["data"]["sdf_dir"])
    tr_idx, va_idx, te_idx = select_split(entries, cfg, args.overfit)
    max_pts = int(cfg["data"].get("max_points", 256))
    tr_idx = filter_by_points(entries, tr_idx, max_pts)
    va_idx = filter_by_points(entries, va_idx, max_pts)
    te_idx = filter_by_points(entries, te_idx, max_pts)
    if args.overfit:
        cand = [i for i in tr_idx
                if 8 <= int(entries[i]["n_points"]) <= 40]
        rng = np.random.default_rng(seed)
        tr_idx = [cand[j] for j in rng.choice(len(cand), size=min(args.overfit, len(cand)),
                                              replace=False)]
    if args.limit_train:
        tr_idx = tr_idx[: args.limit_train]
    print(f"trajectories {len(entries)} | train {len(tr_idx)} | val {len(va_idx)} | "
          f"test {len(te_idx)} | map groups "
          f"{len({e['map_group'] for e in entries})}", flush=True)

    device = torch.device(cfg["train"]["device"] if cfg["train"]["device"] != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"device {device} | torch {torch.__version__}", flush=True)

    bsz = int(cfg["train"]["batch_size"])
    train_loader = build_loader([entries[i] for i in tr_idx], cfg["data"]["trajectory_dir"],
                                cfg["data"]["sdf_dir"], bsz, cfg,
                                augment=bool(cfg["train"]["augment"]), shuffle=True)
    if args.eval_overfit:
        val_files = [entries[i] for i in tr_idx][: int(cfg["train"]["val_limit"])]
    else:
        val_files = [entries[i] for i in va_idx][: int(cfg["train"]["val_limit"])]
    val_loader = (build_loader(val_files, cfg["data"]["trajectory_dir"],
                               cfg["data"]["sdf_dir"], bsz, cfg, augment=False,
                               shuffle=False) if val_files else None)

    model = PSENet(cfg).to(device)
    n_par = count_parameters(model)
    print(f"model params {n_par / 1e6:.2f} M", flush=True)
    criterion = GeometricLoss(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]),
                              weight_decay=float(cfg["train"]["weight_decay"]))
    amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    tag = args.tag or f"seed{seed}"
    out_dir = os.path.join(cfg["train"]["out_dir"], tag)
    os.makedirs(out_dir, exist_ok=True)
    start_epoch = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        start_epoch = int(ck.get("epoch", -1)) + 1
        print(f"resumed from {args.resume} at epoch {start_epoch}", flush=True)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "run_meta.json"), "w", encoding="utf-8") as fh:
        json.dump({"git_commit": git_commit(), "seed": seed, "params": n_par,
                   "data_version": os.path.basename(cfg["data"]["sdf_dir"]),
                   "trajectory_dir": cfg["data"]["trajectory_dir"],
                   "n_train": len(tr_idx), "n_val": len(va_idx), "n_test": len(te_idx),
                   "torch": torch.__version__, "device": str(device),
                   "map_groups": sorted({e["map_group"] for e in entries}),
                   "split_seed": int(cfg["data"]["split_seed"])},
                  fh, indent=2)

    steps_per_epoch = len(train_loader)
    if args.max_steps:
        steps_per_epoch = min(steps_per_epoch, args.max_steps)
    epochs = int(cfg["train"]["epochs"])
    total_steps = max(steps_per_epoch * epochs, 1)
    csv_file = open(os.path.join(out_dir, "log.csv"), "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(LOG_COLUMNS)

    best = {"score": -1e9, "epoch": -1}
    step = start_epoch * steps_per_epoch          # keep the LR schedule continuous
    t0 = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        model.train()
        # reshuffle the scene-grouped order every epoch, otherwise a second pass
        # replays exactly the same batches (flips are scene-stable by design)
        sampler = getattr(train_loader, "batch_sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        running: Dict[str, List[float]] = {}
        for it, batch in enumerate(train_loader):
            if args.max_steps and it >= args.max_steps:
                break
            lr = lr_at(step, total_steps, float(cfg["train"]["lr"]),
                       float(cfg["train"]["warmup_frac"]))
            for g in optim.param_groups:
                g["lr"] = lr
            frac = step / max(total_steps - 1, 1)
            tau = float(cfg["train"]["tau_start"]) + frac * (
                float(cfg["train"]["tau_end"]) - float(cfg["train"]["tau_start"]))
            b = move(batch, device)
            try:
                with torch.amp.autocast(device_type=device.type, enabled=amp):
                    out = model(b, tau=tau)
                out32 = {k: (v.float() if torch.is_tensor(v) else v) for k, v in out.items()}
                loss, logs = criterion(out32, b)
                optim.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               float(cfg["train"]["grad_clip"]))
                scaler.step(optim)
                scaler.update()
            except torch.cuda.OutOfMemoryError:
                optim.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print(f"  !! CUDA OOM at step {step}: batch skipped", flush=True)
                step += 1
                continue
            for k, v in logs.items():
                running.setdefault(k, []).append(v)
            if step % int(cfg["train"]["log_every"]) == 0:
                mean = {k: float(np.mean(v[-25:])) for k, v in running.items()}
                el = (time.perf_counter() - t0) / 60.0
                print(f"[{el:6.1f}m] ep {epoch} step {step}/{total_steps} lr {lr:.2e} "
                      f"tau {tau:.2f} | total {mean['total']:.3f} "
                      f"choice {mean['choice']:.3f} conn {mean['conn']:.2f} "
                      f"| raw d {mean['raw_dist']:.2f} s {mean['raw_safe']:.3f} "
                      f"a {mean['raw_area']:.2f} c {mean['raw_conn']:.2f} "
                      f"| S(q) {mean['s_center_mean']:.2f} area {mean['area_mean']:.2f} "
                      f"H {mean['pi_entropy']:.2f} pmax {mean['pi_max']:.3f}", flush=True)
                writer.writerow([step, epoch, f"{lr:.6e}", f"{tau:.4f}"]
                                + [f"{mean.get(c, float('nan')):.6f}" for c in
                                   LOG_COLUMNS[4:]])
                csv_file.flush()
            step += 1

        if val_loader is not None:
            metrics = run_evaluation(model, criterion, val_loader, device, cfg,
                                     exact=True,
                                     max_exact_traj=int(args.exact_val_traj), amp=amp)
            metrics.pop("trajectories", None)
            safe = metrics.get("exact_safe_rate") or 0.0
            cont = metrics.get("continuity_pair_rate") or 0.0
            score = safe + 0.1 * cont
            print(f"  == val ep {epoch}: exact safe {safe:.4f} "
                  f"cont pair {cont:.4f} | S(q) mean "
                  f"{metrics['selected_center_clearance']['sdf_mean']:.2f} "
                  f"p10 {metrics['selected_center_clearance']['sdf_p10']:.2f} | "
                  f"area p50 {metrics['ellipse_area_m2']['p50']:.2f} | "
                  f"border {metrics['border_violation_rate']:.4f} | "
                  f"H {metrics['direction_entropy']['mean']:.2f} | "
                  f"gt centre med {metrics['gt_center_error_m']['p50']}", flush=True)
            with open(os.path.join(out_dir, "val_metrics.json"), "w", encoding="utf-8") as fh:
                json.dump(metrics, fh, indent=2, ensure_ascii=False, default=float)
            if score > best["score"]:
                best = {"score": score, "epoch": epoch, "metrics": metrics}
                torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch,
                            "step": step, "metrics": metrics,
                            "seed": seed},
                           os.path.join(out_dir, "best.pt"))
                print(f"  -> new best (score {score:.4f}) saved", flush=True)

        torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "step": step},
                   os.path.join(out_dir, "last.pt"))
    csv_file.close()
    print(f"training done in {(time.perf_counter() - t0) / 60:.1f} min -> {out_dir}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

