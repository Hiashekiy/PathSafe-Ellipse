"""Train PathSafe-EllipseNet on the V3 trajectory dataset.

    python pathsafe_ellipse_net/train.py --config pathsafe_ellipse_net/configs/train.yaml

Splits are by ``map_group`` (never by point).  Loss components are logged
separately (``center``/``axes``/``angle``/``ellipse_safe``/``center_free``/
``border``/``continuity``) because the plan forbids judging progress by the total
loss alone.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from pathsafe_ellipse_net.datasets import (SceneGroupedBatchSampler, TrajectoryDataset,
                                          collate_paths, list_trajectories, read_meta,
                                          split_by_group)
from pathsafe_ellipse_net.evaluation.runner import run_evaluation
from pathsafe_ellipse_net.losses import TotalLoss
from pathsafe_ellipse_net.models import EllipseNet


def load_cfg(path: str) -> Dict:
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for key, value in list(cfg.get("data", {}).items()):
        if isinstance(value, str) and value.startswith("./"):
            cfg["data"][key] = os.path.normpath(os.path.join(PROJECT_ROOT, value))
    out_dir = cfg["train"].get("out_dir", "")
    if isinstance(out_dir, str) and out_dir.startswith("./"):
        cfg["train"]["out_dir"] = os.path.normpath(os.path.join(PROJECT_ROOT, out_dir))
    return cfg


def resolve_device(name: str) -> torch.device:
    if name and name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def groups_for(files: List[str]) -> List[str]:
    """``map_group`` per file, taken from the dataset index when available."""
    index_path = os.path.join(os.path.dirname(files[0]), "index.json")
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as fh:
            entries = json.load(fh).get("entries", [])
        by_name = {e["file"]: e.get("map_group", "unknown") for e in entries}
        groups = [by_name.get(os.path.basename(f)) for f in files]
        if all(g is not None for g in groups):
            return groups
    return [str(read_meta(f).get("map_group", "unknown")) for f in files]


def scenes_for(files: List[str]) -> List[str]:
    """``scene_id`` per file (from the dataset index when available)."""
    index_path = os.path.join(os.path.dirname(files[0]), "index.json")
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as fh:
            entries = json.load(fh).get("entries", [])
        by_name = {e["file"]: e.get("scene_id", "?") for e in entries}
        scenes = [by_name.get(os.path.basename(f)) for f in files]
        if all(s is not None for s in scenes):
            return scenes
    return [str(read_meta(f).get("scene_id", "?")) for f in files]


def make_loader(cfg: Dict, files: List[str], augment: bool, shuffle: bool) -> DataLoader:
    """Scene-grouped batching: the map encoder runs once per scene (plan section 9)."""
    ds = TrajectoryDataset(files, cfg, augment=augment)
    nw = int(cfg["data"].get("num_workers", 0))
    sampler = SceneGroupedBatchSampler(
        scenes_for(files), batch_size=int(cfg["train"]["batch_size"]), shuffle=shuffle,
        seed=int(cfg["data"]["split_seed"]))
    # pin_memory only makes sense with worker processes; with workers=0 the batch is
    # already in host memory and pinning just costs another copy (memory pressure)
    return DataLoader(ds, batch_sampler=sampler, num_workers=nw,
                      collate_fn=collate_paths, persistent_workers=nw > 0,
                      pin_memory=torch.cuda.is_available() and nw > 0)


def lr_at(step: int, total: int, base_lr: float, warmup_frac: float) -> float:
    warm = max(int(total * warmup_frac), 1)
    if step < warm:
        return base_lr * (step + 1) / warm
    prog = (step - warm) / max(total - warm, 1)
    return base_lr * 0.5 * (1.0 + np.cos(np.pi * min(prog, 1.0)))


def tau_at(step: int, total: int, t0: float, t1: float) -> float:
    return float(t0 + (t1 - t0) * min(step / max(total, 1), 1.0))


def move(batch: Dict, device: torch.device) -> Dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(PROJECT_ROOT, "pathsafe_ellipse_net",
                                                     "configs", "train.yaml"))
    ap.add_argument("--trajectory-dir", default="")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--max-trajectories", type=int, default=0)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="")
    ap.add_argument("--num-workers", type=int, default=-1)
    ap.add_argument("--eval-every", type=int, default=0, help="0 = config value")
    ap.add_argument("--snapshot-every", type=int, default=0,
                    help="save epoch{NN}.pt every N epochs (0 = only best/last)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    cfg["model"]["geom_in"] = 7
    if args.trajectory_dir:
        cfg["data"]["trajectory_dir"] = os.path.abspath(args.trajectory_dir)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size
    if args.max_trajectories:
        cfg["data"]["max_trajectories"] = args.max_trajectories
    if args.seed >= 0:
        cfg["train"]["seeds"] = [args.seed]
    if args.device:
        cfg["train"]["device"] = args.device
    if args.num_workers >= 0:
        cfg["data"]["num_workers"] = args.num_workers
    if args.eval_every:
        cfg["train"]["eval_every"] = args.eval_every

    files = list_trajectories(cfg["data"]["trajectory_dir"])
    limit = cfg["data"].get("max_trajectories")
    if limit:
        files = files[: int(limit)]
    if len(files) < 10:
        print(f"only {len(files)} trajectories under {cfg['data']['trajectory_dir']}")
        return 1
    groups = groups_for(files)
    tr_idx, va_idx, te_idx = split_by_group(groups, frac=tuple(cfg["data"]["split_frac"]),
                                            seed=int(cfg["data"]["split_seed"]))
    print(f"trajectories {len(files)} | train {len(tr_idx)} | val {len(va_idx)} | "
          f"test {len(te_idx)} | map groups {len(set(groups))}", flush=True)

    device = resolve_device(cfg["train"]["device"])
    print(f"device {device} | torch {torch.__version__}", flush=True)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True          # fixed shapes -> +13% measured
    seed = int(cfg["train"]["seeds"][0])
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = make_loader(cfg, [files[i] for i in tr_idx], augment=True, shuffle=True)
    val_cap = int(cfg["train"].get("val_limit", 512))
    val_files = [files[i] for i in va_idx][:val_cap]
    val_loader = (make_loader(cfg, val_files, augment=False, shuffle=False)
                  if val_files else None)

    model = EllipseNet(cfg).to(device)
    print(f"model params {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M", flush=True)

    steps_per_epoch = max(len(train_loader), 1)
    epochs = int(cfg["train"]["epochs"])
    total_steps = steps_per_epoch * epochs
    criterion = TotalLoss(cfg, total_steps)
    optim = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]),
                              weight_decay=float(cfg["train"]["weight_decay"]))
    amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    tag = args.tag or f"seed{seed}"
    out_dir = os.path.join(cfg["train"]["out_dir"], tag)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    csv_file = open(os.path.join(out_dir, "log.csv"), "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["step", "epoch", "lr", "tau", "loss_total", "center", "axes", "angle",
                     "continuity", "ellipse_safe", "center_free", "border", "safety_scale"])

    best_val = float("inf")
    n_oom = 0
    t_start = time.perf_counter()
    step = 0
    for epoch in range(epochs):
        model.train()
        running: Dict[str, List[float]] = {}
        for batch in train_loader:
            lr = lr_at(step, total_steps, float(cfg["train"]["lr"]),
                       float(cfg["train"]["warmup_frac"]))
            for g in optim.param_groups:
                g["lr"] = lr
            tau = tau_at(step, total_steps, float(cfg["train"]["gumbel_tau_start"]),
                         float(cfg["train"]["gumbel_tau_end"]))
            b = move(batch, device)
            try:
                with torch.amp.autocast(device_type=device.type, enabled=amp):
                    pred = model(b["occupancy"], b["path_xy"], b["lidar"] / model.max_range_m,
                                 b["geometry"], b["r_scene"], b["mask"], lidar_m=b["lidar"],
                                 tau=tau, map_index=b.get("map_index"))
                # geometry and matrix algebra stay in FP32 (plan section 9)
                pred32 = {k: (v.float() if torch.is_tensor(v) else v)
                          for k, v in pred.items()}
                loss, logs = criterion(pred32, b, step)
                optim.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               float(cfg["train"]["grad_clip"]))
                scaler.step(optim)
                scaler.update()
            except torch.cuda.OutOfMemoryError:
                # a single oversized batch must not kill a multi-hour run
                optim.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                n_oom += 1
                print(f"  !! CUDA OOM at step {step}: batch skipped "
                      f"({n_oom} skipped so far)", flush=True)
                step += 1
                continue
            for k, v in logs.items():
                running.setdefault(k, []).append(v)
            if step % int(cfg["train"]["log_every"]) == 0:
                mean = {k: float(np.mean(v[-50:])) for k, v in running.items()}
                el = (time.perf_counter() - t_start) / 60
                print(f"[{el:6.1f}m] ep {epoch} step {step}/{total_steps} lr {lr:.2e} "
                      f"tau {tau:.2f} | total {mean['total']:.3f} c {mean['center']:.3f} "
                      f"ax {mean['axes']:.3f} ang {mean['angle']:.3f} "
                      f"safe {mean['ellipse_safe']:.4f} sdf {mean['center_free']:.4f} "
                      f"brd {mean['border']:.4f} cont {mean['continuity']:.4f}", flush=True)
                writer.writerow([step, epoch, lr, tau, mean["total"], mean["center"],
                                 mean["axes"], mean["angle"], mean["continuity"],
                                 mean["ellipse_safe"], mean["center_free"], mean["border"],
                                 mean["safety_scale"]])
                csv_file.flush()
            step += 1

        if val_loader is not None and (epoch + 1) % int(cfg["train"]["eval_every"]) == 0:
            metrics = run_evaluation(model, val_loader, device,
                                     rho_max=float(cfg["label"]["rho_max"]))
            ce = metrics.get("center_error", {})
            ex = metrics.get("exact_safety", {})
            print(f"  == val ep {epoch}: center median {ce.get('median')} "
                  f"p95 {ce.get('p95')} | iou {metrics.get('ellipse_iou', {}).get('mean')} | "
                  f"raw safety {ex.get('raw_safety_rate')} | "
                  f"cont pair {ex.get('continuity_pair_rate')} | "
                  f"border {ex.get('border_violation_rate')}", flush=True)
            with open(os.path.join(out_dir, "val_metrics.json"), "w", encoding="utf-8") as fh:
                json.dump(metrics, fh, indent=2, default=float)
            score = float(ce.get("median") or 1e9)
            if score < best_val:
                best_val = score
                torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch,
                            "metrics": metrics}, os.path.join(out_dir, "best.pt"))
        torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "step": step},
                   os.path.join(out_dir, "last.pt"))
        if args.snapshot_every and (epoch + 1) % args.snapshot_every == 0:
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch,
                        "step": step}, os.path.join(out_dir, f"epoch{epoch + 1:03d}.pt"))

    csv_file.close()
    print(f"training done in {(time.perf_counter() - t_start) / 60:.1f} min -> {out_dir}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
