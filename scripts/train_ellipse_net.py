"""Train the path-conditioned ellipse network on the generated dataset."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pathsafe_ellipse.config import load_config, project_paths, setup_logging  # noqa: E402
from pathsafe_ellipse.evaluation import EllipseCriterion, evaluate_model  # noqa: E402
from pathsafe_ellipse.model import EllipseNet  # noqa: E402
from pathsafe_ellipse.torch_dataset import SplitData  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="")
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--val-samples", type=int, default=3000)
    p.add_argument("--val-safety-samples", type=int, default=256)
    p.add_argument("--num-threads", type=int, default=8)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg, "train")
    pp = project_paths(cfg)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(int(args.num_threads))
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    res = float(cfg["map"]["resolution"])

    train_path = os.path.join(pp.splits_dir, "train.npz")
    val_path = os.path.join(pp.splits_dir, "val.npz")
    for path in (train_path, val_path):
        if not os.path.exists(path):
            print(f"ERROR: missing {path}; run dataset generation first")
            return 2
    train = SplitData(train_path)
    val = SplitData(val_path)
    if args.max_train_samples:
        train.n = min(train.n, args.max_train_samples)
        print(f"limiting training samples to {train.n}")
    print(f"train samples={train.n}  val samples={val.n}  device={device}")

    model = EllipseNet(width=args.width).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params/1e6:.2f}M")
    criterion = EllipseCriterion()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, train.n // args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
        pct_start=0.15, div_factor=10.0, final_div_factor=100.0)

    ckpt_dir = os.path.join(pp.outputs_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    history = {"epochs": [], "train_loss": [], "val": [], "lr": []}
    best = float("inf")
    best_state = None
    rng = np.random.default_rng(args.seed)
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_seen = 0
        comp_sums: Dict[str, float] = {}
        t0 = time.time()
        for idx in train.epoch(args.batch_size, rng):
            batch = train.batch(idx)
            tensors = {k: torch.from_numpy(batch[k]).to(device, non_blocking=True)
                       for k in ("patch", "rays", "hits", "tangent")}
            targets = {k: torch.from_numpy(batch[k]).to(device, non_blocking=True)
                       for k in ("center_offset", "ab", "sincos", "angle_valid")}
            preds = model(tensors["patch"], tensors["rays"], tensors["hits"], tensors["tangent"])
            loss, comps = criterion(preds, targets)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            scheduler.step()
            epoch_loss += float(loss.detach()) * len(idx)
            n_seen += len(idx)
            for k, v in comps.items():
                comp_sums[k] = comp_sums.get(k, 0.0) + v * len(idx)

        train_loss = epoch_loss / max(n_seen, 1)
        val_metrics = evaluate_model(
            model, val, device, res, batch_size=args.batch_size,
            max_samples=args.val_samples, eta=float(cfg["label"]["point_inside_eta"]),
            safety_check=True, max_safety_samples=args.val_safety_samples,
            criterion=criterion)

        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val"].append({k: float(v) for k, v in val_metrics.items()})
        history["lr"].append(float(optim.param_groups[0]["lr"]))

        print(f"epoch {epoch:3d}/{args.epochs} train_loss={train_loss:.4f} "
              f"({', '.join(f'{k}={v/max(n_seen,1):.4f}' for k, v in comp_sums.items())}) "
              f"| val loss={val_metrics['loss']:.4f} center={val_metrics['center_err_m']:.3f}m "
              f"axes=({val_metrics['a_err_m']:.3f},{val_metrics['b_err_m']:.3f})m "
              f"angle={val_metrics['angle_err_deg']:.1f}deg IoU={val_metrics['iou']:.3f} "
              f"rho_ok={val_metrics['rho_ok_rate']:.3f} safe={val_metrics['safety_rate']:.3f} "
              f"safe+inside={val_metrics['safe_and_contains_rate']:.3f} "
              f"| {time.time()-t0:.1f}s", flush=True)

        score = val_metrics["loss"]
        if score < best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"state_dict": best_state, "args": vars(args),
                        "epoch": epoch, "val": val_metrics, "width": args.width},
                       os.path.join(ckpt_dir, f"ellipse_net_best{args.tag}.pt"))

    if best_state is not None:
        torch.save({"state_dict": best_state, "args": vars(args),
                    "width": args.width, "best_val_loss": best},
                   os.path.join(ckpt_dir, f"ellipse_net_last{args.tag}.pt"))
    elapsed = time.time() - t_start
    history["best_val_loss"] = best
    history["elapsed_s"] = round(elapsed, 1)
    history["n_parameters"] = n_params
    history["n_train"] = train.n
    history["n_val"] = val.n
    with open(os.path.join(pp.reports_dir, f"training_history{args.tag}.json"), "w",
              encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)

    # training curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
        axes[0].plot(history["epochs"], history["train_loss"], label="train")
        axes[0].plot(history["epochs"], [v["loss"] for v in history["val"]], label="val")
        axes[0].set_title("loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
        axes[1].plot(history["epochs"], [v["center_err_m"] for v in history["val"]], label="centre (m)")
        axes[1].plot(history["epochs"], [v["a_err_m"] for v in history["val"]], label="semi-axis a (m)")
        axes[1].plot(history["epochs"], [v["b_err_m"] for v in history["val"]], label="semi-axis b (m)")
        axes[1].set_title("regression error"); axes[1].set_xlabel("epoch"); axes[1].legend()
        axes[2].plot(history["epochs"], [v["iou"] for v in history["val"]], label="IoU")
        axes[2].plot(history["epochs"], [v["safe_and_contains_rate"] for v in history["val"]],
                     label="safe & contains point")
        axes[2].plot(history["epochs"], [v["rho_ok_rate"] for v in history["val"]],
                     label="rho <= eta")
        axes[2].set_title("geometric metrics"); axes[2].set_xlabel("epoch"); axes[2].legend()
        for ax in axes:
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(pp.reports_dir, f"training_curves{args.tag}.png"), dpi=120)
        plt.close(fig)
    except Exception as exc:  # plotting is best-effort
        print(f"curve plot skipped: {exc}")

    print(f"\nbest val loss {best:.4f}  elapsed {elapsed/60:.1f} min")
    print(f"checkpoints -> {ckpt_dir}")
    print(f"history    -> {os.path.join(pp.reports_dir, f'training_history{args.tag}.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
