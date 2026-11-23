"""Single-trajectory inference latency (batch = 1, no batching tricks).

    python scripts/bench_inference.py --checkpoint checkpoints/v3_best.pt --iters 50

Reports, for one complete A* trajectory (map + N path points + 16 LiDAR rays):

* end-to-end  : read the trajectory NPZ + the SDF map, forward, argmax decode
* model only  : the same batch reused, so no I/O and no map decode
* breakdown   : the two ResUNets versus the rest (queries, attention, heads)
* scaling     : model-only latency at N = 10 / 30 / 70 path points
* CPU         : the same measurement on CPU for reference
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from ellipsenet.data.dataset import (TrajectoryDataset, collate, load_index,  # noqa: E402
                                  split_by_group)
from ellipsenet.models.psenet import PSENet, count_parameters  # noqa: E402


def sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def timeit(fn, iters: int, device, warmup: int = 10) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    sync(device)
    ts: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync(device)
        ts.append((time.perf_counter() - t0) * 1e3)
    a = np.asarray(ts)
    return {"mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "min_ms": float(a.min()),
            "iters": iters}


def to_device(batch: Dict, device) -> Dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/v3_best.pt")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--amp", action="store_true", help="also time float16 autocast")
    ap.add_argument("--out", default="results/metrics/inference_latency.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    model = PSENet(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    print(f"checkpoint {args.checkpoint} | device {device} | "
          f"{count_parameters(model) / 1e6:.2f} M params", flush=True)

    entries = load_index(cfg["data"]["sdf_dir"])
    tr, va, te = split_by_group(entries, frac=tuple(cfg["data"]["split_frac"]),
                                seed=int(cfg["data"]["split_seed"]))
    pick = {"train": tr, "val": va, "test": te}[args.split]
    pick = [i for i in pick if int(entries[i]["n_points"]) <= int(cfg["data"]["max_points"])]
    # a trajectory with a typical number of path points (dataset mean is ~29)
    typical = [i for i in pick if 24 <= int(entries[i]["n_points"]) <= 40][:1]
    entry = entries[typical[0]]
    print(f"trajectory {entry['trajectory']} | N={entry['n_points']} | "
          f"res {entry['resolution_m']} m", flush=True)

    ds = TrajectoryDataset([entry], cfg["data"]["trajectory_dir"], cfg["data"]["sdf_dir"],
                           augment=False)
    item = ds[0]
    batch = to_device(collate([item]), device)

    with torch.no_grad():
        # ---- end-to-end: fresh decode of the NPZ files + forward + argmax
        def end_to_end():
            it = ds[0]                       # reads both NPZ files from disk
            b = to_device(collate([it]), device)
            out = model(b, tau=1.0)
            model.decode_argmax(out)

        def model_only():
            out = model(batch, tau=1.0)
            model.decode_argmax(out)

        def unets_only():
            model.occ_unet(batch["occupancy"])
            model.dist_unet(torch.clamp(batch["sdf"] / model.d_clip, -1.0, 1.0))

        def forward_only():
            model(batch, tau=1.0)

        results = {
            "device": str(device),
            "checkpoint": os.path.abspath(args.checkpoint),
            "trajectory": entry["trajectory"],
            "n_points": int(entry["n_points"]),
            "resolution_m": float(entry["resolution_m"]),
            "params_m": count_parameters(model) / 1e6,
            "end_to_end": timeit(end_to_end, args.iters, device),
            "model_only": timeit(model_only, args.iters, device),
            "forward_only": timeit(forward_only, args.iters, device),
            "unets_only": timeit(unets_only, args.iters, device),
        }

        if args.amp and device.type == "cuda":
            def model_amp():
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out = model(batch, tau=1.0)
                model.decode_argmax({k: (v.float() if torch.is_tensor(v) else v)
                                     for k, v in out.items()})
            results["model_only_amp"] = timeit(model_amp, args.iters, device)

        # ---- scaling with the number of path points (model only, no I/O)
        scale = {}
        n_avail = int(entry["n_points"])
        for n_req in (10, 30, 70):
            if n_req > n_avail:
                continue
            b = dict(batch)
            b["path_xy"] = batch["path_xy"][:, :n_req].contiguous()
            b["lidar"] = batch["lidar"][:, :n_req].contiguous()
            b["ray_valid"] = batch["ray_valid"][:, :n_req].contiguous()
            m = torch.ones((batch["path_xy"].shape[0], n_req), dtype=torch.bool,
                           device=batch["path_xy"].device)
            b["mask"] = m
            b["path_mask"] = m
            scale[n_req] = timeit(lambda b=b: model.decode_argmax(model(b, tau=1.0)),
                                  30, device, warmup=5)
        results["scaling_model_only"] = scale

    # ---- CPU reference (fewer iterations; it is slow on purpose)
    cpu = torch.device("cpu")
    model_cpu = PSENet(cfg).to(cpu).eval()
    model_cpu.load_state_dict(ck["model"])
    b_cpu = to_device(collate([item]), cpu)
    with torch.no_grad():
        results["cpu_model_only"] = timeit(
            lambda: model_cpu.decode_argmax(model_cpu(b_cpu, tau=1.0)), 5, cpu, warmup=1)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
