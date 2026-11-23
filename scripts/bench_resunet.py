"""Where does the 256x256 ResUNet time actually go? (read-only profiling)

    python scripts/bench_resunet.py

Prints, for one ResUNet of the delivered configuration (in_ch 1 -> 128, base 24):
* analytic MACs per stage,
* measured latency per stage (CUDA events),
* the same network at base 16 / 24 / 32,
* `channels_last` and `torch.compile` variants of the delivered weights.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from ellipsenet.models.resunet import ResUNet  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bench(fn, iters: int = 50, warmup: int = 10) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    a = np.asarray(ts)
    return {"mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50))}


def macs_conv(cin: int, cout: int, hw: int, k: int = 3) -> float:
    return cin * cout * k * k * hw * hw


def stage_macs(base: int, out_ch: int = 128, levels=3) -> Dict[str, float]:
    c = [base * 2 ** i for i in range(levels + 1)]
    out = {
        "stem 3x3 1->base @256": macs_conv(1, base, 256) + macs_conv(base, base, 256),
        "enc1 (48ch) @128": 2 * macs_conv(c[1], c[1], 128) + macs_conv(c[0], c[1], 128),
        "enc2 (96ch) @64": 2 * macs_conv(c[2], c[2], 64) + macs_conv(c[1], c[2], 64),
        "enc3 (192ch) @32": 2 * macs_conv(c[3], c[3], 32) + macs_conv(c[2], c[3], 32),
        "up1 -> 24ch @256": macs_conv(c[0] + c[1], c[0], 256) + macs_conv(c[0], c[0], 256),
        "up2 -> 48ch @128": macs_conv(c[1] + c[2], c[1], 128) + macs_conv(c[1], c[1], 128),
        "up3 -> 96ch @64": macs_conv(c[2] + c[3], c[2], 64) + macs_conv(c[2], c[2], 64),
        "head 1x1 base->out": base * out_ch * 256 * 256,
    }
    return out


def main() -> int:
    print(f"device {DEV}")
    x = torch.randn(1, 1, 256, 256, device=DEV)

    # ---- analytic MACs
    m = stage_macs(24)
    total = sum(m.values())
    print("\n== analytic MACs per forward, one ResUNet (base 24 -> 128 out)")
    for k, v in m.items():
        print(f"  {k:22s} {v / 1e9:7.3f} G   {100 * v / total:5.1f}%")
    print(f"  {'total':22s} {total / 1e9:7.3f} G")
    print(f"  (two branches = {2 * total / 1e9:.2f} G MAC ~= "
          f"{4 * total / 1e9:.2f} GFLOP per sample, batch 1)")

    # ---- measured per-stage latency
    net = ResUNet(1, 128, base=24, depths=[1, 1, 1, 1]).to(DEV).eval()
    with torch.no_grad():
        s0 = net.stem(x)
        s1 = net.enc1(net.down1(s0))
        s2 = net.enc2(net.down2(s1))
        s3 = net.enc3(net.down3(s2))
        u3 = net.up3(s3, s2)
        u2 = net.up2(u3, s1)
        u1 = net.up1(u2, s0)
        stages = {"stem @256": (net.stem, (x,)),
                  "down1+enc1": (net.enc1, (net.down1(s0),)),
                  "down2+enc2": (net.enc2, (net.down2(s1),)),
                  "down3+enc3": (net.enc3, (net.down3(s2),)),
                  "up3 @64": (net.up3, (s3, s2)),
                  "up2 @128": (net.up2, (u3, s1)),
                  "up1 @256": (net.up1, (u2, s0)),
                  "head 1x1": (net.head, (u1,))}
        print("\n== measured latency per stage (batch 1)")
        tot_ms = 0.0
        for name, (fn, args) in stages.items():
            r = bench(lambda fn=fn, args=args: fn(*args))
            tot_ms += r["mean_ms"]
            print(f"  {name:14s} {r['mean_ms']:6.2f} ms")
        print(f"  {'sum':14s} {tot_ms:6.2f} ms")
        full = bench(lambda: net(x))
        print(f"  full forward   {full['mean_ms']:6.2f} ms (p50 {full['p50_ms']:.2f})")

    # ---- width variants (needs retraining to be usable, shown for scale)
    print("\n== width variants, one ResUNet, batch 1")
    for base in (16, 24, 32):
        n2 = ResUNet(1, 128, base=base, depths=[1, 1, 1, 1]).to(DEV).eval()
        with torch.no_grad():
            r = bench(lambda n2=n2: n2(x), iters=30, warmup=5)
        p = sum(q.numel() for q in n2.parameters()) / 1e6
        print(f"  base {base:2d}: {r['mean_ms']:6.2f} ms | params {p:.2f} M")

    # ---- inference-only tricks with the delivered weights
    print("\n== inference tricks (same weights, batch 1)")
    with torch.no_grad():
        r = bench(lambda: net(x))
        print(f"  float32 / NCHW          {r['mean_ms']:6.2f} ms")
        ncl = net.to(memory_format=torch.channels_last)
        xn = x.to(memory_format=torch.channels_last)
        r = bench(lambda: ncl(xn))
        print(f"  float32 / channels_last {r['mean_ms']:6.2f} ms")
        r = bench(lambda: ncl.half()(xn.half()))
        print(f"  float16 / channels_last {r['mean_ms']:6.2f} ms")
        net = net.float()
    try:
        import torch._dynamo as dynamo
        dynamo.config.suppress_errors = True
        cm = torch.compile(net, mode="reduce-overhead")
        with torch.no_grad():
            r = bench(lambda: cm(x), iters=20, warmup=5)
        print(f"  torch.compile           {r['mean_ms']:6.2f} ms")
    except Exception as exc:                     # pragma: no cover
        print(f"  torch.compile           failed: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
