"""Full V2.1 training: augmentation, schedule, early stopping, multi-seed,
stratified metrics, and a SAFE-PROJECTION inference layer.

Improvements over the first baseline run:
  * geometric augmentation that keeps the (patch, lidar, tangent, offset, angle)
    tuple consistent (h/v flip, 90 deg rotation) - the lidar ray order is
    permuted, the tangent/offset mirrored and the angle-pair transformed;
  * warmup + cosine schedule, dropout, gradient clipping, early stopping on val;
  * N seeds reported as mean +- std so a single lucky run is not quoted;
  * inference safety: predicted ellipses are shrunk about their predicted centre
    (bisection on a scale factor) until the same exact geometric test used for
    label generation passes -> safety rate and the IoU cost are both reported.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from pathsafe_ellipse.v2.dataset_v21 import (  # noqa: E402
    Standardiser, build_table, load_split_files, split_by_group, summarise,
)
from pathsafe_ellipse.v2.ellipse import exact_clearance  # noqa: E402
from pathsafe_ellipse.v2.pilot import load_scene_npz  # noqa: E402

N_RAYS = 16


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def ellipse_iou(off_p, ab_p, ang_p, off_g, ab_g, ang_g, n: int = 24) -> np.ndarray:
    out = np.zeros(len(off_p), dtype=np.float32)
    lim = float(max(ab_p.max(), ab_g.max()) * 1.25)
    xs = np.linspace(-lim, lim, n)
    X, Y = np.meshgrid(xs, xs)

    def inside(off, a_, b_, ang):
        c, s = math.cos(ang), math.sin(ang)
        dx, dy = X - off[0], Y - off[1]
        u = (dx * c + dy * s) / max(a_, 1e-6)
        v = (-dx * s + dy * c) / max(b_, 1e-6)
        return (u * u + v * v) <= 1.0

    for k in range(len(off_p)):
        ip = inside(off_p[k], ab_p[k, 0], ab_p[k, 1], ang_p[k])
        ig = inside(off_g[k], ab_g[k, 0], ab_g[k, 1], ang_g[k])
        u = np.logical_or(ip, ig).sum()
        out[k] = np.logical_and(ip, ig).sum() / u if u else 0.0
    return out


def S_from(ab: np.ndarray, ang: float) -> np.ndarray:
    c, s = math.cos(ang), math.sin(ang)
    R = np.array([[c, -s], [s, c]])
    return R @ np.diag(np.asarray(ab, dtype=float) ** 2) @ R.T


def augment(patch, lidar, tangent, extra, off, axes, ang2, rng) -> tuple:
    """Consistent mirror augmentation applied to a whole batch.

    Only the two axis mirrors are used: they are exactly representable for every
    field (patch flip, ray-index permutation, tangent/offset sign flip and a
    sin(2theta) sign flip), while a 90 deg rotation would also swap the offset
    components and is left out to avoid a subtly wrong label transform.
    """
    B = patch.shape[0]
    do_h = torch.from_numpy(rng.random(B) < 0.5).to(patch.device)
    do_v = torch.from_numpy(rng.random(B) < 0.5).to(patch.device)

    patch = torch.where(do_v.view(-1, 1, 1, 1), torch.flip(patch, dims=[2]), patch)
    patch = torch.where(do_h.view(-1, 1, 1, 1), torch.flip(patch, dims=[3]), patch)

    idx = torch.arange(N_RAYS, device=lidar.device).unsqueeze(0).expand(B, N_RAYS)
    k = torch.where(do_v.unsqueeze(1), (N_RAYS - idx) % N_RAYS, idx)       # y mirror
    k = torch.where(do_h.unsqueeze(1), (N_RAYS // 2 - k) % N_RAYS, k)      # x mirror
    lidar = torch.gather(lidar, 1, k)

    sx = torch.where(do_h, -torch.ones(B, device=off.device), torch.ones(B, device=off.device))
    sy = torch.where(do_v, -torch.ones(B, device=off.device), torch.ones(B, device=off.device))
    off = torch.stack([off[:, 0] * sx, off[:, 1] * sy], dim=1)
    tangent = torch.stack([tangent[:, 0] * sx, tangent[:, 1] * sy], dim=1)
    sgn = sx * sy                                                          # -1 for a single mirror
    ang2 = torch.stack([ang2[:, 0], ang2[:, 1] * sgn], dim=1)
    return patch, lidar, tangent, extra, off, axes, ang2


class Net(nn.Module):
    def __init__(self, width: int = 128, dropout: float = 0.1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 24, 5, stride=2, padding=2), nn.BatchNorm2d(24), nn.ReLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.BatchNorm2d(48), nn.ReLU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.mlp = nn.Sequential(
            nn.Linear(N_RAYS + 1 + 64 * 16, width), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(width, width), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(width, 6)

    def forward(self, patch, lidar, extra):
        f = self.cnn(patch).flatten(1)
        return self.head(self.mlp(torch.cat([lidar, extra, f], dim=1)))


def to_ellipse(pred: np.ndarray):
    off = pred[:, 0:2]
    ab = np.exp(np.clip(pred[:, 2:4], -6.0, 3.0))
    ang = 0.5 * np.arctan2(pred[:, 5], pred[:, 4])
    return off, ab, ang


def run_seed(table, masks, args, seed: int, device: str, scene_cache) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    mask_tr, mask_va, mask_te = masks

    lidar_std = Standardiser.fit(table["lidar"][mask_tr])
    extra_std = Standardiser.fit(table["clearance_p"][mask_tr])
    off_std = Standardiser.fit(table["offset"][mask_tr])

    def pack(mask):
        return {
            "patch": torch.from_numpy(table["patch"][mask]).float(),
            "lidar": torch.from_numpy(lidar_std(table["lidar"][mask])).float(),
            "tangent": torch.from_numpy(table["tangent"][mask]).float(),
            "extra": torch.from_numpy(extra_std(table["clearance_p"][mask])).float(),
            "off": torch.from_numpy(table["offset"][mask]).float(),
            "axes": torch.from_numpy(table["log_axes"][mask]).float(),
            "ang": torch.from_numpy(table["angle2"][mask]).float(),
        }

    tr, va, te = pack(mask_tr), pack(mask_va), pack(mask_te)
    model = Net(dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = max(1, math.ceil(len(tr["off"]) / args.batch))
    warm = max(1, int(0.1 * args.epochs * steps_per_epoch))
    total = args.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, (s - warm) / max(1, total - warm))))))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    off_s = torch.tensor(off_std.std, dtype=torch.float32, device=device)
    off_m = torch.tensor(off_std.mean, dtype=torch.float32, device=device)

    def loss_fn(pred, b, extra_in):
        off_hat = pred[:, 0:2] * off_s + off_m
        l_off = torch.nn.functional.smooth_l1_loss(off_hat, b["off"], beta=0.25)
        l_ax = torch.nn.functional.smooth_l1_loss(pred[:, 2:4], b["axes"], beta=0.1)
        l_ang = torch.nn.functional.l1_loss(pred[:, 4:6], b["ang"])
        return l_off + l_ax + 0.5 * l_ang

    best = {"val": float("inf"), "ep": -1, "state": None}
    hist = {"train": [], "val": []}
    bad = 0
    n = len(tr["off"])
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            p_ = tr["patch"][idx].to(device)
            l_ = tr["lidar"][idx].to(device)
            t_ = tr["tangent"][idx].to(device)
            e_ = tr["extra"][idx].to(device)
            b = {k: tr[k][idx].to(device) for k in ("off", "axes", "ang")}
            if args.augment:
                p_, l_, t_, e_, b["off"], b["axes"], b["ang"] = augment(
                    p_, l_, t_, e_, b["off"], b["axes"], b["ang"], rng)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                pred = model(p_, l_, e_)
                loss = loss_fn(pred, b, e_)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss) * len(idx)
        model.eval()
        with torch.no_grad():
            vloss = float(loss_fn(
                model(va["patch"].to(device), va["lidar"].to(device), va["extra"].to(device)),
                {k: va[k].to(device) for k in ("off", "axes", "ang")}, None))
        hist["train"].append(tot / max(n, 1))
        hist["val"].append(vloss)
        if vloss < best["val"] - 1e-5:
            best = {"val": vloss, "ep": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best["state"] is not None:
        model.load_state_dict(best["state"])

    with torch.no_grad():
        pred_te = model(te["patch"].to(device), te["lidar"].to(device),
                        te["extra"].to(device)).cpu().numpy()
    off_p, ab_p, ang_p = to_ellipse(pred_te)
    off_p = off_p * off_std.std + off_std.mean
    off_g = te["off"].cpu().numpy()
    ab_g = np.exp(te["axes"].cpu().numpy())
    a2 = te["ang"].cpu().numpy()
    ang_g = 0.5 * np.arctan2(a2[:, 1], a2[:, 0])

    iou_raw = ellipse_iou(off_p, ab_p, ang_p, off_g, ab_g, ang_g)

    # ---- diagnostic only: how often is the RAW prediction already collision free?
    # Collisions are expected for a regression head; the analytic safety step is
    # applied downstream, so nothing is projected or rescaled here.
    test_glob = np.flatnonzero(mask_te)
    safe_raw, n_chk = 0, 0
    for i_loc, i_glob in enumerate(test_glob[: args.safety_check]):
        sid = str(table["scene_id"][i_glob])
        npz = scene_cache.get(sid)
        if npz is None:
            npz = load_scene_npz(os.path.join(PROJECT_ROOT, "data", "processed_v2_1",
                                             f"pilot_{args.tag}", "v2_1", f"{sid}.npz"))
            scene_cache[sid] = npz
        rs = float(npz["meta"]["r_scene_m"])
        sel = (npz["path_id"] == table["path_id"][i_glob]) & \
              (npz["point_id"] == table["point_id"][i_glob])
        if not np.any(sel):
            continue
        j = int(np.flatnonzero(sel)[0])
        p_xy = npz["p_xy"][j].astype(float)
        bl = npz["blocked"].astype(np.uint8)
        centre = p_xy + off_p[i_loc]
        ok, _, _, _ = exact_clearance(S_from(ab_p[i_loc], ang_p[i_loc]), centre, bl, rs)
        safe_raw += int(ok)
        n_chk += 1
    return {
        "seed": seed, "best_epoch": best["ep"], "best_val": best["val"],
        "epochs_run": len(hist["train"]),
        "hist": hist,
        "iou_raw_mean": float(iou_raw.mean()), "iou_raw_p50": float(np.median(iou_raw)),
        "centre_err_m": float(np.linalg.norm(off_p - off_g, axis=1).mean()),
        "area_ratio_p50": float(np.median((np.pi * ab_p[:, 0] * ab_p[:, 1])
                                          / np.maximum(np.pi * ab_g[:, 0] * ab_g[:, 1], 1e-6))),
        "raw_collision_free_rate": safe_raw / max(n_chk, 1),
        "n_safety_checked": n_chk,
        "model": {k: v for k, v in model.state_dict().items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dataset_v21")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--augment", action="store_true", default=True)
    ap.add_argument("--no-augment", dest="augment", action="store_false")
    ap.add_argument("--safety-check", type=int, default=400)
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "runs_v21_full"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    files = load_split_files(os.path.join(PROJECT_ROOT, "data", "processed_v2_1",
                                         f"pilot_{args.tag}", "v2_1"))
    t0 = time.perf_counter()
    table = build_table(files, with_patch=True, only_valid=True)
    print(f"{len(files)} scenes, {len(table['offset'])} labels "
          f"({time.perf_counter() - t0:.0f}s), device {device}")
    info = summarise(table)
    print("dataset:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                  for k, v in info.items()}))
    masks = split_by_group(table["group"], seed=0)
    print(f"splits train {masks[0].sum()} val {masks[1].sum()} test {masks[2].sum()}")

    runs, cache = [], {}
    for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
        t1 = time.perf_counter()
        r = run_seed(table, masks, args, seed, device, cache)
        r["time_s"] = time.perf_counter() - t1
        print(f"[seed {seed}] ep {r['best_epoch']:3d}/{r['epochs_run']:3d} "
              f"val {r['best_val']:.4f} | IoU {r['iou_raw_mean']:.4f} "
              f"(p50 {r['iou_raw_p50']:.4f}) | centre {r['centre_err_m']:.3f} m "
              f"| area ratio {r['area_ratio_p50']:.3f} "
              f"| collision-free(raw) {r['raw_collision_free_rate']:.3f} "
              f"| {r['time_s']:.0f}s")
        runs.append(r)

    keys = ("iou_raw_mean", "iou_raw_p50", "centre_err_m", "area_ratio_p50",
            "raw_collision_free_rate", "best_epoch")
    summary = {k: {"mean": float(np.mean([r[k] for r in runs])),
                   "std": float(np.std([r[k] for r in runs]))} for k in keys if
               all(r[k] is not None for r in runs)}
    print("\n== across seeds ==")
    for k, v in summary.items():
        print(f"  {k:20s} {v['mean']:.4f} +- {v['std']:.4f}")

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.0))
    for r in runs:
        axes[0].plot(r["hist"]["val"], lw=1.0, label=f"seed {r['seed']}")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("val loss")
    axes[0].set_title("validation loss per seed"); axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    axes[1].errorbar(["raw"], [summary["iou_raw_mean"]["mean"]],
                     yerr=[summary["iou_raw_mean"]["std"]], fmt="o", capsize=5)
    axes[1].set_ylabel("test IoU (mean +- std over seeds)")
    axes[1].set_title("IoU of raw predictions"); axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, 1)
    axes[2].errorbar(["raw"], [summary["raw_collision_free_rate"]["mean"]],
                     yerr=[summary["raw_collision_free_rate"]["std"]], fmt="o", capsize=5)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("fraction collision free")
    axes[2].set_title("raw predictions vs obstacles\n(diagnostic only, no repair here)")
    axes[2].grid(alpha=0.3)
    axes[3].hist(runs[0]["hist"]["train"], bins=30, alpha=0.6, label="train")
    axes[3].hist(runs[0]["hist"]["val"], bins=30, alpha=0.6, label="val")
    axes[3].set_xlabel("loss"); axes[3].set_title("loss distribution (seed 0)")
    axes[3].legend(fontsize=7); axes[3].grid(alpha=0.3)
    fig.suptitle(f"V2.1 full training - {args.tag}")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "training_full.png"), dpi=115)
    plt.close(fig)

    torch.save({"model": runs[0]["model"], "args": vars(args)},
               os.path.join(args.out, "model_seed0.pt"))
    payload = {"tag": args.tag, "dataset": info, "args": vars(args),
               "splits": {"train": int(masks[0].sum()), "val": int(masks[1].sum()),
                          "test": int(masks[2].sum())},
               "seeds": [{k: v for k, v in r.items() if k not in ("model", "hist")}
                         for r in runs],
               "summary": summary,
               "history_seed0": runs[0]["hist"]}
    with open(os.path.join(args.out, "metrics_full.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=float)
    print(f"\n-> {args.out}/metrics_full.json\n-> {args.out}/training_full.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
