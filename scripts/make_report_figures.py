"""Report figures/tables from a finished ``ellipsenet`` run (no training, read-only).

    python scripts/make_report_figures.py --run-dir checkpoints/full_s0 \
        --out-dir results/figures/curves

Writes ``loss_curves.png`` (raw + normalised loss components), ``metrics_table.md``
(val/test numbers of plan section 14) and ``log_summary.json``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


def read_log(run_dir: str, log_name: str = "log.csv"):
    path = os.path.join(run_dir, log_name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    data = {k: [] for k in rows[0]}
    for r in rows:
        for k, v in r.items():
            try:
                data[k].append(float(v))
            except (TypeError, ValueError):
                data[k].append(np.nan)
    return {k: np.asarray(v) for k, v in data.items()}


def plot_curves(log, out_png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    s = log["step"]
    ax = axes[0]
    for k in ("total", "choice", "conn"):
        ax.plot(s, log[k], label=k, lw=1.2)
    ax.set_xlabel("step"); ax.set_ylabel("loss (normalised)")
    ax.set_title("total / choice / connection")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    for k in ("raw_dist", "raw_safe", "raw_area", "raw_conn"):
        label = {"raw_dist": "clearance S(q) [m]", "raw_safe": "safety penalty (weighted)",
                 "raw_area": "-log(area)", "raw_conn": "connection (raw)"}[k]
        ax.plot(s, log[k], label=label, lw=1.2)
    ax.set_xlabel("step"); ax.set_ylabel("raw magnitude")
    ax.set_title("raw loss magnitudes (plan 10.7)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(s, log["pi_entropy"], label="direction entropy", lw=1.2)
    ax.plot(s, log["pi_max"], label="max probability", lw=1.2)
    ax.set_xlabel("step")
    ax.set_title("direction distribution (max entropy = %.2f)" % np.log(16.0))
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def metrics_table(metrics: dict, title: str) -> str:
    cc = metrics.get("selected_center_clearance", {})
    ar = metrics.get("ellipse_area_m2", {})
    ent = metrics.get("direction_entropy", {})
    rows = [
        ("argmax 选中中心净空 S(q) [m] mean/p10/p50/p95",
         f"{cc.get('sdf_mean'):.3f} / {cc.get('sdf_p10'):.3f} / {cc.get('sdf_p50'):.3f} / "
         f"{cc.get('sdf_p95'):.3f}"),
        ("椭圆面积 [m²] mean/p10/p50/p95",
         f"{ar.get('mean'):.3f} / {ar.get('p10'):.3f} / {ar.get('p50'):.3f} / "
         f"{ar.get('p95'):.3f}"),
        ("SDF 采样安全率", f"{metrics.get('sdf_safety_rate'):.4f}"),
        ("**blocked 栅格精确碰撞率**", f"{metrics.get('exact_collision_rate'):.5f}"),
        ("中心落在 blocked 格比例", f"{metrics.get('exact_center_collision_rate'):.5f}"),
        ("相邻椭圆连续覆盖率（边界互相包含）", f"{metrics.get('boundary_cover_rate'):.4f}"),
        ("相邻中点连续性通过率", f"{metrics.get('continuity_pair_rate'):.4f}"),
        ("整条路径走廊成功率", f"{metrics.get('corridor_success_rate'):.4f}"),
        ("越出地图边界比例", f"{metrics.get('border_violation_rate'):.5f}"),
        ("Direction 熵 mean / max 概率 mean",
         f"{ent.get('mean'):.3f} / {metrics.get('direction_p_max', {}).get('mean'):.3f}"),
        ("argmax 与最小代价候选一致率", f"{metrics.get('argmax_vs_min_loss_agreement'):.4f}"),
        ("GT 中心误差 [m] p50 / p95",
         f"{metrics.get('gt_center_error_m', {}).get('p50'):.3f} / "
         f"{metrics.get('gt_center_error_m', {}).get('p95'):.3f}"),
        ("GT 椭圆 IoU mean / p50",
         f"{metrics.get('gt_ellipse_iou', {}).get('mean'):.3f} / "
         f"{metrics.get('gt_ellipse_iou', {}).get('p50'):.3f}"),
        ("GT 面积比 p50", f"{metrics.get('gt_area_ratio', {}).get('p50'):.3f}"),
        ("轨迹 / 查询点", f"{metrics.get('n_trajectories')} / {metrics.get('n_points')}"),
    ]
    lines = [f"### {title}", "", "| 指标 | 数值 |", "|---|---|"]
    lines += [f"| {a} | {b} |" for a, b in rows]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="checkpoints")
    ap.add_argument("--metrics-dir", default="results/metrics")
    ap.add_argument("--out-dir", default="results/figures/curves")
    ap.add_argument("--log-name", default="v3_train_log.csv",
                    help="training log csv inside the run dir")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    log = read_log(args.run_dir, args.log_name)
    if log is not None:
        plot_curves(log, os.path.join(args.out_dir, "loss_curves.png"))
        summary = {k: {"first": float(log[k][0]), "last": float(log[k][-1])}
                   for k in log if k not in ("step", "epoch")}
        with open(os.path.join(args.out_dir, "log_summary.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print("loss curves ->", os.path.join(args.out_dir, "loss_curves.png"))

    blocks = []
    for name, title in (("eval_val.json", "验证集（全量）"),
                        ("val_monitor.json", "验证集（训练期监控子集）"),
                        ("eval_test.json", "测试集（map_group 完全 held-out）")):
        path = os.path.join(args.metrics_dir, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
            metrics = payload.get("metrics", payload)
            blocks.append(metrics_table(metrics, title))
    if blocks:
        out = os.path.join(args.out_dir, "metrics_table.md")
        with open(out, "w", encoding="utf-8") as fh:
            fh.write("\n\n".join(blocks) + "\n")
        print("\n\n".join(blocks))
        print("-> ", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
