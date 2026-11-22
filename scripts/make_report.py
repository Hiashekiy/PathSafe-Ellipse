"""Assemble the full statistics report (section 11.4).

Aggregates from the **per-map artefacts** (`data/intermediate/paths/*.npz`,
`data/intermediate/samples/*.npz`) which are the authoritative per-map outputs;
per-shard JSON logs are merged when present for the rejection-reason breakdown.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pathsafe_ellipse.config import load_config, project_paths, setup_logging  # noqa: E402
from pathsafe_ellipse.map_io import map_group_id  # noqa: E402


def pct(num: float, den: float) -> str:
    return f"{num}/{den} ({100.0 * num / den:.2f}%)" if den else "0/0 (n/a)"


def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def merge_shard_logs(reports_dir: str, prefix: str) -> List[Dict]:
    """Merge every per-shard log of a stage (they are per-map record lists)."""
    records: List[Dict] = []
    for path in sorted(glob.glob(os.path.join(reports_dir, f"{prefix}*.json"))):
        data = load_json(path, [])
        if isinstance(data, list):
            records.extend([r for r in data if isinstance(r, dict) and "error" not in r])
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg, "make_report")
    pp = project_paths(cfg)

    inventory = load_json(os.path.join(pp.reports_dir, "map_inventory.json"), {})
    summary = load_json(os.path.join(pp.reports_dir, "generation_summary.json"), {})
    validation = load_json(os.path.join(pp.reports_dir, "validation_summary.json"), {})
    shard_samples = merge_shard_logs(pp.reports_dir, "sample_generation")

    path_files = sorted(glob.glob(os.path.join(pp.paths_dir, "*.npz")))
    sample_files = sorted(glob.glob(os.path.join(pp.samples_dir, "*.npz")))

    md: List[str] = []
    md.append("# PathSafe-Ellipse :: 数据生成统计报告\n")
    md.append(f"- 配置文件：`{os.path.basename(cfg['_config_path'])}`")
    md.append(f"- 随机种子：{cfg['project']['random_seed']}")
    md.append(f"- 分辨率：{cfg['map']['resolution']} m/cell；机器人半径：{cfg['robot']['radius']} m；"
              f"膨胀：3x3 方形结构元素 × {cfg['robot']['inflation_iterations']} 次")
    md.append(f"- 每图目标路径数：{cfg['path_search']['paths_per_map']}；"
              f"弧长采样间距：{cfg['path_sampling']['spacing']} cells；"
              f"射线数：{cfg['local_features']['ray_count']}；最大量程："
              f"{cfg['local_features']['ray_max_range']} cells\n")

    # ------------------------------------------------------------ maps/paths
    md.append("## 1. 地图与路径\n")
    if inventory:
        md.append(f"- 递归发现 `.map` 文件：**{inventory.get('n_files')}** 个")
        md.append(f"- 成功解析：{pct(inventory.get('n_parsed_ok', 0), inventory.get('n_files', 0))}")
        md.append(f"- 文件头尺寸集合：{inventory.get('header_sizes')}"
                  "（文件名中的 256/512/1024 不代表实际尺寸）")
        md.append(f"- 分辨率字段：{inventory.get('resolutions')}")
        md.append(f"- 地图组数：{inventory.get('n_groups')}")
        md.append(f"- 障碍占比：mean={inventory.get('obstacle_ratio_mean'):.3f} "
                  f"min={inventory.get('obstacle_ratio_min'):.3f} "
                  f"max={inventory.get('obstacle_ratio_max'):.3f}")
        md.append(f"- 原始目录自带的 train/val 标签计数：{inventory.get('provided_split_counts')}"
                  "（仅作参考，本项目按地图组重新划分）\n")

    n_paths = 0
    arc, tort, clr_min = [], [], []
    for path in path_files:
        with np.load(path) as d:
            n_paths += len(d["path_offsets"]) - 1
            arc.append(d["arc_length"]); tort.append(d["tortuosity"]); clr_min.append(d["clearance_min"])
    if arc:
        arc = np.concatenate(arc); tort = np.concatenate(tort); clr_min = np.concatenate(clr_min)
        md.append("### A* 路径生成\n")
        md.append(f"- 生成有效路径：**{n_paths}** 条，覆盖 {len(path_files)} 张地图")
        md.append(f"- 路径弧长 (cells)：mean={arc.mean():.1f} p50={np.median(arc):.1f} "
                  f"p95={np.percentile(arc, 95):.1f} max={arc.max():.0f} "
                  f"（{cfg['map']['resolution']} m/cell → mean {arc.mean()*cfg['map']['resolution']:.1f} m）")
        md.append(f"- 曲折度（弧长/直线距离）：mean={tort.mean():.3f} p95={np.percentile(tort, 95):.3f} "
                  f"max={tort.max():.3f}")
        md.append(f"- 路径上最小净空 (cells)：mean={clr_min.mean():.2f} p5={np.percentile(clr_min, 5):.2f} "
                  f"min={clr_min.min():.2f}")
        md.append("- 每条路径的离散栅格都在 `occupancy_inflated` 自由区，8 邻域步长合法且禁止对角穿角"
                  "（见 `scripts/test_geometry.py` 的 A* 检查）\n")

    # --------------------------------------------------------------- labels
    if sample_files:
        n_samples = 0
        cand_total = 0
        cand_valid = 0
        iters = []
        per_map_samples = []
        for path in sample_files:
            with np.load(path) as d:
                n = len(d["label_valid"])
                n_samples += n
                per_map_samples.append(n)
                cand_total += int(d["n_candidates"].sum())
                cand_valid += int(d["n_candidates_valid"].sum())
                iters.append(d["ellipse_iters"])
        iters = np.concatenate(iters)
        md.append("## 2. 候选中心与椭圆标签（全部地图）\n")
        md.append(f"- 有效样本总数：**{n_samples}**（{len(sample_files)} 张地图，"
                  f"每图 mean={np.mean(per_map_samples):.0f} / min={np.min(per_map_samples)} / "
                  f"max={np.max(per_map_samples)}）")
        md.append(f"- 每个样本的候选中心数：mean={cand_total/max(n_samples,1):.2f}（射线 16 × 比例 3 + 不偏置）")
        md.append(f"- 每个样本通过验收的候选数：mean={cand_valid/max(n_samples,1):.2f}"
                  "（上界剪枝后只求解必要的候选，因此该值接近 1）")
        md.append(f"- IRIS 迭代次数：mean={iters.mean():.2f} p50={np.median(iters):.0f} "
                  f"max={iters.max()}（上限 {cfg['ellipse_solver']['max_iters']}）")
        md.append("- 每条样本的 `ellipse_iters` / `ellipse_converged` 已写入 NPZ\n")

    if shard_samples:
        reasons = collections.Counter()
        cs = collections.Counter()
        n_points = 0
        covered = 0
        for rec in shard_samples:
            covered += 1
            stats = rec.get("candidate_stats", {})
            n_points += int(stats.get("points", 0))
            for k, v in rec.get("reasons", {}).items():
                reasons[k] += int(v)
            for k, v in stats.items():
                if isinstance(v, (int, float)):
                    cs[k] += v
        n_shard_samples = sum(int(r.get("n_samples", 0)) for r in shard_samples)
        md.append(f"### 候选与拒绝原因明细（覆盖 {covered} 张地图，{n_shard_samples} 个样本）\n")
        md.append(f"- 参与候选生成的路径点：{n_points}")
        md.append(f"- 得到有效标签：{pct(n_shard_samples, n_points)}")
        md.append(f"- 无有效标签的路径点：{cs.get('points_without_label', 0)}")
        md.append(f"- 求解的候选：{cs.get('candidates_examined', 0)} / 剪枝跳过：{cs.get('candidates_pruned', 0)}")
        md.append(f"- 求解成功：{pct(cs.get('candidates_solved', 0), cs.get('candidates_examined', 0))}")
        md.append(f"- 通过全部几何验收：{pct(cs.get('candidates_valid', 0), cs.get('candidates_solved', 0))}")
        md.append(f"- 求解器调用：{cs.get('solver_calls', 0)}（barrier {cs.get('barrier_solves', 0)}，"
                  f"CLARABEL 回退 {cs.get('cvxpy_fallbacks', 0)}）\n")
        md.append("| 拒绝原因 | 次数 |")
        md.append("|---|---|")
        for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
            md.append(f"| `{k}` | {v} |")
        md.append("")

    # --------------------------------------------------------------- splits
    if summary:
        md.append("## 3. 数据集划分\n")
        md.append("- 划分方式：地图组级（同一母地图的全部 `_part_` 分块归入同一 split）")
        md.append(f"- 参与地图：{summary.get('n_maps_with_samples')} / {summary.get('n_maps_selected')}")
        md.append(f"- 原始样本：{summary.get('n_samples_raw')}，过滤后：{summary.get('n_samples_filtered')}\n")
        md.append("| split | 样本数 | 地图数 | 路径数 | 地图组 | 大小 (MB) |")
        md.append("|---|---|---|---|---|---|")
        for name, info in summary.get("splits", {}).items():
            groups = sorted({map_group_id(k) for k in info.get("maps", [])})
            md.append(f"| {name} | {info['n_samples']} | {info['n_maps']} | {info['n_paths']} | "
                      f"{', '.join(groups)} | {info['size_mb']} |")
        md.append("")

    # --------------------------------------------------- path-level quality
    md.append("## 4. 路径级质量\n")
    rows = []
    for name in ("train", "val", "test"):
        path = os.path.join(pp.splits_dir, f"{name}.npz")
        if not os.path.exists(path):
            continue
        with np.load(path) as d:
            if "label_valid" not in d.files or len(d["label_valid"]) == 0:
                continue
            ab = d["ellipse_ab"]; ang = d["ellipse_angle"]; off = d["ellipse_center_offset"]
            res = float(cfg["map"]["resolution"])
            keys = np.stack([d["map_id"], d["path_id"]], axis=1)
            overlap_hits = overlap_tot = 0
            center_steps = []
            coverage = []
            for key in np.unique(keys, axis=0):
                sel = np.flatnonzero((keys[:, 0] == key[0]) & (keys[:, 1] == key[1]))
                if len(sel) < 2:
                    coverage.append(len(sel))
                    continue
                order = sel[np.argsort(d["point_id"][sel])]
                # global ellipse centre = path point + in-patch centre offset
                c = d["point_xy"][order].astype(float) + off[order] / res
                a = ab[order]
                for i in range(len(order) - 1):
                    step = float(np.linalg.norm(c[i + 1] - c[i]))
                    center_steps.append(step)
                    r = min(a[i, 1], a[i + 1, 1]) / res
                    if step <= 2.0 * r:
                        overlap_hits += 1
                    overlap_tot += 1
                coverage.append(len(order))
            rows.append((name, len(np.unique(keys, axis=0)), float(np.mean(coverage)),
                         float(np.mean(center_steps)) if center_steps else 0.0,
                         float(np.max(center_steps)) if center_steps else 0.0,
                         overlap_hits, overlap_tot))
    md.append("| split | 路径数 | 每路径有效点数 mean | 相邻中心步长 mean (cells) | 最大步长 | 相邻椭圆重叠 |")
    md.append("|---|---|---|---|---|---|")
    for name, npaths, cov, step, maxstep, oh, ot in rows:
        md.append(f"| {name} | {npaths} | {cov:.1f} | {step:.2f} | {maxstep:.1f} | {pct(oh, ot)} |")
    md.append("\n相邻椭圆重叠按「相邻中心距 ≤ 2×较短半轴」近似判定，仅作为路径连续性统计，"
              "不用于删除样本。\n")

    # ------------------------------------------------------------ validation
    if validation:
        md.append("## 5. 校验结论\n")
        fails = validation.get("failures", [])
        md.append(f"- 自动检查失败项：**{len(fails)}**")
        for f in fails[:20]:
            md.append(f"  - {f}")
        md.append("- 详细独立复验见 `reports/dataset_quality_report.md`\n")

    out = os.path.join(pp.reports_dir, "dataset_statistics.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
