"""Independent dataset validation + quality report (section 11.4 / S5 acceptance).

Nothing here trusts the generator: labels are re-read from the NPZ, the ellipse
geometry is rebuilt from ``(a, b, sin, cos, centre offset)`` and re-checked
against the *stored* patch with the exact cell/ellipse test, the rays are
recomputed from the patch and compared with the stored features, and the split
files are checked for map-group / path leakage.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pathsafe_ellipse.config import load_config, project_paths, setup_logging  # noqa: E402
from pathsafe_ellipse.ellipse_validation import (  # noqa: E402
    ellipse_obstacle_intersections, point_metric,
)
from pathsafe_ellipse.local_features import _cast_one, ray_directions  # noqa: E402
from pathsafe_ellipse.map_io import build_map_catalog, map_key  # noqa: E402


def rebuild_L(a_m: float, b_m: float, sin_t: float, cos_t: float, resolution: float) -> np.ndarray:
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    return R @ np.diag([a_m / resolution, b_m / resolution]) @ R.T


def load_split(path: str) -> Dict[str, np.ndarray]:
    """Load a split without materialising the unpacked patches (memory)."""
    with np.load(path) as data:
        out = {k: data[k] for k in data.files if k != "map_patch"}
    if "map_patch_packed" not in out:
        raise KeyError(f"{path}: map_patch_packed is required")
    return out


def patch_of(packed_row: np.ndarray) -> np.ndarray:
    """Unpack one (128, 16) bit-packed patch back to (128, 128).

    ``np.packbits`` was applied along the last axis, so bits must be restored
    along the same axis (byte j of row i holds columns 8j..8j+7).
    """
    return np.unpackbits(packed_row, axis=1).reshape(128, 128)


def main() -> int:
    args = argparse.ArgumentParser()
    args.add_argument("--config", default=None)
    args.add_argument("--tag", default="")
    args.add_argument("--max-check", type=int, default=4000,
                      help="per split: how many samples get the expensive re-verification")
    ns = args.parse_args()

    cfg = load_config(ns.config)
    setup_logging(cfg, "validate_dataset")
    pp = project_paths(cfg)
    res = float(cfg["map"]["resolution"])
    eta = float(cfg["label"]["point_inside_eta"])
    ray_count = int(cfg["local_features"]["ray_count"])
    ray_max = float(cfg["local_features"]["ray_max_range"])
    dirs = ray_directions(ray_count)
    rng = np.random.default_rng(0)

    catalog = [r for r in build_map_catalog(pp.map_dir, cfg["map"]["resolution"]) if r.parsed_ok]
    keys = [map_key(r) for r in catalog]
    group_of = {map_key(r): r.group_id for r in catalog}
    split_path = os.path.join(pp.splits_dir, f"map_group_split{ns.tag}.json")
    with open(split_path, "r", encoding="utf-8") as fh:
        split_meta = json.load(fh)
    split_groups = split_meta["split"]

    report: Dict[str, object] = {"splits": {}, "checks": {}, "config": {
        "eta": eta, "resolution": res, "ray_max_range": ray_max, "ray_count": ray_count,
        "min_semi_axis_m": cfg["label"]["min_semi_axis"],
        "max_semi_axis_m": cfg["label"]["max_semi_axis"],
    }}
    failures: List[str] = []
    seen_groups: Dict[str, str] = {}
    seen_paths: Dict[Tuple[int, int], str] = {}

    for name in ("train", "val", "test"):
        path = os.path.join(pp.splits_dir, f"{name}.npz")
        if not os.path.exists(path):
            continue
        d = load_split(path)
        n = int(len(d["label_valid"]))
        info: Dict[str, object] = {"n_samples": n, "file": path,
                                   "size_mb": round(os.path.getsize(path) / 1e6, 2)}
        # ---- structural checks
        lengths = {k: len(v) for k, v in d.items()}
        info["array_lengths_consistent"] = len(set(lengths.values())) == 1
        if not info["array_lengths_consistent"]:
            failures.append(f"{name}: inconsistent array lengths {lengths}")
        finite_ok = True
        for field in ("ray_distances", "ray_distances_raw", "path_tangent", "ellipse_ab",
                      "ellipse_angle", "ellipse_center_offset", "ellipse_area",
                      "point_inside_metric", "offset_ratio", "clearance"):
            if field in d and not np.all(np.isfinite(d[field])):
                finite_ok = False
                failures.append(f"{name}: non-finite values in {field}")
        info["all_finite"] = finite_ok

        packed = d["map_patch_packed"]
        centre_idx = rng.permutation(n)[: min(5000, n)]
        occupied_centre = sum(int(patch_of(packed[i])[64, 64] != 0) for i in centre_idx)
        info["patch_centre_free"] = bool(occupied_centre == 0)
        info["patch_centre_checked"] = int(len(centre_idx))
        if occupied_centre:
            failures.append(f"{name}: {occupied_centre} samples have an occupied patch centre")

        groups_in_split = {group_of[keys[int(m)]] for m in np.unique(d["map_id"])}
        info["groups"] = sorted(groups_in_split)
        info["n_maps"] = int(len(np.unique(d["map_id"])))
        info["n_paths"] = int(len({(int(m), int(p)) for m, p in zip(d["map_id"], d["path_id"])}))
        for g in groups_in_split:
            if g in seen_groups and seen_groups[g] != name:
                failures.append(f"map group {g} appears in both {seen_groups[g]} and {name}")
            seen_groups[g] = name
        for m, p in zip(d["map_id"], d["path_id"]):
            key = (int(m), int(p))
            if key in seen_paths and seen_paths[key] != name:
                failures.append(f"path {keys[key[0]]}#{p} appears in both {seen_paths[key]} and {name}")
            seen_paths[key] = name

        # ---- independent geometric re-verification on a random subset
        idx = rng.permutation(n)[: min(ns.max_check, n)]
        n_safe = n_inside = n_axes = 0
        worst_rho = 0.0
        min_quad = np.inf
        ray_err = 0.0
        for i in idx:
            a_m, b_m = (float(x) for x in d["ellipse_ab"][i])
            sin_t, cos_t = (float(x) for x in d["ellipse_angle"][i])
            off = d["ellipse_center_offset"][i] / res
            centre = np.array([64.0 + off[0], 64.0 + off[1]])
            L = rebuild_L(a_m, b_m, sin_t, cos_t, res)
            collides, q, _ = ellipse_obstacle_intersections(L, centre, patch_of(packed[i]))
            if not collides:
                n_safe += 1
            min_quad = min(min_quad, q)
            rho = point_metric(L, centre, np.array([64.0, 64.0]))
            worst_rho = max(worst_rho, rho)
            if rho <= eta + 1e-6:
                n_inside += 1
            if (b_m >= cfg["label"]["min_semi_axis"] - 1e-9
                    and a_m <= cfg["label"]["max_semi_axis"] + 1e-9 and a_m >= b_m - 1e-9):
                n_axes += 1
            # independent ray recomputation
            patch_i = patch_of(packed[i])
            for k in range(ray_count):
                dist, hit = _cast_one(patch_i, 64.0, 64.0, float(dirs[k, 0]), float(dirs[k, 1]), ray_max)
                expect = min(dist, ray_max) / ray_max if hit else 1.0
                ray_err = max(ray_err, abs(expect - float(d["ray_distances"][i][k])))
        n_checked = int(len(idx))
        info["reverified"] = n_checked
        info["ellipse_safe"] = [n_safe, n_checked]
        info["point_inside"] = [n_inside, n_checked]
        info["axes_in_range"] = [n_axes, n_checked]
        info["worst_rho"] = round(worst_rho, 6)
        info["min_obstacle_quadratic"] = (None if not np.isfinite(min_quad) else round(float(min_quad), 6))
        info["max_ray_deviation"] = round(float(ray_err), 6)
        if n_safe != n_checked:
            failures.append(f"{name}: {n_checked - n_safe} re-verified ellipses intersect obstacles")
        if n_inside != n_checked:
            failures.append(f"{name}: {n_checked - n_inside} re-verified samples violate rho <= eta")
        if ray_err > 1e-4:
            failures.append(f"{name}: stored rays deviate from recomputed rays by {ray_err:.4f}")

        # ---- distributions
        ab = d["ellipse_ab"]
        info["ellipse_ab_m"] = {
            "a": dict(zip(("p5", "p50", "p95"), np.percentile(ab[:, 0], [5, 50, 95]).round(3).tolist())),
            "b": dict(zip(("p5", "p50", "p95"), np.percentile(ab[:, 1], [5, 50, 95]).round(3).tolist())),
            "axis_ratio_p50": round(float(np.median(ab[:, 0] / np.maximum(ab[:, 1], 1e-9))), 3),
        }
        info["ellipse_area_m2"] = dict(zip(
            ("p5", "p50", "p95"),
            np.percentile(d["ellipse_area"], [5, 50, 95]).round(3).tolist()))
        info["offset_direction_hist"] = {int(k): int(v) for k, v in
                                        sorted(collections.Counter(d["offset_direction"].tolist()).items())}
        info["offset_ratio_hist"] = {float(k): int(v) for k, v in
                                     sorted(collections.Counter(np.round(d["offset_ratio"], 3).tolist()).items())}
        info["unbiased_fraction"] = round(float(np.mean(d["offset_direction"] == 16)), 4)
        info["angle_valid_fraction"] = round(float(np.mean(d["angle_valid"])), 4)
        if "ellipse_converged" in d:
            info["iris_converged_fraction"] = round(float(np.mean(d["ellipse_converged"])), 4)
            info["ellipse_iters_p50"] = float(np.median(d["ellipse_iters"]))
        if "n_candidates" in d:
            info["n_candidates_p50"] = float(np.median(d["n_candidates"]))
            info["n_valid_candidates_mean"] = round(float(np.mean(d["n_candidates_valid"])), 2)
        if "clearance_class" in d:
            names = {0: "compact(1-3)", 1: "medium(3-8)", 2: "open(>8)"}
            info["clearance_class_hist"] = {names[int(k)]: int(v) for k, v in
                                            sorted(collections.Counter(d["clearance_class"].tolist()).items())}
        info["clearance_cells"] = dict(zip(
            ("p5", "p50", "p95"),
            np.percentile(d["clearance"], [5, 50, 95]).round(2).tolist()))
        info["point_inside_metric"] = dict(zip(
            ("p50", "p95", "max"),
            np.concatenate([np.percentile(d["point_inside_metric"], [50, 95]),
                            [d["point_inside_metric"].max()]]).round(4).tolist()))
        report["splits"][name] = info

    # ------------------------------------------------------------- markdown
    md: List[str] = []
    md.append("# PathSafe-Ellipse :: dataset quality report\n")
    md.append("### 1. 数据集构成\n")
    md.append("| split | 样本数 | 地图数 | 路径数 | 地图组 | 文件大小 (MB) |")
    md.append("|---|---|---|---|---|---|")
    for name, info in report["splits"].items():
        md.append(f"| {name} | {info['n_samples']} | {info['n_maps']} | {info['n_paths']} | "
                  f"{', '.join(info['groups'])} | {info['size_mb']} |")
    total = sum(i["n_samples"] for i in report["splits"].values())
    md.append(f"\n合计 **{total}** 个有效样本。划分方式：地图组级（同一母地图的所有 `_part_` 分块"
              f"归入同一 split），文件 `{os.path.basename(split_path)}`。\n")

    md.append("### 2. 独立复验（不信任生成过程，从 NPZ 重建几何）\n")
    md.append("| split | 复验样本 | 椭圆无碰撞 | ρ ≤ η | 半轴范围 | 最小障碍二次型 | 最大射线偏差 (cells) |")
    md.append("|---|---|---|---|---|---|---|")
    for name, info in report["splits"].items():
        md.append(f"| {name} | {info['reverified']} | {info['ellipse_safe'][0]}/{info['ellipse_safe'][1]} | "
                  f"{info['point_inside'][0]}/{info['point_inside'][1]} | "
                  f"{info['axes_in_range'][0]}/{info['axes_in_range'][1]} | "
                  f"{info['min_obstacle_quadratic']} | {info['max_ray_deviation']} |")
    md.append("\n`最小障碍二次型 > 1` 表示椭圆与所有膨胀障碍栅格严格不相交（连续几何判定，"
              "非采样近似）。`最大射线偏差 ≈ 0` 表示存储的 16 条射线与从 patch 重新投射的结果一致。\n")

    md.append("### 3. 标签分布\n")
    md.append("| split | 椭圆 a (m) p5/p50/p95 | 椭圆 b (m) p5/p50/p95 | 面积 (m²) p5/p50/p95 | 偏置比例 a/b p50 | 不偏置占比 | angle_valid |")
    md.append("|---|---|---|---|---|---|---|")
    for name, info in report["splits"].items():
        a = info["ellipse_ab_m"]["a"]; b = info["ellipse_ab_m"]["b"]; ar = info["ellipse_area_m2"]
        md.append(f"| {name} | {a['p5']}/{a['p50']}/{a['p95']} | {b['p5']}/{b['p50']}/{b['p95']} | "
                  f"{ar['p5']}/{ar['p50']}/{ar['p95']} | {info['ellipse_ab_m']['axis_ratio_p50']} | "
                  f"{info['unbiased_fraction']} | {info['angle_valid_fraction']} |")

    md.append("\n### 4. 偏置方向分布（0-15 射线方向，16 = 不偏置）\n")
    for name, info in report["splits"].items():
        hist = info["offset_direction_hist"]
        md.append(f"- **{name}**: " + ", ".join(f"{k}:{v}" for k, v in hist.items()))
    md.append("\n### 5. 偏置比例分布\n")
    for name, info in report["splits"].items():
        hist = info["offset_ratio_hist"]
        md.append(f"- **{name}**: " + ", ".join(f"{k:.2f}:{v}" for k, v in hist.items()))

    md.append("\n### 6. 净空分层与局部净空\n")
    for name, info in report["splits"].items():
        md.append(f"- **{name}**: 净空类别 {info.get('clearance_class_hist')}；"
                  f"净空 (cells) p5/p50/p95 = {info['clearance_cells']}；"
                  f"ρ p50/p95/max = {info['point_inside_metric']}")
    if "iris_converged_fraction" in next(iter(report["splits"].values()), {}):
        md.append("\n### 7. IRIS 求解统计\n")
        for name, info in report["splits"].items():
            md.append(f"- **{name}**: 收敛比例 {info['iris_converged_fraction']}, "
                      f"迭代次数中位数 {info['ellipse_iters_p50']}, "
                      f"候选数中位数 {info['n_candidates_p50']}, "
                      f"有效候选中位数 {info['n_valid_candidates_mean']}")

    md.append("\n### 8. 自动检查结论\n")
    if failures:
        md.append("**未通过检查：**\n")
        for f in failures[:50]:
            md.append(f"- {f}")
    else:
        md.append("全部自动检查通过：数组长度一致、无 NaN/Inf、patch 中心自由、"
                  "地图组与路径均不跨 split、椭圆独立复验无碰撞、路径点满足 ρ ≤ η、射线特征可复现。\n")

    md_path = os.path.join(pp.reports_dir, "dataset_quality_report.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    with open(os.path.join(pp.reports_dir, "validation_summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"report": report, "failures": failures}, fh, indent=2, ensure_ascii=False)
    print(f"wrote {md_path}")
    print(f"failures: {len(failures)}")
    for f in failures[:20]:
        print("  -", f)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
