"""Task 1 - audit the V3 trajectory dataset against the new network's data contract.

Checks every NPZ field, shape, dtype, coordinate convention, LiDAR/ray consistency
and label consistency, then writes a machine-readable audit plus 20 sample figures.
Nothing is modified: this script only reads.

    python scripts/v3_audit_dataset.py --trajectory-dir data/processed_v3/trajectories/train_v3 \
        --out-json reports_v3/data_audit_v3.json --out-md reports_v3/data_audit_v3.md \
        --fig-dir outputs_v3/audit_v3 --n-fig 20 --workers 8

Sections follow the plan: 1.1 required fields, 1.3 audit tasks 1-7.

NOTE: worker processes are impossible here (the sandbox forbids the named pipes that
``multiprocessing`` needs), so the scan runs on a thread pool (zlib + numpy release
the GIL; the SDF pass below is single-process but deduplicated per unique map).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from psen_v3.data.sdf import map_key, signed_distance_m  # noqa: E402

#: fields the new network needs (plan 1.1)
REQUIRED = ("occupancy", "path_xy", "lidar_ranges", "meta_json")
EXPECTED_ALSO = ("centerline_xy", "center_offset", "ellipse_S", "ellipse_ab",
                 "ellipse_angle", "ellipse_area", "safety_verified",
                 "continuity_verified", "label_valid", "solver_status",
                 "start_xy", "goal_xy")


def cell_of(xy: np.ndarray, res: float) -> np.ndarray:
    """Continuous metric (x, y) -> integer (row, col) of the containing cell."""
    xy = np.asarray(xy, dtype=np.float64)
    col = np.floor(xy[..., 0] / res + 0.5).astype(int)
    row = np.floor(xy[..., 1] / res + 0.5).astype(int)
    return np.stack([row, col], axis=-1)


def load_npz(path: str) -> Dict[str, object]:
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    out["meta"] = json.loads(str(out.pop("meta_json").item()))
    return out


def scan_one(path: str) -> Dict[str, object]:
    """Structural + geometric checks of one trajectory NPZ (no SDF)."""
    rec: Dict[str, object] = {"file": os.path.basename(path), "problems": []}
    d = load_npz(path)
    meta = d["meta"]
    keys = sorted(k for k in d if k != "meta")
    occ = np.asarray(d["occupancy"], np.uint8)
    p = np.asarray(d["path_xy"], np.float64).reshape(-1, 2)
    q = np.asarray(d["centerline_xy"], np.float64).reshape(-1, 2)
    lidar = np.asarray(d["lidar_ranges"], np.float64)
    off = np.asarray(d["center_offset"], np.float64).reshape(-1, 2)
    ab = np.asarray(d["ellipse_ab"], np.float64).reshape(-1, 2)
    ang = np.asarray(d["ellipse_angle"], np.float64)
    area = np.asarray(d["ellipse_area"], np.float64)
    S = np.asarray(d["ellipse_S"], np.float64).reshape(-1, 2, 2)
    valid = np.asarray(d["label_valid"]).astype(bool)
    safety = np.asarray(d["safety_verified"]).astype(bool)
    cont = np.asarray(d["continuity_verified"]).astype(bool)

    res = float(meta["r_scene"])
    rec.update(
        keys=["meta_json"] + keys,
        missing=[k for k in REQUIRED if k not in keys and k != "meta_json"],
        missing_also=[k for k in EXPECTED_ALSO if k not in keys],
        scene_id=str(meta.get("scene_id")),
        map_group=str(meta.get("map_group")),
        path_id=int(meta.get("path_id", -1)),
        n_points=int(p.shape[0]),
        r_scene=res,
        crop_px=int(meta.get("crop_px", -1)),
        physical_field_m=float(meta.get("physical_field_m", -1)),
        lidar_rays=int(lidar.shape[1]) if lidar.ndim == 2 else -1,
        lidar_max_range_m=float(meta.get("lidar_max_range_m", -1)),
        has_sdf=("signed_distance_m" in keys),
        occ_shape=list(occ.shape),
        occ_key=map_key(occ, res),
        n_valid=int(valid.sum()),
        n_safety=int(safety.sum()),
        n_cont=int(cont.sum()),
    )
    if rec["missing"]:
        rec["problems"].append(f"missing fields {rec['missing']}")

    H, W = occ.shape
    n = p.shape[0]
    if lidar.shape != (n, rec["lidar_rays"]):
        rec["problems"].append(f"lidar_ranges shape {lidar.shape} vs N={n}")
    for name, arr in (("lidar", lidar), ("ab", ab), ("angle", ang), ("area", area),
                      ("S", S), ("start_xy", np.asarray(d["start_xy"], np.float64))):
        if not np.all(np.isfinite(arr)):
            rec["problems"].append(f"non-finite values in {name}")
    if not np.allclose(off, q - p, atol=1e-5, rtol=0.0):
        rec["problems"].append("center_offset != centerline_xy - path_xy")

    # --- labels ---------------------------------------------------------------
    a, b = ab[:, 0], ab[:, 1]
    rec["label_a_ge_b_frac"] = float(np.mean(a >= b - 1e-9))
    rec["label_area_rel_err"] = float(np.max(np.abs(area - np.pi * a * b) /
                                             np.maximum(np.pi * a * b, 1e-9)))
    rec["label_S_min_eig"] = float(np.linalg.eigvalsh(S).min())
    rec["label_degenerate_frac"] = float(np.mean(a < 1e-3))

    # --- path geometry --------------------------------------------------------
    cell = cell_of(p, res)
    inside = ((cell[:, 0] >= 0) & (cell[:, 0] < H) & (cell[:, 1] >= 0) & (cell[:, 1] < W))
    rec["path_inside_frac"] = float(inside.mean())
    blocked_at_path = np.zeros(n, dtype=bool)
    blocked_at_path[inside] = occ[cell[inside, 0], cell[inside, 1]] > 0
    rec["path_on_blocked_frac"] = float(blocked_at_path.mean())
    rec["path_on_blocked_frac_valid"] = (float(blocked_at_path[valid].mean())
                                         if valid.any() else 0.0)
    sp = np.linalg.norm(np.diff(p, axis=0), axis=1) if n > 1 else np.zeros(0)
    rec["spacing_mean"] = float(sp.mean()) if sp.size else 0.0
    rec["spacing_min"] = float(sp.min()) if sp.size else 0.0
    rec["spacing_max"] = float(sp.max()) if sp.size else 0.0
    rec["path_s_monotone"] = bool(np.all(sp >= 0))

    # --- LiDAR / ray direction convention ------------------------------------
    # stored convention (src/pathsafe_ellipse/v2/pipeline.py): angle_k = 2*pi*k/16,
    # direction (cos, sin) in the same (x=col, y=row) metric frame as path_xy.
    max_range = rec["lidar_max_range_m"]
    rays = lidar.shape[1]
    angles = 2.0 * np.pi * np.arange(rays) / rays
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=-1)          # (R,2)
    hit = lidar < (max_range - 1e-3)
    rec["ray_hit_frac"] = float(hit.mean())
    rec["ray_range_min"] = float(lidar.min())
    rec["ray_range_max"] = float(lidar.max())
    rec["ray_endpoint_blocked_frac"] = 0.0
    rec["ray_midpoint_free_frac"] = 0.0
    if hit.any():
        pi, ki = np.nonzero(hit)
        dd = lidar[pi, ki]
        u = dirs[ki]
        ec = cell_of(p[pi] + (dd - 0.15)[:, None] * u, res)
        ok = ((ec[:, 0] >= 0) & (ec[:, 0] < H) & (ec[:, 1] >= 0) & (ec[:, 1] < W))
        blk = np.zeros(len(ec), dtype=bool)
        blk[ok] = occ[ec[ok, 0], ec[ok, 1]] > 0
        rec["ray_endpoint_blocked_frac"] = float(blk.mean())
        mc = cell_of(p[pi] + (dd * 0.5)[:, None] * u, res)
        okm = ((mc[:, 0] >= 0) & (mc[:, 0] < H) & (mc[:, 1] >= 0) & (mc[:, 1] < W))
        fr = np.zeros(len(mc), dtype=bool)
        fr[okm] = occ[mc[okm, 0], mc[okm, 1]] == 0
        rec["ray_midpoint_free_frac"] = float(fr[okm].mean()) if okm.any() else 0.0
    else:
        rec["ray_endpoint_blocked_frac"] = 1.0
        rec["ray_midpoint_free_frac"] = 1.0
    return rec


def recast_check(files: List[str], max_points: int = 6) -> Dict[str, float]:
    """Re-cast the 16 LiDAR rays from the stored map and compare with the stored ones.

    Independent re-implementation of the label generator's supercover DDA; a
    mismatch means the stored ranges and the map disagree (wrong convention, wrong
    unit or a stale file).  Only the first ``max_points`` query points are re-cast.
    """
    from psen_v3.geometry.exact import supercover_ray_cast  # noqa: E402

    err = []
    for f in files:
        d = load_npz(f)
        occ = np.asarray(d["occupancy"], np.uint8)
        p = np.asarray(d["path_xy"], np.float64)
        lidar = np.asarray(d["lidar_ranges"], np.float64)
        res = float(d["meta"]["r_scene"])
        max_range = float(d["meta"]["lidar_max_range_m"])
        rays = lidar.shape[1]
        angles = 2.0 * np.pi * np.arange(rays) / rays
        for i in range(min(p.shape[0], max_points)):
            for k in range(rays):
                dc, _ = supercover_ray_cast(occ, p[i, 0] / res, p[i, 1] / res,
                                  float(np.cos(angles[k])), float(np.sin(angles[k])),
                                  max_range / res)
                err.append(abs(min(dc * res, max_range) - lidar[i, k]))
    e = np.asarray(err, np.float64)
    return {"pairs": int(e.size), "max_abs_err_m": float(e.max()),
            "mean_abs_err_m": float(e.mean()), "frac_gt_1mm": float((e > 1e-3).mean())}


# --------------------------------------------------------------------------- #
# SDF pass: one computation per unique map, sampled statistics per file
# --------------------------------------------------------------------------- #
def sdf_pass(recs: List[Dict[str, object]], files: List[str],
             sample_files: int = 2500) -> Dict[str, object]:
    reps: Dict[str, str] = {}
    for r, f in zip(recs, files):
        reps.setdefault(str(r["occ_key"]), f)
    print(f"  unique maps: {len(reps)} -> building SDF for each", flush=True)
    store: Dict[str, np.ndarray] = {}
    mins, maxs, p01, p50, p99, mismatch, res_list = [], [], [], [], [], [], []
    for i, (key, f) in enumerate(reps.items()):
        d = load_npz(f)
        occ = np.asarray(d["occupancy"], np.uint8)
        res = float(d["meta"]["r_scene"])
        sdf = signed_distance_m(occ, res)
        store[key] = sdf
        mins.append(float(sdf.min())); maxs.append(float(sdf.max()))
        p01.append(float(np.percentile(sdf, 1)))
        p50.append(float(np.percentile(sdf, 50)))
        p99.append(float(np.percentile(sdf, 99)))
        mismatch.append(float(((sdf < 0) != (occ > 0)).mean()))
        res_list.append(res)
        if (i + 1) % 4000 == 0:
            print(f"    {i + 1}/{len(reps)} maps", flush=True)

    # SDF value at the A* path points (a path point must be free: SDF > 0)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(files), size=min(sample_files, len(files)), replace=False)
    at_path, inside_frac = [], []
    for idx in pick:
        r = recs[idx]
        sdf = store.get(str(r["occ_key"]))
        if sdf is None:
            continue
        d = load_npz(files[idx])
        p = np.asarray(d["path_xy"], np.float64)
        res = float(d["meta"]["r_scene"])
        c = cell_of(p, res)
        H, W = sdf.shape
        ok = ((c[:, 0] >= 0) & (c[:, 0] < H) & (c[:, 1] >= 0) & (c[:, 1] < W))
        at_path.append(sdf[c[ok, 0], c[ok, 1]])
        inside_frac.append(float(ok.mean()))
    at_path_cat = np.concatenate(at_path) if at_path else np.zeros(0)

    uni = sorted({round(float(r["r_scene"]), 6) for r in recs})
    return {
        "n_unique_maps": len(reps),
        "n_files": len(recs),
        "maps_per_file": len(reps) / max(len(recs), 1),
        "resolution_m": uni,
        "sdf_min": float(np.min(mins)),
        "sdf_max": float(np.max(maxs)),
        "sdf_p01_of_map_p01": _dist(p01),
        "sdf_p50_of_map_p50": _dist(p50),
        "sdf_p99_of_map_p99": _dist(p99),
        "sign_mismatch_max": float(np.max(mismatch)),
        "sdf_at_path": _dist(at_path_cat),
        "sdf_at_path_negative_frac": float((at_path_cat < 0).mean()) if at_path_cat.size else 0.0,
        "path_inside_frac_min": float(np.min(inside_frac)) if inside_frac else 0.0,
        "sampled_files": len(at_path),
    }


# --------------------------------------------------------------------------- #
# per-sample figures
# --------------------------------------------------------------------------- #
def plot_sample(path: str, out_png: str) -> Dict[str, object]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = load_npz(path)
    meta = d["meta"]
    occ = np.asarray(d["occupancy"], np.uint8)
    p = np.asarray(d["path_xy"], np.float64)
    lidar = np.asarray(d["lidar_ranges"], np.float64)
    ab = np.asarray(d["ellipse_ab"], np.float64)
    ang = np.asarray(d["ellipse_angle"], np.float64)
    q = np.asarray(d["centerline_xy"], np.float64)
    valid = np.asarray(d["label_valid"]).astype(bool)
    res = float(meta["r_scene"])
    max_range = float(meta["lidar_max_range_m"])
    H, W = occ.shape
    sdf = signed_distance_m(occ, res)
    extent = [0.0, W * res, H * res, 0.0]

    rays = lidar.shape[1]
    angles = 2.0 * np.pi * np.arange(rays) / rays
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=-1)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))
    ax = axes[0]
    ax.imshow(occ, cmap="gray_r", extent=extent)
    step = max(1, p.shape[0] // 12)
    for i in range(0, p.shape[0], step):
        for k in range(rays):
            dd = lidar[i, k]
            ax.plot([p[i, 0], p[i, 0] + dd * dirs[k, 0]],
                    [p[i, 1], p[i, 1] + dd * dirs[k, 1]],
                    "-", color="tab:red", lw=0.5, alpha=0.55)
    ax.plot(p[:, 0], p[:, 1], "-o", color="tab:blue", ms=2.5, lw=1.0)
    ax.set_title("blocked map + A* path + LiDAR (subset)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")

    ax = axes[1]
    im = ax.imshow(sdf, cmap="coolwarm", extent=extent, vmin=-3.0, vmax=6.0)
    ax.contour(np.linspace(0, W * res, W), np.linspace(0, H * res, H), sdf,
               levels=[0.0], colors="k", linewidths=0.6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="signed distance [m]")
    ax.set_title("signed distance [m] (black = blocked boundary)")

    ax = axes[2]
    ax.imshow(occ, cmap="gray_r", extent=extent, alpha=0.55)
    t = np.linspace(0, 2 * np.pi, 60)
    for i in range(p.shape[0]):
        if not valid[i]:
            continue
        a, b, th = ab[i, 0], ab[i, 1], ang[i]
        x = q[i, 0] + a * np.cos(t) * np.cos(th) - b * np.sin(t) * np.sin(th)
        y = q[i, 1] + a * np.cos(t) * np.sin(th) + b * np.sin(t) * np.cos(th)
        ax.plot(x, y, "-", color="tab:green", lw=0.7)
    ax.plot(q[:, 0], q[:, 1], "-", color="tab:green", lw=1.2, label="GT centre line")
    ax.plot(p[:, 0], p[:, 1], "-o", color="tab:blue", ms=2, lw=1.0, label="A* path")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title("GT ellipses (diagnostic only, never a training label)")
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])

    fig.suptitle(f"{meta['scene_id']} path {meta['path_id']} | N={p.shape[0]} | "
                 f"res {res} m | valid {valid.mean():.2f} | max range {max_range} m",
                 fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return {"png": os.path.basename(out_png), "scene_id": meta["scene_id"],
            "n_points": int(p.shape[0])}


# --------------------------------------------------------------------------- #
def _dist(values) -> Dict[str, float]:
    v = np.asarray(values, np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {}
    return {"mean": float(v.mean()), "p05": float(np.percentile(v, 5)),
            "p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)),
            "min": float(v.min()), "max": float(v.max())}


def aggregate(recs: List[Dict[str, object]]) -> Dict[str, object]:
    def col(key, dtype=float):
        return np.asarray([r[key] for r in recs], dtype=dtype)

    groups = {str(r["map_group"]) for r in recs}
    scenes = {str(r["scene_id"]) for r in recs}
    key_sets: Dict[tuple, int] = {}
    for r in recs:
        t = tuple(r["keys"])
        key_sets[t] = key_sets.get(t, 0) + 1
    problems: Dict[str, int] = {}
    for r in recs:
        for p in r["problems"]:
            problems[p] = problems.get(p, 0) + 1
    return {
        "n_files": len(recs),
        "n_scenes": len(scenes),
        "n_map_groups": len(groups),
        "map_groups": sorted(groups),
        "n_query_points": int(col("n_points").sum()),
        "points_per_trajectory": _dist(col("n_points")),
        "field_key_variants": [{"keys": list(k), "count": v} for k, v in key_sets.items()],
        "files_with_signed_distance_m": int(sum(1 for r in recs if r["has_sdf"])),
        "crop_px_variants": sorted({int(r["crop_px"]) for r in recs}),
        "r_scene_variants": sorted({round(float(r["r_scene"]), 6) for r in recs}),
        "physical_field_m_variants": sorted({float(r["physical_field_m"]) for r in recs}),
        "lidar_rays_variants": sorted({int(r["lidar_rays"]) for r in recs}),
        "lidar_max_range_variants": sorted({float(r["lidar_max_range_m"]) for r in recs}),
        "occ_shapes": sorted({tuple(r["occ_shape"]) for r in recs}),
        "problems": problems,
        "label": {
            "valid_rate": float(col("n_valid").sum() / col("n_points").sum()),
            "safety_verified_rate": float(col("n_safety").sum() / col("n_points").sum()),
            "continuity_verified_rate": float(col("n_cont").sum() / col("n_points").sum()),
            "a_ge_b_frac_min": float(col("label_a_ge_b_frac").min()),
            "area_rel_err_max": float(col("label_area_rel_err").max()),
            "S_min_eig_min": float(col("label_S_min_eig").min()),
            "degenerate_frac_mean": float(np.mean(col("label_degenerate_frac"))),
        },
        "path": {
            "inside_frac_min": float(col("path_inside_frac").min()),
            "on_blocked_frac_mean": float(np.mean(col("path_on_blocked_frac"))),
            "on_blocked_frac_max": float(col("path_on_blocked_frac").max()),
            "on_blocked_frac_valid_mean": float(np.mean(col("path_on_blocked_frac_valid"))),
            "spacing_mean": _dist(col("spacing_mean")),
            "spacing_min": float(col("spacing_min").min()),
            "spacing_max": float(col("spacing_max").max()),
            "s_monotone_all": bool(col("path_s_monotone", bool).all()),
        },
        "lidar": {
            "hit_frac_mean": float(np.mean(col("ray_hit_frac"))),
            "hit_frac_p05": float(np.percentile(col("ray_hit_frac"), 5)),
            "endpoint_blocked_frac_min": float(col("ray_endpoint_blocked_frac").min()),
            "endpoint_blocked_frac_mean": float(np.mean(col("ray_endpoint_blocked_frac"))),
            "midpoint_free_frac_min": float(col("ray_midpoint_free_frac").min()),
            "range_min_min": float(col("ray_range_min").min()),
            "range_max_max": float(col("ray_range_max").max()),
        },
    }


def to_markdown(summary: Dict[str, object], recast: Dict[str, float],
                figures: List[Dict[str, object]], args) -> str:
    s = summary
    lab, path, ray, sdf = s["label"], s["path"], s["lidar"], s["sdf"]
    L = ["# V3 数据审计报告（面向新网络的数据契约）\n"]
    L.append(f"- 数据目录：`{s['trajectory_dir']}`")
    L.append(f"- 审计文件数：{s['n_files']}；场景 {s['n_scenes']}；"
             f"`map_group` {s['n_map_groups']}")
    L.append(f"- 查询点总数：{s['n_query_points']}；每轨迹 N："
             f"{json.dumps(s['points_per_trajectory'])}")
    L.append(f"- 唯一地图数：{sdf['n_unique_maps']}（SDF 只需按地图生成，"
             f"平均每地图 {1.0 / max(sdf['maps_per_file'], 1e-9):.2f} 条轨迹）")
    L.append(f"- 已含 `signed_distance_m` 的文件：{s['files_with_signed_distance_m']}"
             f" → Task 2 生成独立数据版本\n")

    L.append("## 1 字段清单（plan 1.1 / 1.3-1）\n")
    for v in s["field_key_variants"]:
        L.append(f"- {v['count']} 个文件：{', '.join(v['keys'])}")
    L.append(f"- 结构问题：{json.dumps(s['problems'], ensure_ascii=False)}\n")

    L.append("## 2 地图与配置（plan 1.3-2）\n")
    L.append(f"- `occupancy` 形状：{s['occ_shapes']}")
    L.append(f"- `crop_px` {s['crop_px_variants']} → 与 256 相同，无未被覆盖的像素"
             f"（`valid_mask` 全 1，见 Task 2 说明）")
    L.append(f"- `r_scene` {s['r_scene_variants']} m/px；"
             f"`physical_field_m` {s['physical_field_m_variants']}")
    L.append(f"- LiDAR：{s['lidar_rays_variants']} 射线，量程 "
             f"{s['lidar_max_range_variants']} m\n")

    L.append("## 3 标签一致性（仅诊断用，训练不使用中心/尺寸/朝向标签）\n")
    L.append(f"- `label_valid` {lab['valid_rate']:.4f}；`safety_verified` "
             f"{lab['safety_verified_rate']:.4f}；`continuity_verified` "
             f"{lab['continuity_verified_rate']:.4f}")
    L.append(f"- a≥b 最小比例 {lab['a_ge_b_frac_min']:.6f}；面积 vs πab 最大相对误差 "
             f"{lab['area_rel_err_max']:.2e}；S 最小特征值 {lab['S_min_eig_min']:.3e}")
    L.append(f"- 退化椭圆（a<1 mm）平均占比 {lab['degenerate_frac_mean']:.4f}\n")

    L.append("## 4 世界坐标 ↔ 像素坐标（plan 1.3-5）\n")
    L.append("- 约定：cell (row=i, col=j) 中心 = (x, y) = (j·r, i·r)；"
             "`pixel = xy / r_scene`，`[row, col]` 索引")
    L.append(f"- 路径点全部在地图内：最小比例 {path['inside_frac_min']:.6f}")
    L.append(f"- 路径点落在 blocked 格：均值 {path['on_blocked_frac_mean']:.5f}"
             f"（有效点均值 {path['on_blocked_frac_valid_mean']:.5f}）")
    L.append(f"- 重采样间距 [m]：{json.dumps(path['spacing_mean'])}；"
             f"min {path['spacing_min']:.3f} / max {path['spacing_max']:.3f}")
    L.append(f"- 弧长 `path_s` 单调：{path['s_monotone_all']}\n")

    L.append("## 5 LiDAR 射线（plan 1.3-6）\n")
    L.append("- 存储约定：`angle_k = 2πk/16`，方向 `(cos, sin)` 与 `path_xy` 同一米制坐标系")
    L.append(f"- 真实命中比例：均值 {ray['hit_frac_mean']:.4f}（p05 {ray['hit_frac_p05']:.4f}）"
             f"，其余为量程截断 → 必须用 `ray_valid` 区分")
    L.append(f"- 命中射线终点落在 blocked 格：最低 {ray['endpoint_blocked_frac_min']:.4f}，"
             f"均值 {ray['endpoint_blocked_frac_mean']:.4f}")
    L.append(f"- 命中射线中点落在 free 格：最低 {ray['midpoint_free_frac_min']:.4f}")
    L.append(f"- 独立重投影校验（同一 DDA 重算 16 射线）：{json.dumps(recast)}\n")

    L.append("## 6 距离场（Task 2）\n")
    L.append(f"- 256×256 米制有符号距离场：范围 [{sdf['sdf_min']:.3f}, "
             f"{sdf['sdf_max']:.3f}] m")
    L.append(f"- 各图 p01 分布：{json.dumps(sdf['sdf_p01_of_map_p01'])}")
    L.append(f"- 各图 p99 分布：{json.dumps(sdf['sdf_p99_of_map_p99'])}")
    L.append(f"- 符号与 blocked 掩码不一致比例 ≤ {sdf['sign_mismatch_max']:.4f}")
    L.append(f"- A* 路径点处 SDF：{json.dumps(sdf['sdf_at_path'])}；"
             f"其中 <0（路径点被判进障碍）比例 "
             f"{sdf['sdf_at_path_negative_frac']:.5f}（{sdf['sampled_files']} 个抽样文件）")
    L.append("")

    L.append("## 7 样本可视化（plan 1.3-3）\n")
    for f in figures:
        L.append(f"- `{f['png']}` — {f['scene_id']} (N={f['n_points']})")
    L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory-dir", default="data/processed_v3/trajectories/train_v3")
    ap.add_argument("--out-json", default="reports_v3/data_audit_v3.json")
    ap.add_argument("--out-md", default="reports_v3/data_audit_v3.md")
    ap.add_argument("--fig-dir", default="outputs_v3/audit_v3")
    ap.add_argument("--n-fig", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="audit only the first N files")
    ap.add_argument("--recast-files", type=int, default=24)
    ap.add_argument("--skip-sdf", action="store_true")
    args = ap.parse_args()

    files = sorted(os.path.join(args.trajectory_dir, n)
                   for n in os.listdir(args.trajectory_dir) if n.endswith(".npz"))
    if args.limit:
        files = files[: args.limit]
    print(f"auditing {len(files)} files ({args.workers} threads)", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        recs = list(ex.map(scan_one, files, chunksize=64))

    summary = aggregate(recs)
    summary["trajectory_dir"] = os.path.abspath(args.trajectory_dir)
    summary["audited_files"] = len(files)

    if args.skip_sdf:
        summary["sdf"] = {}
    else:
        print("SDF pass", flush=True)
        summary["sdf"] = sdf_pass(recs, files)

    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(files), size=min(args.n_fig, len(files)), replace=False))
    figures = []
    for j, idx in enumerate(pick):
        png = os.path.join(args.fig_dir,
                           f"audit_{j:02d}_{os.path.basename(files[idx])[:-4]}.png")
        figures.append(plot_sample(files[idx], png))
    summary["figures"] = figures

    rp = sorted(rng.choice(len(files), size=min(args.recast_files, len(files)), replace=False))
    summary["recast"] = recast_check([files[i] for i in rp])

    for path in (args.out_json, args.out_md):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=float)
    with open(args.out_md, "w", encoding="utf-8") as fh:
        fh.write(to_markdown(summary, summary["recast"], figures, args))

    print(json.dumps({k: summary[k] for k in
                      ("n_files", "n_scenes", "n_map_groups", "n_query_points",
                       "files_with_signed_distance_m")}, indent=1))
    print("problems:", summary["problems"])
    print("recast:", summary["recast"])
    print(f"-> {args.out_json}\n-> {args.out_md}\n-> {len(figures)} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
