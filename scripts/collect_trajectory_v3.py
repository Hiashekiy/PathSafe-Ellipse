"""V3 trajectory-level collection (parallel, resumable).

    # small debug run (sequential, with figures)
    python scripts/collect_trajectory_v3.py --target 10 --paths-per-scene 2 --tag pilot10

    # bulk production (14 processes, no per-file figures)
    python scripts/collect_trajectory_v3.py --target 50000 --paths-per-scene 3 \
        --workers 14 --no-viz --tag train_v3

Every accepted path becomes ONE NPZ containing the 256x256 inflated occupancy, the
resampled A* path, the optimised centre line, per-point LiDAR and per-point ellipse
labels (strictly aligned).  Geometry code is unchanged.

Resuming: finished scene slots are appended to ``progress.jsonl`` in the trajectory
directory; re-running the same command skips them and continues.  NPZ writes are
atomic, so an interrupted run never leaves a corrupt sample behind.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from types import SimpleNamespace as NS
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from pathsafe_ellipse.map_io import map_group_id  # noqa: E402
from pathsafe_ellipse.v2.dataset_v3 import (  # noqa: E402
    check_sample, load_trajectory, summarise_files,
)
from pathsafe_ellipse.v2.map_cache import load_catalog_cached, load_map_cached  # noqa: E402
from pathsafe_ellipse.v2.ellipse import exact_clearance, midpoint_metric  # noqa: E402
from pathsafe_ellipse.v2.pipeline import lidar_ranges as cast_lidar  # noqa: E402
from pathsafe_ellipse.v2.pipeline import process_scene  # noqa: E402
from pathsafe_ellipse.v2.pilot import (  # noqa: E402
    git_commit, save_trajectory_npz, trajectory_metadata, write_index,
)
from pathsafe_ellipse.v2.scene import build_scene, free_components  # noqa: E402
from pathsafe_ellipse.v2.scene_slots import build_slots, load_cfg  # noqa: E402


# --------------------------------------------------------------------------- #
# Figures (small runs only)
# --------------------------------------------------------------------------- #
def draw_trajectory(ax, sample: Dict[str, object], title: str = "") -> None:
    occ = np.asarray(sample["occupancy"])
    rs = float(sample["meta"]["r_scene"])
    ext = [0, occ.shape[1] * rs, occ.shape[0] * rs, 0]
    ax.imshow(occ, cmap="gray_r", alpha=0.55, interpolation="nearest", extent=ext)
    p = np.asarray(sample["path_xy"], float)
    q = np.asarray(sample["centerline_xy"], float)
    lidar = np.asarray(sample["lidar_ranges"], float)
    angles = 2.0 * np.pi * np.arange(lidar.shape[1]) / lidar.shape[1]
    for i in range(len(p)):
        for k, a in enumerate(angles):
            ax.plot([p[i, 0], p[i, 0] + lidar[i, k] * np.cos(a)],
                    [p[i, 1], p[i, 1] + lidar[i, k] * np.sin(a)],
                    "-", color="tab:cyan", lw=0.2, alpha=0.3, zorder=1)
    ax.plot(p[:, 0], p[:, 1], "-", lw=1.1, color="tab:orange", zorder=3)
    ax.plot(p[:, 0], p[:, 1], "o", ms=2.2, color="tab:orange", zorder=4)
    ax.plot(q[:, 0], q[:, 1], "-", lw=1.2, color="tab:blue", alpha=0.9, zorder=4)
    ab = np.asarray(sample["ellipse_ab"], float)
    ang = np.asarray(sample["ellipse_angle"], float)
    valid = np.asarray(sample["label_valid"]).astype(bool)
    for i in np.flatnonzero(valid):
        ax.add_patch(Ellipse((q[i, 0], q[i, 1]), 2 * ab[i, 0], 2 * ab[i, 1],
                             angle=np.degrees(float(ang[i])), fill=False,
                             ec="tab:green", lw=0.6, alpha=0.8, zorder=5))
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


#: scene ids already on disk — a per-process set instead of a directory scan per slot
#: (24 processes globbing one 50k-entry directory serialise on the NTFS directory lock)
_EXISTING_SCENES: set = set()


# --------------------------------------------------------------------------- #
# One scene slot -> trajectories on disk (runs inside a worker process)
# --------------------------------------------------------------------------- #
def collect_slot(task: Dict[str, object]) -> Dict[str, object]:
    """Collect every accepted trajectory of one map/crop slot and save its NPZs."""
    t0 = time.perf_counter()
    cfg = load_cfg(str(task["config"]))
    cfg["path"]["paths_per_scene"] = int(task["paths_per_scene"])
    slot = task["slot"]
    rec = slot["rec"]
    out: Dict[str, object] = {"key": task["key"], "rows": [], "errors": [],
                              "scene_id": "", "n_files": 0, "skipped": False}
    try:
        src, smeta = load_map_cached(cfg["paths"]["map_dir"], rec.rel_path,
                                    str(task["cache_dir"]))
        r0 = float(smeta["resolution"] or 0.25)
        field_m = float(slot["field_m"])
        crop_px = int(round(field_m / r0))
        h, w = src.shape
        if crop_px >= h or crop_px >= w:
            out["errors"].append(f"map {w}x{h} < crop {crop_px}")
            return out
        rng = np.random.default_rng(slot["seed"])
        x0 = int(rng.integers(0, w - crop_px))
        y0 = int(rng.integers(0, h - crop_px))
        scene_id = (f"{os.path.splitext(rec.file_name)[0]}_c{x0}_{y0}"
                    f"_f{int(field_m)}_k{slot['crop']}")
        out["scene_id"] = scene_id
        if task.get("skip_existing", True) and scene_id in _EXISTING_SCENES:
            out["skipped"] = True              # slot already collected (any shard layout)
            out["ms"] = (time.perf_counter() - t0) * 1e3
            return out
        scene = build_scene(src, r0, (x0, y0), crop_px, int(cfg["scene"]["size"]), field_m,
                            float(cfg["robot"]["radius_m"]),
                            float(cfg["robot"]["safety_margin_m"]),
                            int(cfg["robot"]["edt_upsample"]), scene_id, rec.file_name)
        res = process_scene(scene, cfg, np.random.default_rng(slot["seed"]))
        group = map_group_id(rec.file_name)
        n_comp = free_components(scene.free)[1]
        for j, traj in enumerate(res.trajectories):
            path_id = int(traj["path_id"])
            path_rec = next((r for r in res.paths if int(r["path_id"]) == path_id), None)
            meta = trajectory_metadata(scene, cfg, seed=int(slot["seed"]), path_id=path_id,
                                      source_rel_path=rec.rel_path, map_group=group,
                                      path_record=path_rec)
            meta["free_components"] = int(n_comp)
            meta["n_points"] = int(len(traj["path_xy"]))
            meta["n_base_points"] = int(traj["n_base_points"])
            meta["n_inserted"] = int(traj["n_inserted"])
            dst = save_trajectory_npz(
                scene=scene, path_xy=traj["path_xy"],
                centerline_xy=traj["centerline_xy"], lidar_ranges=traj["lidar_ranges"],
                labels=traj, start_xy=traj["start_xy"], goal_xy=traj["goal_xy"],
                meta=meta, out_dir=str(task["out_dir"]))
            sample = load_trajectory(dst) if task.get("viz") and j < int(
                task.get("viz_limit", 0)) else None
            if sample is not None:
                stem = os.path.splitext(os.path.basename(dst))[0]
                fig, ax = plt.subplots(figsize=(6.0, 6.0))
                draw_trajectory(ax, sample, f"{scene_id} path {path_id}  "
                                            f"N={len(sample['path_xy'])}")
                fig.tight_layout()
                fig.savefig(os.path.join(str(task["viz_dir"]), f"{stem}.png"), dpi=110)
                plt.close(fig)
            out["rows"].append({
                "file": os.path.basename(dst), "scene_id": scene_id,
                "map_group": group, "source_map_id": rec.file_name,
                "path_id": path_id, "n_points": int(len(traj["path_xy"])),
                "n_base_points": int(traj["n_base_points"]),
                "n_inserted": int(traj["n_inserted"]),
                "n_label_valid": int(np.asarray(traj["label_valid"]).sum()),
                "n_safety": int(np.asarray(traj["safety_verified"]).sum()),
                "n_continuity": int(np.asarray(traj["continuity_verified"]).sum()),
                "physical_field_m": field_m, "r_scene": float(scene.r_scene),
                "label_status": str(path_rec.get("label_status", "")) if path_rec else "",
                "size_bytes": os.path.getsize(dst),
            })
            out["n_files"] = int(out["n_files"]) + 1
            _EXISTING_SCENES.add(scene_id)
    except Exception as exc:                                  # never kill the worker
        out["errors"].append(f"{type(exc).__name__}: {exc}")
    out["ms"] = (time.perf_counter() - t0) * 1e3
    return out


# --------------------------------------------------------------------------- #
# Verification (spec 6.1 / 6.2)
# --------------------------------------------------------------------------- #
def verify_dataset(files: List[str], cfg: Dict, deep_limit: int = 200) -> Dict[str, object]:
    """Structural + cheap geometry checks on every file, exact re-checks on a sample."""
    size = int(cfg["scene"]["size"])
    rho_max = float(cfg["continuity"]["rho_max"])
    idx_deep: set = set()
    if deep_limit > 0 and files:
        idx_deep = {int(i) for i in np.linspace(0, len(files) - 1,
                                                min(deep_limit, len(files))).astype(int)}
    rows: List[Dict[str, object]] = []
    tot = {"points": 0, "safe": 0, "safe_ok": 0, "pairs": 0, "pairs_ok": 0,
           "path_blocked": 0, "centerline_blocked": 0, "lidar_max_err": 0.0,
           "struct_checked": 0, "struct_bad": 0, "deep_checked": 0, "deep_bad": 0}
    for k, path in enumerate(files):
        s = load_trajectory(path)
        meta = s.get("meta", {})
        p = np.asarray(s["path_xy"], np.float64)
        n = int(len(p))
        safety = np.asarray(s["safety_verified"]).astype(bool)
        struct = check_sample(s, size=size)
        tot["struct_checked"] += 1

        # cheap geometry gate, every file: query points must lie in traversable cells
        blocked_pts = 0
        if "r_scene" in meta:
            rs = float(meta["r_scene"])
            occ = np.asarray(s["occupancy"])
            cc = np.clip(np.rint(p[:, 0] / rs).astype(int), 0, occ.shape[1] - 1)
            rr = np.clip(np.rint(p[:, 1] / rs).astype(int), 0, occ.shape[0] - 1)
            blocked_pts = int(np.sum(occ[rr, cc] != 0))
            tot["path_blocked"] += blocked_pts
            if blocked_pts:
                struct.append(f"{blocked_pts} query points in blocked cells")
        tot["struct_bad"] += int(bool(struct))

        deep = k in idx_deep
        problems = list(struct)
        safe_ok = pairs = pairs_ok = 0
        err = 0.0
        if deep:
            tot["deep_checked"] += 1
            rs = float(meta["r_scene"])
            rays = int(meta["lidar_rays"])
            max_range = float(meta["lidar_max_range_m"])
            occ = np.asarray(s["occupancy"]).astype(np.uint8)
            q = np.asarray(s["centerline_xy"], np.float64)
            S = np.asarray(s["ellipse_S"], np.float64)
            cont_flags = np.asarray(s["continuity_verified"]).astype(np.uint8)
            ccq = np.clip(np.rint(q[:, 0] / rs).astype(int), 0, occ.shape[1] - 1)
            rrq = np.clip(np.rint(q[:, 1] / rs).astype(int), 0, occ.shape[0] - 1)
            centerline_blocked = int(np.sum(occ[rrq, ccq] != 0))
            tot["centerline_blocked"] += centerline_blocked
            if centerline_blocked:
                problems.append(f"{centerline_blocked} centre-line points in blocked cells")

            for i in np.flatnonzero(safety):
                safe_ok += int(bool(exact_clearance(S[i], q[i], occ, rs)[0]))
            if safe_ok != int(safety.sum()):
                problems.append(f"ellipse re-verification {safe_ok}/{int(safety.sum())}")

            shim = NS(blocked=occ.astype(bool), r_scene=rs)
            lidar = np.asarray(s["lidar_ranges"], np.float64)
            err = max(float(np.max(np.abs(cast_lidar(shim, p[i], rays, max_range) - lidar[i])))
                      for i in range(n))
            if err > 1e-5:
                problems.append(f"lidar mismatch {err:.3e} m")

            # continuity: a point is verified iff every adjacent pair it belongs to passes
            expected = np.ones(n, np.uint8)
            for i in range(1, n):
                if not (safety[i - 1] and safety[i]):
                    expected[i - 1] = 0
                    expected[i] = 0
                    continue
                mid = 0.5 * (q[i - 1] + q[i])
                ok = (midpoint_metric(S[i - 1], q[i - 1], mid) <= rho_max
                      and midpoint_metric(S[i], q[i], mid) <= rho_max)
                pairs += 1
                pairs_ok += int(ok)
                if not ok:
                    expected[i - 1] = 0
                    expected[i] = 0
            mismatch = int(np.sum(expected != cont_flags))
            if mismatch:
                problems.append(f"continuity flags disagree at {mismatch} points")
            tot["points"] += n
            tot["safe"] += int(safety.sum())
            tot["safe_ok"] += safe_ok
            tot["pairs"] += pairs
            tot["pairs_ok"] += pairs_ok
            tot["lidar_max_err"] = max(tot["lidar_max_err"], err)
            tot["deep_bad"] += int(bool(problems))

        if problems or deep:
            rows.append({"file": os.path.basename(path), "scene_id": meta.get("scene_id", ""),
                         "path_id": meta.get("path_id", -1), "n_points": n,
                         "n_safe": int(safety.sum()), "n_safe_reverified": safe_ok,
                         "n_pairs": pairs, "n_pairs_ok": pairs_ok, "deep": deep,
                         "problems": problems})
    summary = {
        "n_files": len(files),
        "n_structurally_checked": tot["struct_checked"],
        "n_structurally_bad": tot["struct_bad"],
        "n_deep_checked": tot["deep_checked"],
        "n_deep_bad": tot["deep_bad"],
        "n_query_points_deep": tot["points"],
        "ellipse_safety_reverified_rate": (tot["safe_ok"] / tot["safe"]) if tot["safe"] else None,
        "continuity_pair_pass_rate": (tot["pairs_ok"] / tot["pairs"]) if tot["pairs"] else None,
        "path_points_in_blocked_cells": tot["path_blocked"],
        "centerline_points_in_blocked_cells": tot["centerline_blocked"],
        "lidar_max_abs_err_m": tot["lidar_max_err"],
    }
    return {"summary": summary, "files": rows}


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _fmt(v, nd: int = 3) -> str:
    return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def write_report(dst: str, tag: str, stats: Dict, checks: Dict, cfg: Dict,
                 files: List[str], timing: Dict, totals: Dict) -> None:
    s, c = stats, checks["summary"]
    lines = [
        f"# PathSafe-Ellipse V3 — trajectory dataset ({tag})",
        "",
        f"- git commit: `{git_commit()}`",
        f"- label_version: `{cfg['project'].get('label_version')}` "
        f"(query point = every resampled A* station)",
        f"- trajectory dir: `{cfg['paths']['trajectory_dir']}`",
        f"- wall clock: {timing['total_s'] / 60:.1f} min "
        f"({timing['per_trajectory_s']:.2f} s / trajectory, {timing['shards']} shard(s))",
        "",
        "## 1. Totals",
        "",
        "| metric | value |",
        "|---|---|",
        f"| trajectories (NPZ) | {s['n_trajectories']} |",
        f"| scenarios | {s['n_scenes']} |",
        f"| map groups | {s['n_map_groups']} |",
        f"| query points | {s['n_query_points']} |",
        f"| points / trajectory (mean, p05, p95) | "
        f"{_fmt(s['points_per_trajectory']['mean'], 1)} / "
        f"{_fmt(s['points_per_trajectory']['p05'], 0)} / "
        f"{_fmt(s['points_per_trajectory']['p95'], 0)} |",
        f"| label_valid rate (all points) | {_fmt(totals['label_valid_rate'], 4)} |",
        f"| safety_verified rate | {_fmt(totals['safety_verified_rate'], 4)} |",
        f"| continuity_verified rate | {_fmt(totals['continuity_verified_rate'], 4)} |",
        f"| inserted continuity points | {totals['n_inserted']} "
        f"({_fmt(totals['inserted_share'], 4)} of all points) |",
        f"| total size | {totals['size_gb']:.2f} GB |",
        "",
        "## 2. Structure / geometry checks (spec 6.1, 6.2)",
        "",
        "| check | result |",
        "|---|---|",
        f"| structurally checked files | {c['n_structurally_checked']} |",
        f"| files with structural problems | {c['n_structurally_bad']} |",
        f"| deeply checked files (evenly sampled) | {c['n_deep_checked']} |",
        f"| deeply checked files with problems | {c['n_deep_bad']} |",
        f"| A* query points in blocked cells | {c['path_points_in_blocked_cells']} |",
        f"| centre-line points in blocked cells | {c['centerline_points_in_blocked_cells']} |",
        f"| safety_verified ellipses re-verified exactly | "
        f"{_fmt(c['ellipse_safety_reverified_rate'], 4)} |",
        f"| continuity pass rate over adjacent points | "
        f"{_fmt(c['continuity_pair_pass_rate'], 4)} |",
        f"| LiDAR max abs error after recast | {c['lidar_max_abs_err_m']:.2e} m |",
        "",
        "## 3. Distributions (sampled files, spec 6.4 — no clearance metrics)",
        "",
        "| metric | p05 / p50 / p95 |",
        "|---|---|",
        f"| ellipse area [m²] | {_fmt(s['ellipse_area']['p05'], 2)} / "
        f"{_fmt(s['ellipse_area']['p50'], 2)} / {_fmt(s['ellipse_area']['p95'], 2)} |",
        f"| semi-major a [m] | {_fmt(s['semi_major_a']['p05'], 2)} / "
        f"{_fmt(s['semi_major_a']['p50'], 2)} / {_fmt(s['semi_major_a']['p95'], 2)} |",
        f"| semi-minor b [m] | {_fmt(s['semi_minor_b']['p05'], 2)} / "
        f"{_fmt(s['semi_minor_b']['p50'], 2)} / {_fmt(s['semi_minor_b']['p95'], 2)} |",
        f"| centre offset [m] | {_fmt(s['center_offset_norm']['p05'], 2)} / "
        f"{_fmt(s['center_offset_norm']['p50'], 2)} / "
        f"{_fmt(s['center_offset_norm']['p95'], 2)} |",
        f"| LiDAR range [m] | {_fmt(s['lidar_range']['p05'], 2)} / "
        f"{_fmt(s['lidar_range']['p50'], 2)} / {_fmt(s['lidar_range']['p95'], 2)} |",
        f"| solver status | {json.dumps(s['solver_status_counts'])} |",
        "",
        "## 4. Notes",
        "",
        "- one NPZ == one complete trajectory; every per-point array shares one point order",
        "- occupancy is `scene.blocked` (raw obstacles + robot safety inflation), 0 = traversable",
        "- no clearance / patch / dist_m field is stored",
        f"- {len(files)} files, listing: see `index.json`",
    ]
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def row_from_file(path: str) -> Dict[str, object]:
    """Index row reconstructed from a saved NPZ (used when the slot was skipped)."""
    s = load_trajectory(path)
    m = s["meta"]
    return {
        "file": os.path.basename(path), "scene_id": m["scene_id"],
        "map_group": m["map_group"], "source_map_id": m.get("source_map_id", ""),
        "path_id": int(m["path_id"]), "n_points": int(len(s["path_xy"])),
        "n_base_points": int(m.get("n_base_points", len(s["path_xy"]))),
        "n_inserted": int(m.get("n_inserted", 0)),
        "n_label_valid": int(np.asarray(s["label_valid"]).sum()),
        "n_safety": int(np.asarray(s["safety_verified"]).sum()),
        "n_continuity": int(np.asarray(s["continuity_verified"]).sum()),
        "physical_field_m": float(m["physical_field_m"]), "r_scene": float(m["r_scene"]),
        "label_status": str(m.get("label_status", "")),
        "size_bytes": os.path.getsize(path),
    }


# --------------------------------------------------------------------------- #
# Progress bookkeeping
# --------------------------------------------------------------------------- #
def load_progress(paths: List[str]) -> Tuple[set, List[Dict[str, object]]]:
    done: set = set()
    rows: List[Dict[str, object]] = []
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue                      # truncated last line after a kill
                done.add(rec["key"])
                rows.extend(rec.get("rows", []))
    return done, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs",
                                                     "data_generation_v3.yaml"))
    ap.add_argument("--target", type=int, default=10, help="number of trajectory samples")
    ap.add_argument("--paths-per-scene", type=int, default=3)
    ap.add_argument("--n-scenes", type=int, default=0, help="0 = derive from target")
    ap.add_argument("--slot-factor", type=float, default=1.45,
                    help="scene slots per target/paths_per_scene (rejected paths budget)")
    ap.add_argument("--crops-per-map", type=int, default=3)
    ap.add_argument("--shard", type=int, default=0,
                    help="0..num-shards-1; shards run as separate processes in parallel")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--finalize-only", action="store_true",
                    help="skip collection: rebuild index/statistics/report from disk")
    ap.add_argument("--build-cache", action="store_true",
                    help="only pre-parse the source maps into the cache, then exit")
    ap.add_argument("--cache-dir", default=os.path.join(PROJECT_ROOT, "data", "cache_v3"))
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--no-viz", action="store_true")
    ap.add_argument("--viz-limit", type=int, default=2, help="figures per scene slot")
    ap.add_argument("--verify-limit", type=int, default=200, help="deep checks (files)")
    ap.add_argument("--stats-sample", type=int, default=2000)
    ap.add_argument("--report-every", type=float, default=60.0, help="progress log [s]")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    cfg["path"]["paths_per_scene"] = args.paths_per_scene
    traj_dir = os.path.join(cfg["paths"]["trajectory_dir"], args.tag)
    viz_dir = os.path.join(cfg["paths"]["viz_dir"], f"trajectory_{args.tag}")
    report_dir = cfg["paths"]["report_dir"]
    for d in (traj_dir, viz_dir, report_dir):
        os.makedirs(d, exist_ok=True)
    shard_tag = ("" if args.num_shards <= 1
                 else f"progress_shard{args.shard}of{args.num_shards}.jsonl")
    own_progress = os.path.join(traj_dir, shard_tag or "progress.jsonl")
    all_progress = sorted(glob.glob(os.path.join(traj_dir, "progress*.jsonl")))

    records = [r for r in load_catalog_cached(cfg["paths"]["map_dir"], 0.25,
                                              args.cache_dir) if r.parsed_ok]
    if args.build_cache:
        subset = records[args.shard::max(args.num_shards, 1)]
        t0 = time.perf_counter()
        for j, rec in enumerate(subset):
            load_map_cached(cfg["paths"]["map_dir"], rec.rel_path, args.cache_dir)
        print(f"cache shard {args.shard}/{args.num_shards}: parsed {len(subset)}/"
              f"{len(records)} maps in {time.perf_counter() - t0:.1f} s", flush=True)
        return 0
    n_scenes = args.n_scenes or int(np.ceil(args.target / max(args.paths_per_scene, 1)
                                            * args.slot_factor))
    # each shard writes into its own sub-directory: 24 processes replacing files in
    # one 50k-entry directory serialise on the NTFS directory lock
    out_dir = traj_dir if args.num_shards <= 1 else os.path.join(
        traj_dir, f"shard{args.shard:02d}")
    os.makedirs(out_dir, exist_ok=True)
    slots = build_slots(records, n_scenes, args.crops_per_map, args.seed,
                        cfg["scene"]["physical_fields_m"])
    tasks = [{
        "key": f"{s['rec'].file_name}|{s['rec'].rel_path}|{s['crop']}|{s['seed']}|{s['field_m']}",
        "config": os.path.abspath(args.config), "paths_per_scene": args.paths_per_scene,
        "slot": s, "out_dir": out_dir, "viz_dir": viz_dir,
        "viz": not args.no_viz, "viz_limit": args.viz_limit,
        "cache_dir": args.cache_dir,
    } for s in slots]
    if args.num_shards > 1:
        tasks = tasks[args.shard::args.num_shards]
    shard_target = (args.target if args.num_shards <= 1
                    else int(np.ceil(args.target / args.num_shards)))

    rows: List[Dict[str, object]] = []
    total_s = 0.0
    if args.finalize_only:
        _, rows = load_progress(all_progress)
        print(f"finalize-only: {len(rows)} trajectory rows from "
              f"{len(all_progress)} progress files", flush=True)
    else:
        done, rows = load_progress([own_progress])
        pending = [t for t in tasks if t["key"] not in done]
        have = len(rows)
        _EXISTING_SCENES.clear()
        for p in glob.glob(os.path.join(traj_dir, "**", "*.npz"), recursive=True):
            name = os.path.basename(p)
            if ".tmp." in name:
                continue
            _EXISTING_SCENES.add(name.split("_path_")[0])
        print(f"shard {args.shard}/{args.num_shards}: target {shard_target} trajectories "
              f"| {len(tasks)} slots assigned, {len(pending)} pending "
              f"| resume: {len(done)} slots done, {have} trajectories present, "
              f"{len(_EXISTING_SCENES)} scene slots already on disk "
              f"| out {out_dir}", flush=True)
        t0 = time.perf_counter()
        log = open(own_progress, "a", encoding="utf-8")
        n_files = have
        last = t0
        started = time.perf_counter()
        done_before = have

        for task in pending:
            res = collect_slot(task)
            n_files += len(res["rows"])
            rows.extend(res["rows"])
            log.write(json.dumps({"key": res["key"], "scene_id": res["scene_id"],
                                  "errors": res["errors"], "ms": res["ms"],
                                  "rows": res["rows"]}) + "\n")
            log.flush()                                   # resumable log, one handle
            if res["errors"]:
                print(f"  !! {res['scene_id'] or res['key']}: {res['errors'][0]}", flush=True)
            now = time.perf_counter()
            if now - last >= args.report_every:
                elapsed = max(now - started, 1e-9)
                produced = max(n_files - done_before, 1)
                rate_min = produced / elapsed * 60.0
                eta = max(shard_target - n_files, 0) / max(rate_min, 1e-9)
                print(f"[{time.strftime('%H:%M:%S')}] {n_files}/{shard_target} "
                      f"({rate_min:.1f}/min, {produced / elapsed:.2f}/s) "
                      f"ETA {eta:.0f} min | {res['scene_id']} +{len(res['rows'])} "
                      f"({res['ms']:.0f} ms)", flush=True)
                last = now
            if n_files >= shard_target:
                break
        log.close()
        total_s = time.perf_counter() - t0
        print(f"shard {args.shard}/{args.num_shards} done: {n_files} trajectories "
              f"({total_s / 60:.1f} min, "
              f"{n_files / max(total_s, 1e-9) * 60:.1f}/min)", flush=True)
        if args.num_shards > 1:
            return 0

    # shards write into sub-directories; the dataset layout is flat, so move them
    nested = [p for p in glob.glob(os.path.join(traj_dir, "**", "*.npz"), recursive=True)
              if os.path.dirname(os.path.abspath(p)) != os.path.abspath(traj_dir)
              and ".tmp." not in os.path.basename(p)]
    for p in nested:
        os.replace(p, os.path.join(traj_dir, os.path.basename(p)))
    for d in glob.glob(os.path.join(traj_dir, "shard*")):
        try:
            os.rmdir(d)
        except OSError:
            pass
    if nested:
        print(f"flattened {len(nested)} files from shard sub-directories", flush=True)
    saved = sorted(p for p in glob.glob(os.path.join(traj_dir, "*.npz"))
                   if ".tmp." not in os.path.basename(p))
    known = {r["file"] for r in rows}
    extra = [p for p in saved if os.path.basename(p) not in known]
    if extra:
        print(f"recovering index rows for {len(extra)} files missing from the "
              f"progress logs ...", flush=True)
        rows.extend(row_from_file(p) for p in extra)
    uniq, seen = [], set()
    for r in rows:
        if r["file"] in seen:
            continue
        seen.add(r["file"])
        uniq.append(r)
    rows = uniq
    print(f"\ncollected {len(rows)} trajectory rows from {len(saved)} NPZ files "
          f"in {total_s / 60:.1f} min", flush=True)
    if not saved:
        return 1

    # ---------------- verification, statistics, index, report
    print("verifying (structural: all files, geometry: sample) ...", flush=True)
    checks = verify_dataset(saved, cfg, deep_limit=args.verify_limit)
    sample = saved if len(saved) <= args.stats_sample else [
        saved[i] for i in np.linspace(0, len(saved) - 1, args.stats_sample).astype(int)]
    print(f"statistics on {len(sample)} sampled files ...", flush=True)
    stats = summarise_files(sample)
    stats["n_files"] = len(saved)
    stats["n_trajectories"] = len(rows) or len(saved)
    stats["n_query_points"] = int(sum(int(r["n_points"]) for r in rows)) \
        if rows else stats["n_query_points"]

    totals = {
        "n_points": int(sum(int(r["n_points"]) for r in rows)),
        "n_inserted": int(sum(int(r.get("n_inserted", 0)) for r in rows)),
        "n_label_valid": int(sum(int(r["n_label_valid"]) for r in rows)),
        "n_safety": int(sum(int(r["n_safety"]) for r in rows)),
        "n_continuity": int(sum(int(r["n_continuity"]) for r in rows)),
        "size_gb": sum(int(r.get("size_bytes", 0)) for r in rows) / 2 ** 30,
    }
    npts = max(totals["n_points"], 1)
    totals["label_valid_rate"] = totals["n_label_valid"] / npts
    totals["safety_verified_rate"] = totals["n_safety"] / npts
    totals["continuity_verified_rate"] = totals["n_continuity"] / npts
    totals["inserted_share"] = totals["n_inserted"] / npts
    stats["n_scenes"] = len({r["scene_id"] for r in rows})
    stats["n_map_groups"] = len({r["map_group"] for r in rows})
    stats["points_per_trajectory"] = (
        {"mean": float(np.mean([int(r["n_points"]) for r in rows])),
         "p05": float(np.percentile([int(r["n_points"]) for r in rows], 5)),
         "p50": float(np.median([int(r["n_points"]) for r in rows])),
         "p95": float(np.percentile([int(r["n_points"]) for r in rows], 95)),
         "min": float(min(int(r["n_points"]) for r in rows)),
         "max": float(max(int(r["n_points"]) for r in rows))} if rows else None)

    index = write_index(traj_dir, rows)
    timing = {"total_s": total_s, "per_trajectory_s": total_s / max(len(saved), 1),
              "shards": args.num_shards}
    payload = {"tag": args.tag, "label_version": cfg["project"].get("label_version"),
               "git_commit": git_commit(), "config": args.config,
               "trajectory_dir": traj_dir, "index": index, "timing": timing,
               "totals": totals, "statistics": stats, "checks": checks,
               "by_scene_rows": len(rows)}
    summary_path = os.path.join(report_dir, f"trajectory_dataset_{args.tag}.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=float)
    report_path = os.path.join(report_dir, f"trajectory_acceptance_{args.tag}.md")
    write_report(report_path, args.tag, stats, checks, cfg, saved, timing, totals)

    c = checks["summary"]
    print(f"\n=== {args.tag}: {len(saved)} trajectories, {totals['n_points']} query points ===")
    print(f"label_valid {totals['label_valid_rate']:.3f}  "
          f"safety {totals['safety_verified_rate']:.3f}  "
          f"continuity {totals['continuity_verified_rate']:.3f}  "
          f"size {totals['size_gb']:.2f} GB")
    print(f"structural bad {c['n_structurally_bad']}/{c['n_structurally_checked']} | "
          f"deep bad {c['n_deep_bad']}/{c['n_deep_checked']} | "
          f"ellipse reverified {c['ellipse_safety_reverified_rate']} | "
          f"continuity pairs {c['continuity_pair_pass_rate']} | "
          f"lidar err {c['lidar_max_abs_err_m']:.1e} m")
    print(f"npz     -> {traj_dir}\nreport  -> {report_path}\nsummary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
