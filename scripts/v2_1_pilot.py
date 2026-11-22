"""V2.1 pilot runner (stage 2 / stage 4).

For every selected scene it runs V2 (legacy baseline) and V2.1 on the SAME scene
and start/goal, saves the scene + path + label arrays as NPZ for both variants in
separate directories, and writes a summary with per-path detail and the eight
counters required by the spec (safety and continuity counted independently).

    python scripts/v2_1_pilot.py --n-scenes 20 --paths-per-scene 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from pathsafe_ellipse.map_io import build_map_catalog, load_moving_ai_map, map_group_id  # noqa: E402
from pathsafe_ellipse.v2.pilot import (  # noqa: E402
    load_scene_npz, save_scene_npz, scene_metadata, write_index,
)
from pathsafe_ellipse.v2.pipeline import process_scene  # noqa: E402
from pathsafe_ellipse.v2.scene import build_scene, free_components  # noqa: E402


def load_cfg(path: str) -> Dict:
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for k, v in list(cfg["paths"].items()):
        if isinstance(v, str) and v.startswith("./"):
            cfg["paths"][k] = os.path.normpath(os.path.join(PROJECT_ROOT, v))
    return cfg


def pick_scenes(records, n: int, seed: int) -> List[Dict]:
    """Stratified by map group and by filename scale to avoid near-duplicates."""
    rng = np.random.default_rng(seed)
    by_group: Dict[str, List] = {}
    for rec in records:
        by_group.setdefault(map_group_id(rec.file_name), []).append(rec)
    groups = sorted(by_group)
    rng.shuffle(groups)
    out: List[Dict] = []
    gi = 0
    while len(out) < n and groups:
        g = groups[gi % len(groups)]
        pool = by_group[g]
        rec = pool[int(rng.integers(0, len(pool)))]
        if all(r.file_name != rec.file_name for r in out):
            out.append(rec)
        gi += 1
        if gi > 40 * n:
            break
    return out


def draw(ax, scene, res, title):
    ext = [0, scene.size * scene.r_scene, scene.size * scene.r_scene, 0]
    ax.imshow(scene.blocked, cmap="gray_r", alpha=0.55, interpolation="nearest", extent=ext)
    for rec in res.paths:
        if not rec.get("accepted"):
            continue
        p = np.asarray(rec["p_xy"])
        q = np.asarray(rec["q_xy"])
        ax.plot(p[:, 0], p[:, 1], "-", lw=1.0, color="tab:orange", alpha=0.85)
        ax.plot(q[:, 0], q[:, 1], "-", lw=1.2, color="tab:blue", alpha=0.9)
    lab = res.labels
    if len(lab.get("q_xy", [])):
        for k in range(len(lab["q_xy"])):
            if not int(lab["label_valid"][k]):
                continue
            c = lab["q_xy"][k]
            ab = lab["ellipse_ab"][k]
            ax.add_patch(Ellipse((c[0], c[1]), 2 * ab[0], 2 * ab[1],
                                 angle=np.degrees(float(lab["ellipse_angle"][k])),
                                 fill=False, ec="tab:green", lw=0.7, alpha=0.7))
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def run_variant(scene, cfg, seed, rec, out_dir, prefix, extra):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    res = process_scene(scene, cfg, rng)
    ms_total = (time.perf_counter() - t0) * 1e3
    lab_c, n_c, largest = free_components(scene.free)
    meta = scene_metadata(scene, cfg, seed, rec.rel_path, map_group_id(rec.file_name), n_c)
    meta.update(extra)
    meta["largest_component_cells"] = int(largest)
    dst = save_scene_npz(res, scene, cfg, out_dir, meta, prefix=prefix)
    return res, meta, dst, ms_total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-v2", default=os.path.join(PROJECT_ROOT, "configs", "data_generation_v2.yaml"))
    ap.add_argument("--config-v21", default=os.path.join(PROJECT_ROOT, "configs",
                                                         "data_generation_v2_1.yaml"))
    ap.add_argument("--n-scenes", type=int, default=3)
    ap.add_argument("--paths-per-scene", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--tag", default="stage2")
    ap.add_argument("--pairs", default="all", help="all|v21 (v21 only, for the 20-scene pilot)")
    args = ap.parse_args()

    cfg2 = load_cfg(args.config_v2)
    cfg21 = load_cfg(args.config_v21)
    for cfg in (cfg2, cfg21):
        cfg["path"]["paths_per_scene"] = args.paths_per_scene

    out_root = os.path.join(cfg21["paths"]["v2_root"], f"pilot_{args.tag}")
    dirs = {
        "v21": os.path.join(out_root, "v2_1"),
        "v2": os.path.join(out_root, "v2"),
        "viz": os.path.join(cfg21["paths"]["viz_dir"], f"pilot_{args.tag}"),
        "report": cfg21["paths"]["report_dir"],
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    records = [r for r in build_map_catalog(cfg21["paths"]["map_dir"], 0.25) if r.parsed_ok]
    picks = pick_scenes(records, args.n_scenes, args.seed)
    print(f"selected {len(picks)} scenes from {len(records)} maps "
          f"({len({map_group_id(r.file_name) for r in picks})} map groups)")

    scenes_out: List[Dict] = []
    index_entries: List[Dict] = []
    for i, rec in enumerate(picks):
        src, smeta = load_moving_ai_map(os.path.join(cfg21["paths"]["map_dir"], rec.rel_path))
        r0 = float(smeta["resolution"] or 0.25)
        field_m = float(cfg21["scene"]["physical_fields_m"][i % 3])
        crop_px = int(round(field_m / r0))
        h, w = src.shape
        if crop_px >= h or crop_px >= w:
            print(f"  skip {rec.file_name}: map {w}x{h} < crop {crop_px}")
            continue
        rng = np.random.default_rng(args.seed + i)
        x0 = int(rng.integers(0, w - crop_px))
        y0 = int(rng.integers(0, h - crop_px))
        scene_id = f"{os.path.splitext(rec.file_name)[0]}_c{x0}_{y0}_f{int(field_m)}"
        row: Dict[str, object] = {
            "scene_id": scene_id, "map": rec.file_name, "rel_path": rec.rel_path,
            "map_group": map_group_id(rec.file_name), "crop": [x0, y0],
            "field_m": field_m, "seed": args.seed + i,
        }
        variants = ("v21",) if args.pairs == "v21" else ("v2", "v21")
        for tag in variants:
            cfg = cfg21 if tag == "v21" else cfg2
            scene = build_scene(src, r0, (x0, y0), crop_px, int(cfg["scene"]["size"]), field_m,
                                float(cfg["robot"]["radius_m"]),
                                float(cfg["robot"]["safety_margin_m"]),
                                int(cfg["robot"]["edt_upsample"]), scene_id, rec.file_name)
            try:
                res, meta, dst, ms = run_variant(scene, cfg, args.seed + i, rec, dirs[tag],
                                                 "" if tag == "v21" else "v2_", {"variant": tag})
            except Exception as exc:                     # keep the failure record
                row[tag] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"  !! {scene_id} [{tag}] failed: {type(exc).__name__}: {exc}")
                continue
            st = res.stats
            row[tag] = {k: st[k] for k in (
                "n_paths", "n_paths_accepted", "n_paths_failed", "n_anchors",
                "n_ellipse_safe", "n_continuity_verified", "n_both_verified",
                "n_solver_converged", "n_feasible_not_converged", "n_optimal_safe",
                "n_fallback", "n_failed", "n_inserted", "solve_ms_median", "solve_ms_p95",
                "density_inflated", "label_version", "centerline_method",
                "centerline_clearance_gain_mean")}
            row[tag].update({
                "paths": st["paths"],
                "npz": os.path.basename(dst),
                "wall_ms": round(ms, 1),
                "areas_p50": (float(np.median(res.labels["ellipse_area"]))
                              if st["n_anchors"] else None),
                "areas_p95": (float(np.percentile(res.labels["ellipse_area"], 95))
                              if st["n_anchors"] else None),
                "areas_min": (float(np.min(res.labels["ellipse_area"]))
                              if st["n_anchors"] else None),
                "areas_small_share": (float(np.mean(res.labels["ellipse_area"] < 5.0))
                                      if st["n_anchors"] else None),
            })
            index_entries.append({"scene_id": scene_id, "variant": tag,
                                  "npz": os.path.basename(dst),
                                  "n_paths": st["n_paths"], "n_anchors": st["n_anchors"]})
            if tag == "v2":
                fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
            else:
                axes[2 - 1].clear()
            draw(axes[0] if tag == "v2" else axes[1], scene, res,
                 f"{'V2' if tag == 'v2' else 'V2.1'}: "
                 f"{st['n_anchors']} labels, area p50 "
                 f"{row[tag]['areas_p50'] or 0:.1f} m2")
            if tag == "v2":
                _pending = (fig, axes, scene, row, scene_id)
            else:
                fig, axes, scene_, row_, sid = _pending
                ax = axes[2]
                for r in res.paths:
                    if not r.get("accepted"):
                        continue
                    q = np.asarray(r["q_xy"])
                    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])
                    ax.plot(s / max(s[-1], 1e-9), r["clearance_q"], "-", lw=1.0, color="tab:blue")
                ax.set_xlabel("normalised arc length"); ax.set_ylabel("D(q) [m]")
                ax.set_title("V2.1 clearance along paths"); ax.grid(alpha=0.3)
                fig.suptitle(f"{sid}  FOV {field_m:.0f} m  r_scene {scene_.r_scene:.4f}")
                fig.tight_layout()
                fig.savefig(os.path.join(dirs["viz"], f"{sid}.png"), dpi=110)
                plt.close(fig)
            print(f"  [{i+1}/{len(picks)}] {scene_id} [{tag:3s}] paths {st['n_paths_accepted']}/"
                  f"{st['n_paths']} labels {st['n_anchors']} safe {st['n_ellipse_safe']} "
                  f"cont {st['n_continuity_verified']} both {st['n_both_verified']} "
                  f"| area p50 {row[tag]['areas_p50'] or 0:.2f} | {ms:.0f} ms")
        scenes_out.append(row)

    # ---------------- summary
    def agg(tag: str, key: str):
        vals = [r[tag][key] for r in scenes_out if tag in r and key in r[tag]
                and r[tag][key] is not None and not isinstance(r[tag][key], list)]
        return float(np.mean(vals)) if vals else None

    summary = {
        "tag": args.tag, "seed": args.seed, "n_scenes": len(scenes_out),
        "paths_per_scene": args.paths_per_scene,
        "label_version": cfg21["project"].get("label_version"),
        "git_commit": __import__("pathsafe_ellipse.v2.pilot", fromlist=["git_commit"]).git_commit(),
        "configs": {"v2": args.config_v2, "v2_1": args.config_v21},
        "dirs": dirs, "scenes": scenes_out,
    }
    if args.pairs != "v21":
        summary["paired"] = {
            "mean_gain_v2": agg("v2", "centerline_clearance_gain_mean"),
            "mean_gain_v21": agg("v21", "centerline_clearance_gain_mean"),
            "mean_area_p50_v2": agg("v2", "areas_p50"),
            "mean_area_p50_v21": agg("v21", "areas_p50"),
            "total_labels_v2": int(sum(r["v2"]["n_anchors"] for r in scenes_out if "v2" in r)),
            "total_labels_v21": int(sum(r["v21"]["n_anchors"] for r in scenes_out if "v21" in r)),
            "total_both_verified_v2": int(sum(r["v2"]["n_both_verified"] for r in scenes_out if "v2" in r)),
            "total_both_verified_v21": int(sum(r["v21"]["n_both_verified"] for r in scenes_out if "v21" in r)),
        }
    dst = os.path.join(dirs["report"], f"pilot_{args.tag}_summary.json")
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)
    idx = write_index(out_root, index_entries)
    print(f"\nsummary -> {dst}\nindex   -> {idx}\nnpz     -> {dirs['v21']} / {dirs['v2']}")

    # ---------------- verify the NPZ round-trip
    ok = 0
    for e in index_entries[:4]:
        p = os.path.join(dirs[e["variant"]], e["npz"])
        d = load_scene_npz(p)
        n = int(d["path_id"].shape[0]) if "path_id" in d else 0
        ok += int(n == int(e["n_anchors"]) and "meta" in d)
    print(f"npz round-trip check: {ok}/{min(4, len(index_entries))} files load and match counts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
