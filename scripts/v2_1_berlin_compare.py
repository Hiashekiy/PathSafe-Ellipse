"""V2.1 experiment 1 + 3: Berlin case comparison and R_max sensitivity.

Experiment 1 (spec 6.1): same scene, same A* path -> compare the V2 centreline
with the V2.1 route-constrained centreline; plot blocked map, P, both Q, the
per-station offset vectors, the clearance profiles and both ellipse sequences.

Experiment 3 (spec 6.3): R_max in {2, 3, 4} m with everything else fixed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from pathsafe_ellipse.map_io import build_map_catalog, load_moving_ai_map  # noqa: E402
from pathsafe_ellipse.v2.centerline import build_path  # noqa: E402
from pathsafe_ellipse.v2.ellipse import make_label  # noqa: E402
from pathsafe_ellipse.v2.pipeline import anchor_indices, sample_start_goal  # noqa: E402
from pathsafe_ellipse.v2.scene import build_scene  # noqa: E402


def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for k in ("map_dir", "report_dir", "viz_dir", "v2_root", "scene_dir", "label_dir"):
        p = cfg["paths"][k]
        if not os.path.isabs(p):
            cfg["paths"][k] = os.path.normpath(os.path.join(PROJECT_ROOT, p))
    return cfg


def ellipses_for(scene, path, cfg, boundary):
    anchors = anchor_indices(scene, path, cfg["anchor"])
    labels = [make_label(scene, path.q_xy[i], boundary, cfg) for i in anchors]
    return anchors, labels


def stats(path, labels, anchors):
    areas = np.array([l.area for l in labels])
    return {
        "n_anchors": int(len(anchors)),
        "area_p50": float(np.median(areas)) if len(areas) else 0.0,
        "area_min": float(areas.min()) if len(areas) else 0.0,
        "area_mean": float(areas.mean()) if len(areas) else 0.0,
        "n_lt5": int((areas < 5.0).sum()),
        "all_verified": bool(all(l.safety_verified for l in labels)),
        **{k: v for k, v in path.centerline_checks.items()},
        "status": path.status,
        "method": path.centerline_method,
        "solver_iters": path.centerline_solver_iterations,
        "solver_ms": round(path.centerline_solver_time_ms, 1),
        "clearance_gain": round(path.centerline_clearance_gain, 3),
        "length_ratio": round(path.centerline_length_ratio, 4),
    }


def draw(ax, scene, path, labels, anchors, title, show_offsets=True, every_offset=3):
    ext = [0, scene.size * scene.r_scene, scene.size * scene.r_scene, 0]
    ax.imshow(scene.blocked, cmap="gray_r", alpha=0.55, interpolation="nearest", extent=ext)
    p, q = path.p_xy, path.q_xy
    ax.plot(p[:, 0], p[:, 1], "-", lw=1.3, color="tab:orange", label="A* P")
    ax.plot(q[:, 0], q[:, 1], "-", lw=1.4, color="tab:blue", label="centreline Q")
    if show_offsets:
        for k in range(0, len(p), every_offset):
            ax.annotate("", xy=(q[k, 0], q[k, 1]), xytext=(p[k, 0], p[k, 1]),
                        arrowprops=dict(arrowstyle="-", color="tab:purple", lw=0.6, alpha=0.8))
        ax.plot(p[::every_offset, 0], p[::every_offset, 1], ".", ms=3, color="tab:red",
                label="A* station p_i")
        ax.plot(q[::every_offset, 0], q[::every_offset, 1], ".", ms=3, color="tab:blue",
                label="centreline q_i")
    for lab in labels:
        if not lab.safety_verified:
            continue
        ax.add_patch(Ellipse((lab.q[0], lab.q[1]), 2 * lab.ab[0], 2 * lab.ab[1],
                             angle=np.degrees(lab.angle), fill=False, ec="tab:green",
                             lw=0.9, alpha=0.85))
    ax.plot(p[0, 0], p[0, 1], "o", ms=6, color="lime", mec="k")
    ax.plot(p[-1, 0], p[-1, 1], "X", ms=8, color="red", mec="k")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[3], ext[2])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-v2", default=os.path.join(PROJECT_ROOT, "configs", "data_generation_v2.yaml"))
    ap.add_argument("--config-v21", default=os.path.join(PROJECT_ROOT, "configs", "data_generation_v2_1.yaml"))
    ap.add_argument("--map", default="Berlin_2_512_part_0_0.map")
    ap.add_argument("--crop", default="247,209")
    ap.add_argument("--field", type=float, default=48.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg_v2 = load_cfg(args.config_v2)
    cfg_v21 = load_cfg(args.config_v21)
    for cfg in (cfg_v2, cfg_v21):
        os.makedirs(cfg["paths"]["report_dir"], exist_ok=True)
        os.makedirs(cfg["paths"]["viz_dir"], exist_ok=True)

    rec = next(r for r in build_map_catalog(cfg_v21["paths"]["map_dir"], 0.25)
               if r.parsed_ok and r.file_name == args.map)
    src, meta = load_moving_ai_map(os.path.join(cfg_v21["paths"]["map_dir"], rec.rel_path))
    r0 = float(meta["resolution"] or 0.25)
    crop_px = int(round(args.field / r0))
    x0, y0 = (int(v) for v in args.crop.split(","))
    scene = build_scene(src, r0, (x0, y0), crop_px, int(cfg_v21["scene"]["size"]), args.field,
                        float(cfg_v21["robot"]["radius_m"]),
                        float(cfg_v21["robot"]["safety_margin_m"]),
                        int(cfg_v21["robot"]["edt_upsample"]), "berlin_case", rec.file_name)
    boundary = scene.obstacle_boundary_cells()

    rng = np.random.default_rng(args.seed)
    pair = sample_start_goal(scene, rng, cfg_v21, [])
    print(f"scene {rec.file_name} crop=({x0},{y0}) FOV={args.field} r_scene={scene.r_scene:.4f}")
    print(f"start={pair[0].round(2)} goal={pair[1].round(2)}")

    out = {}
    paths = {}
    for tag, cfg in (("v2", cfg_v2), ("v2_1", cfg_v21)):
        path = build_path(scene, pair[0], pair[1], cfg)
        anchors, labels = ellipses_for(scene, path, cfg, boundary)
        paths[tag] = (path, anchors, labels)
        out[tag] = stats(path, labels, anchors)
        print(f"[{tag:4s}] {out[tag]['status']:>28s} | D(q) mean {out[tag]['clearance_q_mean']:.2f} "
              f"min {out[tag]['clearance_q_min']:.2f} | gain {out[tag]['clearance_gain']:+.2f} m | "
              f"offset max {out[tag]['max_offset_m']:.2f} p95 {out[tag]['offset_p95_m']:.2f} | "
              f"len x{out[tag]['length_ratio']:.3f} | homotopy {out[tag]['homotopy_verified']} "
              f"({out[tag]['homotopy_states']} states) | area p50 {out[tag]['area_p50']:.2f} "
              f"min {out[tag]['area_min']:.2f} | <5m2 {out[tag]['n_lt5']}/{out[tag]['n_anchors']} "
              f"| {out[tag]['solver_ms']:.0f} ms")

    # ---- figure: side-by-side centreline + ellipses + clearance profile
    fig = plt.figure(figsize=(19, 10))
    ax1 = fig.add_subplot(2, 3, 1)
    draw(ax1, scene, paths["v2"][0], paths["v2"][2], paths["v2"][1],
         f"V2 (legacy weighted): D(q) mean {out['v2']['clearance_q_mean']:.2f} m, "
         f"area p50 {out['v2']['area_p50']:.2f} m2")
    ax1.legend(loc="upper right", fontsize=7)
    ax2 = fig.add_subplot(2, 3, 2)
    draw(ax2, scene, paths["v2_1"][0], paths["v2_1"][2], paths["v2_1"][1],
         f"V2.1 (route-constrained clearance): D(q) mean "
         f"{out['v2_1']['clearance_q_mean']:.2f} m, area p50 {out['v2_1']['area_p50']:.2f} m2")
    ax3 = fig.add_subplot(2, 3, 3)
    for tag, color in (("v2", "tab:orange"), ("v2_1", "tab:blue")):
        path = paths[tag][0]
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path.q_xy, axis=0), axis=1))])
        ax3.plot(s, path.clearance_q, "-o", ms=2, lw=1.0, color=color, label=f"{tag} D(q)")
    path = paths["v2"][0]
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path.p_xy, axis=0), axis=1))])
    ax3.plot(s, path.clearance_p, "-", lw=1.2, color="tab:gray", label="A* D(p)")
    ax3.axhline(out["v2_1"].get("d_safe", 0.30), color="red", ls=":", lw=1.0, label="d_safe")
    ax3.set_xlabel("arc length along Q [m]"); ax3.set_ylabel("clearance [m]")
    ax3.set_title("clearance profile"); ax3.legend(fontsize=7); ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(2, 3, 4)
    for tag, color in (("v2", "tab:orange"), ("v2_1", "tab:blue")):
        path = paths[tag][0]
        off = np.linalg.norm(path.q_xy - path.p_xy, axis=1)
        ax4.plot(np.arange(len(off)), off, "-o", ms=2.5, lw=1.0, color=color, label=f"{tag} offset")
    ax4.axhline(4.0, color="k", ls="--", lw=0.8, label="R_max = 4 m")
    ax4.set_xlabel("station index"); ax4.set_ylabel("|q_i - p_i| [m]")
    ax4.set_title("per-station offset"); ax4.legend(fontsize=7); ax4.grid(alpha=0.3)

    ax5 = fig.add_subplot(2, 3, 5)
    for tag, color in (("v2", "tab:orange"), ("v2_1", "tab:blue")):
        areas = [l.area for l in paths[tag][2]]
        ax5.plot(np.arange(len(areas)), areas, "-o", ms=2.5, lw=1.0, color=color,
                 label=f"{tag} ellipse area")
    ax5.set_xlabel("anchor index"); ax5.set_ylabel("area [m2]")
    ax5.set_title("labelled ellipse area"); ax5.legend(fontsize=7); ax5.grid(alpha=0.3)

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    rows = [("metric", "V2", "V2.1")]
    for key, lab in (("clearance_p_mean", "D(p) mean [m]"), ("clearance_q_mean", "D(q) mean [m]"),
                     ("clearance_q_min", "D(q) min [m]"), ("clearance_gain", "gain [m]"),
                     ("max_offset_m", "max offset [m]"), ("offset_p95_m", "offset p95 [m]"),
                     ("length_ratio", "length ratio"), ("homotopy_verified", "homotopy"),
                     ("n_at_offset_limit", "stations at R limit"),
                     ("area_p50", "area p50 [m2]"), ("area_min", "area min [m2]"),
                     ("n_lt5", "anchors < 5 m2"), ("solver_ms", "solve [ms]")):
        rows.append((lab, str(round(out["v2"].get(key, out["v2"].get(key, 0)), 3))
                     if not isinstance(out["v2"].get(key), bool) else str(out["v2"][key]),
                     str(round(out["v2_1"].get(key, 0), 3))
                     if not isinstance(out["v2_1"].get(key), bool) else str(out["v2_1"][key])))
    table = ax6.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.3)
    ax6.set_title("V2 vs V2.1", fontsize=10)

    fig.suptitle(f"V2.1 experiment 1 - {rec.file_name} crop=({x0},{y0}) FOV {args.field:.0f} m")
    fig.tight_layout()
    fig_path = os.path.join(cfg_v21["paths"]["viz_dir"], "berlin_case", "v2_vs_v2_1.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, dpi=115)
    plt.close(fig)
    print(f"figure -> {fig_path}")

    # ---- experiment 3: R_max sensitivity (V2.1 only)
    sensitivity = []
    for r_max in (2.0, 3.0, 4.0):
        cfg = json.loads(json.dumps(cfg_v21))
        cfg["centerline"]["r_max_m"] = r_max
        path = build_path(scene, pair[0], pair[1], cfg)
        anchors, labels = ellipses_for(scene, path, cfg, boundary)
        st = stats(path, labels, anchors)
        st["r_max_m"] = r_max
        sensitivity.append(st)
        print(f"[R_max={r_max:.0f} m] D(q) mean {st['clearance_q_mean']:.2f} | gain "
              f"{st['clearance_gain']:+.2f} | offset max {st['max_offset_m']:.2f} "
              f"p95 {st['offset_p95_m']:.2f} | at-limit {st['n_at_offset_limit']}/{len(path.p_xy)} | "
              f"len x{st['length_ratio']:.3f} | homotopy {st['homotopy_verified']} | "
              f"area p50 {st['area_p50']:.2f} | {st['status']}")

    out_path = os.path.join(cfg_v21["paths"]["report_dir"], "centerline_comparison.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"scene": rec.file_name, "crop": [x0, y0], "field_m": args.field,
                   "r_scene": scene.r_scene, "label_version": "v2.1",
                   "comparison": out, "r_max_sensitivity": sensitivity}, fh, indent=2)
    print(f"-> {out_path}")

    # sensitivity figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    xs = [s["r_max_m"] for s in sensitivity]
    axes[0].plot(xs, [s["clearance_q_mean"] for s in sensitivity], "-o", label="D(q) mean")
    axes[0].plot(xs, [s["clearance_p_mean"] for s in sensitivity], "-s", label="D(p) mean")
    axes[0].set_xlabel("R_max [m]"); axes[0].set_ylabel("clearance [m]"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(xs, [s["area_p50"] for s in sensitivity], "-o", label="area p50")
    axes[1].plot(xs, [s["area_min"] for s in sensitivity], "-s", label="area min")
    axes[1].set_xlabel("R_max [m]"); axes[1].set_ylabel("ellipse area [m2]"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].plot(xs, [s["max_offset_m"] for s in sensitivity], "-o", label="max offset")
    axes[2].plot(xs, [s["offset_p95_m"] for s in sensitivity], "-s", label="offset p95")
    axes[2].plot(xs, [s["length_ratio"] for s in sensitivity], "-^", label="length ratio")
    axes[2].set_xlabel("R_max [m]"); axes[2].legend(); axes[2].grid(alpha=0.3)
    fig.suptitle("V2.1 experiment 3 - R_max sensitivity")
    fig.tight_layout()
    fig2 = os.path.join(cfg_v21["paths"]["viz_dir"], "berlin_case", "r_max_sensitivity.png")
    fig.savefig(fig2, dpi=115)
    plt.close(fig)
    print(f"figure -> {fig2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
