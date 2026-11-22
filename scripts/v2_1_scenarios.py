"""V2.1 stage 1: five canonical geometry scenarios, V2 vs V2.1.

Scenarios (spec 6.2), each with a fixed start/goal and a synthetic but exactly
known occupancy layout (256x256 @ 0.25 m/cell, 64 m field):

1. ``wall_one_side``  a long wall with open space on the other side
2. ``narrow_corridor`` two obstacles forming a narrow channel
3. ``s_channel``      an S-shaped channel
4. ``right_angle``    an L-shaped (right angle) channel
5. ``u_loop``         a U-shaped channel whose two legs run close to each other

For every scenario the same A* path is optimised by V2 (legacy weighted) and
V2.1 (route-constrained clearance); we report occupancy/A*/Q, per-station
offsets, clearance profiles, ellipse sequences and a *paired* comparison at
matched normalised arc-length positions (the two methods produce different
station counts, so raw index pairing would be misleading).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from pathsafe_ellipse.v2.centerline import build_path  # noqa: E402
from pathsafe_ellipse.v2.ellipse import make_label  # noqa: E402
from pathsafe_ellipse.v2.pipeline import anchor_indices  # noqa: E402
from pathsafe_ellipse.v2.scene import build_scene  # noqa: E402

SIZE = 256
FIELD_M = 64.0          # r_scene = 0.25 m/cell
RES = 0.25


# --------------------------------------------------------------------------- #
# Synthetic layouts
# --------------------------------------------------------------------------- #
def _corridor_mask(polyline_cells, half_width_cells: float) -> np.ndarray:
    """Boolean mask of cells within ``half_width_cells`` of a polyline."""
    ys, xs = np.mgrid[0:SIZE, 0:SIZE].astype(float)
    best = np.full((SIZE, SIZE), np.inf)
    pts = np.asarray(polyline_cells, dtype=float)
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        denom = max(float(ab @ ab), 1e-9)
        t = np.clip(((xs - a[0]) * ab[0] + (ys - a[1]) * ab[1]) / denom, 0.0, 1.0)
        d = np.hypot(xs - (a[0] + t * ab[0]), ys - (a[1] + t * ab[1]))
        best = np.minimum(best, d)
    return best <= half_width_cells


def _rect(occ: np.ndarray, x0, x1, y0, y1) -> None:
    occ[y0:y1, x0:x1] = 1


def build_layout(name: str):
    """Return ``(occupancy uint8, start_xy_m, goal_xy_m)`` for one scenario."""
    occ = np.zeros((SIZE, SIZE), dtype=np.uint8)
    if name == "wall_one_side":
        # long wall on the left, open field to the right
        _rect(occ, 20, 26, 10, 246)
        start = np.array([31.0, 232.0]) * RES     # right next to the wall
        goal = np.array([31.0, 24.0]) * RES
    elif name == "narrow_corridor":
        # channel of 5 cells (1.25 m) between two long blocks
        free = _corridor_mask([(124, 246), (124, 10)], half_width_cells=2.5)
        occ[~free] = 1
        _rect(occ, 100, 122, 0, 256)              # left block
        _rect(occ, 127, 150, 0, 256)              # right block
        _rect(occ, 120, 126, 60, 200)             # bump: the shortest path hugs the right side
        free2 = _corridor_mask([(124, 246), (124, 10)], half_width_cells=2.5)
        occ[free2] = 0
        start = np.array([124.0, 240.0]) * RES
        goal = np.array([124.0, 16.0]) * RES
    elif name == "s_channel":
        poly = [(60, 240), (60, 170), (190, 170), (190, 90), (60, 90), (60, 20)]
        free = _corridor_mask(poly, half_width_cells=2.5)
        occ[~free] = 1
        start = np.array([60.0, 236.0]) * RES
        goal = np.array([60.0, 24.0]) * RES
    elif name == "right_angle":
        poly = [(70, 240), (70, 70), (240, 70)]
        free = _corridor_mask(poly, half_width_cells=2.5)
        occ[~free] = 1
        start = np.array([70.0, 236.0]) * RES
        goal = np.array([236.0, 70.0]) * RES
    elif name == "u_loop":
        # U channel: two legs 8 cells apart, the return leg is a different
        # section of the same route (exercises the non-local safety radius)
        poly = [(80, 230), (80, 60), (150, 60), (150, 230)]
        free = _corridor_mask(poly, half_width_cells=2.5)
        occ[~free] = 1
        _rect(occ, 84, 146, 64, 226)              # island between the legs
        free2 = _corridor_mask(poly, half_width_cells=2.5)
        occ[free2] = 0
        start = np.array([80.0, 226.0]) * RES
        goal = np.array([150.0, 226.0]) * RES
    else:
        raise ValueError(name)
    # a one-cell blocked border keeps the path inside the scene
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = 1
    return occ, start, goal


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for k, v in list(cfg["paths"].items()):
        if isinstance(v, str) and v.startswith("./"):
            cfg["paths"][k] = os.path.normpath(os.path.join(PROJECT_ROOT, v))
    return cfg


def make_scene(occ: np.ndarray, cfg: dict, name: str):
    return build_scene(occ, RES, (0, 0), SIZE, SIZE, FIELD_M,
                       float(cfg["robot"]["radius_m"]), float(cfg["robot"]["safety_margin_m"]),
                       int(cfg["robot"]["edt_upsample"]), name, f"synthetic_{name}")


def normalised_arc(q: np.ndarray) -> np.ndarray:
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])
    return s / max(s[-1], 1e-9)


def sample_at_normalised(values: np.ndarray, s_norm: np.ndarray, positions: np.ndarray) -> np.ndarray:
    return np.interp(positions, s_norm, values)


def label_stats(scene, path, cfg, boundary):
    anchors = anchor_indices(scene, path, cfg["anchor"])
    labels = [make_label(scene, path.q_xy[i], boundary, cfg) for i in anchors]
    areas = np.array([l.area for l in labels]) if labels else np.zeros(1)
    s_norm_q = normalised_arc(path.q_xy)
    s_anchor = s_norm_q[np.asarray(anchors)]
    return anchors, labels, areas, s_anchor


def run_method(scene, start, goal, cfg, boundary):
    t0 = time.perf_counter()
    path = build_path(scene, start, goal, cfg)
    dt = (time.perf_counter() - t0) * 1e3
    anchors, labels, areas, s_anchor = label_stats(scene, path, cfg, boundary)
    return {
        "path": path, "anchors": anchors, "labels": labels, "areas": areas,
        "s_anchor": s_anchor, "wall_ms": dt,
        "chk": dict(path.centerline_checks),
        "method": path.centerline_method,
        "status": path.status,
    }


POS = np.linspace(0.05, 0.95, 10)


def paired_row(scene, res):
    """Paired comparison at matched normalised arc positions."""
    path, areas, s_anchor = res["path"], res["areas"], res["s_anchor"]
    s_norm = normalised_arc(path.q_xy)
    clr = sample_at_normalised(path.clearance_q, s_norm, POS)
    off = sample_at_normalised(np.linalg.norm(path.q_xy - path.p_xy, axis=1), s_norm, POS)
    area = np.interp(POS, s_anchor, areas) if len(areas) == len(s_anchor) and len(areas) > 1 else np.zeros_like(POS)
    return {"s": POS, "clearance": clr, "offset": off, "area": area}


def draw(ax, scene, res, title, show_ellipses=True):
    ext = [0, SIZE * RES, SIZE * RES, 0]
    ax.imshow(scene.blocked, cmap="gray_r", alpha=0.6, interpolation="nearest", extent=ext)
    p, q = res["path"].p_xy, res["path"].q_xy
    ax.plot(p[:, 0], p[:, 1], "-", lw=1.4, color="tab:orange", label="A* P")
    ax.plot(q[:, 0], q[:, 1], "-", lw=1.5, color="tab:blue", label="centreline Q")
    for k in range(0, len(p), max(1, len(p) // 12)):
        ax.annotate("", xy=(q[k, 0], q[k, 1]), xytext=(p[k, 0], p[k, 1]),
                    arrowprops=dict(arrowstyle="-", color="tab:purple", lw=0.7, alpha=0.85))
    if show_ellipses:
        for lab in res["labels"]:
            if lab.safety_verified:
                ax.add_patch(Ellipse((lab.q[0], lab.q[1]), 2 * lab.ab[0], 2 * lab.ab[1],
                                     angle=np.degrees(lab.angle), fill=False,
                                     ec="tab:green", lw=0.8, alpha=0.75))
    ax.plot(p[0, 0], p[0, 1], "o", ms=6, color="lime", mec="k")
    ax.plot(p[-1, 0], p[-1, 1], "X", ms=8, color="red", mec="k")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-v2", default=os.path.join(PROJECT_ROOT, "configs", "data_generation_v2.yaml"))
    ap.add_argument("--config-v21", default=os.path.join(PROJECT_ROOT, "configs", "data_generation_v2_1.yaml"))
    ap.add_argument("--r-max", type=float, default=None, help="override R_max for V2.1")
    args = ap.parse_args()

    cfg_v2 = load_cfg(args.config_v2)
    cfg_v21 = load_cfg(args.config_v21)
    if args.r_max is not None:
        cfg_v21["centerline"]["r_max_m"] = float(args.r_max)
    viz = os.path.join(cfg_v21["paths"]["viz_dir"], "scenarios")
    rep = cfg_v21["paths"]["report_dir"]
    os.makedirs(viz, exist_ok=True)
    os.makedirs(rep, exist_ok=True)

    names = ["wall_one_side", "narrow_corridor", "s_channel", "right_angle", "u_loop"]
    out = {}
    for name in names:
        occ, start, goal = build_layout(name)
        scene = make_scene(occ, cfg_v21, name)
        if not bool(scene.is_free(start[None, :])[0]) or not bool(scene.is_free(goal[None, :])[0]):
            print(f"  !! {name}: start/goal not free, skipping")
            continue
        boundary = scene.obstacle_boundary_cells()
        r2 = run_method(scene, start, goal, cfg_v2, boundary)
        r21 = run_method(scene, start, goal, cfg_v21, boundary)
        if r2["path"] is None or r21["path"] is None:
            print(f"  !! {name}: no A* path")
            continue
        pr2, pr21 = paired_row(scene, r2), paired_row(scene, r21)

        row = {
            "scenario": name,
            "path_length_m": float(np.sum(np.linalg.norm(np.diff(r2["path"].p_xy, axis=0), axis=1))),
            "n_stations": int(len(r2["path"].p_xy)),
            "v2": {"status": r2["status"], "chk": r2["chk"],
                   "label_status": r2["path"].label_status,
                   "accepted": bool(r2["path"].accepted),
                   "solver_success": bool(r2["path"].solver_success),
                   "area_p50": float(np.median(r2["areas"])),
                   "area_min": float(r2["areas"].min()), "n_anchors": len(r2["anchors"]),
                   "n_lt5": int((r2["areas"] < 5).sum()), "wall_ms": r2["wall_ms"],
                   "centreline_ms": r2["path"].centerline_solver_time_ms,
                   "clearance_q_mean": float(np.mean(r2["path"].clearance_q)),
                   "gain": float(r2["chk"]["clearance_gain_mean"]),
                   "offset_max": float(r2["chk"]["max_offset_m"]),
                   "offset_p95": float(r2["chk"]["offset_p95_m"]),
                   "homotopy": bool(r2["chk"]["homotopy_verified"]),
                   "paired_clearance": pr2["clearance"].tolist(),
                   "paired_area": pr2["area"].tolist()},
            "v21": {"status": r21["status"], "chk": r21["chk"],
                    "label_status": r21["path"].label_status,
                    "accepted": bool(r21["path"].accepted),
                    "solver_success": bool(r21["path"].solver_success),
                    "area_p50": float(np.median(r21["areas"])),
                    "area_min": float(r21["areas"].min()), "n_anchors": len(r21["anchors"]),
                    "n_lt5": int((r21["areas"] < 5).sum()), "wall_ms": r21["wall_ms"],
                    "centreline_ms": r21["path"].centerline_solver_time_ms,
                    "clearance_q_mean": float(np.mean(r21["path"].clearance_q)),
                    "gain": float(r21["chk"]["clearance_gain_mean"]),
                    "offset_max": float(r21["chk"]["max_offset_m"]),
                    "offset_p95": float(r21["chk"]["offset_p95_m"]),
                    "homotopy": bool(r21["chk"]["homotopy_verified"]),
                    "at_R_limit": int(r21["chk"]["n_at_offset_limit"]),
                    "paired_clearance": pr21["clearance"].tolist(),
                    "paired_area": pr21["area"].tolist()},
        }
        out[name] = row
        print(f"[{name:15s}] len {row['path_length_m']:5.1f} m | "
              f"V2 gain {row['v2']['gain']:+.2f} area {row['v2']['area_p50']:6.2f} | "
              f"V2.1 gain {row['v21']['gain']:+.2f} area {row['v21']['area_p50']:6.2f} "
              f"(<5m2 {row['v21']['n_lt5']}/{row['v21']['n_anchors']}) atR {row['v21']['at_R_limit']} "
              f"| homo {row['v21']['homotopy']} | {row['v21']['status']:>16s} "
              f"| centreline {row['v21']['centreline_ms']:6.0f} ms")

        # ---------------- figure per scenario
        fig, axes = plt.subplots(2, 3, figsize=(17, 9.6))
        draw(axes[0][0], scene, r2, f"V2: D(q) {row['v2']['clearance_q_mean']:.2f} m, "
                                   f"area p50 {row['v2']['area_p50']:.1f} m2")
        axes[0][0].legend(loc="upper right", fontsize=7)
        draw(axes[0][1], scene, r21, f"V2.1: D(q) {row['v21']['clearance_q_mean']:.2f} m, "
                                    f"area p50 {row['v21']['area_p50']:.1f} m2")
        ax = axes[0][2]
        for tag, res, color in (("V2", r2, "tab:orange"), ("V2.1", r21, "tab:blue")):
            s = normalised_arc(res["path"].q_xy)
            ax.plot(s, res["path"].clearance_q, "-", lw=1.2, color=color, label=f"{tag} D(q)")
        s = normalised_arc(r2["path"].p_xy)
        ax.plot(s, r2["path"].clearance_p, "-", lw=1.0, color="tab:gray", label="A* D(p)")
        ax.axhline(0.30, color="red", ls=":", lw=0.9, label="d_safe")
        ax.set_xlabel("normalised arc length"); ax.set_ylabel("clearance [m]")
        ax.set_title("clearance profile"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
        ax = axes[1][0]
        ax.plot(POS, pr2["offset"], "-o", ms=3, color="tab:orange", label="V2 offset")
        ax.plot(POS, pr21["offset"], "-o", ms=3, color="tab:blue", label="V2.1 offset")
        ax.axhline(float(cfg_v21["centerline"]["r_max_m"]), color="k", ls="--", lw=0.8, label="R_max")
        ax.set_xlabel("normalised arc length"); ax.set_ylabel("|q-p| [m]")
        ax.set_title("offset at matched arc positions"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
        ax = axes[1][1]
        ax.plot(POS, pr2["area"], "-o", ms=3, color="tab:orange", label="V2 area")
        ax.plot(POS, pr21["area"], "-o", ms=3, color="tab:blue", label="V2.1 area")
        ax.set_xlabel("normalised arc length"); ax.set_ylabel("ellipse area [m2]")
        ax.set_title("label area at matched arc positions"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
        ax = axes[1][2]; ax.axis("off")
        rows = [("metric", "V2", "V2.1"),
                ("D(q) mean [m]", f"{row['v2']['clearance_q_mean']:.2f}", f"{row['v21']['clearance_q_mean']:.2f}"),
                ("gain [m]", f"{row['v2']['gain']:+.2f}", f"{row['v21']['gain']:+.2f}"),
                ("max offset [m]", f"{row['v2']['offset_max']:.2f}", f"{row['v21']['offset_max']:.2f}"),
                ("p95 offset [m]", f"{row['v2']['offset_p95']:.2f}", f"{row['v21']['offset_p95']:.2f}"),
                ("homotopy", str(row['v2']['homotopy']), str(row['v21']['homotopy'])),
                ("anchors", str(row['v2']['n_anchors']), str(row['v21']['n_anchors'])),
                ("area p50 [m2]", f"{row['v2']['area_p50']:.1f}", f"{row['v21']['area_p50']:.1f}"),
                ("area min [m2]", f"{row['v2']['area_min']:.2f}", f"{row['v21']['area_min']:.2f}"),
                ("anchors <5 m2", str(row['v2']['n_lt5']), str(row['v21']['n_lt5'])),
                ("status", row['v2']['status'][:14], row['v21']['status'][:14])]
        t = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
        t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.25)
        ax.set_title("V2 vs V2.1", fontsize=9)
        fig.suptitle(f"scenario: {name}  (path {row['path_length_m']:.1f} m, "
                     f"{row['n_stations']} stations, r_scene {RES} m)")
        fig.tight_layout()
        fig.savefig(os.path.join(viz, f"{name}.png"), dpi=110)
        plt.close(fig)

    # ---------------- summary figure + JSON
    if out:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
        labels = list(out.keys())
        x = np.arange(len(labels))
        w = 0.38
        axes[0].bar(x - w / 2, [out[k]["v2"]["gain"] for k in labels], w, label="V2")
        axes[0].bar(x + w / 2, [out[k]["v21"]["gain"] for k in labels], w, label="V2.1")
        axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
        axes[0].set_ylabel("clearance gain [m]"); axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].bar(x - w / 2, [out[k]["v2"]["area_p50"] for k in labels], w, label="V2")
        axes[1].bar(x + w / 2, [out[k]["v21"]["area_p50"] for k in labels], w, label="V2.1")
        axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
        axes[1].set_ylabel("area p50 [m2]"); axes[1].legend(); axes[1].grid(alpha=0.3)
        axes[2].bar(x - w / 2, [out[k]["v2"]["centreline_ms"] for k in labels], w, label="V2")
        axes[2].bar(x + w / 2, [out[k]["v21"]["centreline_ms"] for k in labels], w, label="V2.1")
        axes[2].set_xticks(x); axes[2].set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
        axes[2].set_ylabel("centreline solve [ms]"); axes[2].set_yscale("log")
        axes[2].legend(); axes[2].grid(alpha=0.3)
        fig.suptitle("V2.1 stage 1 - five canonical scenarios")
        fig.tight_layout()
        fig.savefig(os.path.join(viz, "_summary.png"), dpi=115)
        plt.close(fig)

    path_json = os.path.join(rep, f"scenarios_r{float(args.r_max or cfg_v21['centerline']['r_max_m']):.1f}.json")
    with open(path_json, "w", encoding="utf-8") as fh:
        json.dump({"scenarios": out, "r_max_m": float(cfg_v21["centerline"]["r_max_m"]),
                   "positions": POS.tolist(), "r_scene": RES}, fh, indent=2)
    print(f"-> {path_json}")
    print(f"figures -> {viz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
