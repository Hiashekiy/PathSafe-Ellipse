"""V2.1 stage 3: S-channel and goal-neighbourhood degeneration experiments.

Experiment A (unified positions): evaluate P, V2 and V2.1 at the SAME normalised
arc positions by *regenerating* an ellipse label there, so anchor sampling
differences cannot distort the comparison.

Experiment B (initialisation ablation): A* / +0.5 / +1.0 / +-0.5 / +-1.0 seeds.

Experiment C (multi-resolution ablation): full stations vs safety-preserving
simplification at 32 / 24 knots, isolating objective vs init vs knots vs
interpolation vs projection vs homotopy repair.

Experiment D (goal neighbourhood): s = 0.70/0.80/0.90/0.95 with tangent-limit /
end-ramp / smoothing ablations (endpoints stay fixed).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from pathsafe_ellipse.map_io import build_map_catalog, load_moving_ai_map  # noqa: E402
from pathsafe_ellipse.v2 import centerline as C  # noqa: E402
from pathsafe_ellipse.v2.ellipse import make_label  # noqa: E402
from pathsafe_ellipse.v2.pipeline import sample_start_goal  # noqa: E402
from pathsafe_ellipse.v2.scene import build_scene  # noqa: E402
from v2_1_scenarios import build_layout, load_cfg, make_scene  # noqa: E402

POS = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
GOAL_POS = np.array([0.70, 0.80, 0.90, 0.95])


def norm_arc(x: np.ndarray) -> np.ndarray:
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(x, axis=0), axis=1))])
    return s / max(s[-1], 1e-9)


def at_position(q: np.ndarray, s_norm: float) -> np.ndarray:
    """Point on polyline at normalised arc length (linear interpolation)."""
    s = norm_arc(q)
    k = int(np.searchsorted(s, s_norm, side="right") - 1)
    k = max(0, min(k, len(q) - 2))
    t = (s_norm - s[k]) / max(s[k + 1] - s[k], 1e-12)
    return q[k] + t * (q[k + 1] - q[k]), k, s


def curvature_at(q: np.ndarray, k: int) -> float:
    if k <= 0 or k >= len(q) - 1:
        return 0.0
    t0 = q[k] - q[k - 1]
    t1 = q[k + 1] - q[k]
    n0, n1 = np.linalg.norm(t0), np.linalg.norm(t1)
    if n0 < 1e-9 or n1 < 1e-9:
        return 0.0
    cosang = float(np.clip((t0 / n0) @ (t1 / n1), -1.0, 1.0))
    return float(np.arccos(cosang) / max(0.5 * (n0 + n1), 1e-9))


def evaluate_at_positions(scene, boundary, cfg, path, positions, d_safe):
    """Regenerate a label at each normalised position on the given centre line."""
    out = []
    q = path.q_xy
    for s_norm in positions:
        pt, k, _s = at_position(q, float(s_norm))
        lab = make_label(scene, pt, boundary, cfg)
        e = curvature_at(q, k)
        out.append({
            "s": float(s_norm), "q_xy": pt.tolist(), "curvature": e,
            "clearance": float(lab.clearance), "area": float(lab.area),
            "ab": [float(lab.ab[0]), float(lab.ab[1])], "angle": float(lab.angle),
            "status": lab.status, "safety_verified": bool(lab.safety_verified),
            "solve_ms": float(lab.solve_ms),
        })
    return out


def run_one(scene, pair, cfg, boundary, positions):
    t0 = time.perf_counter()
    path = C.build_path(scene, pair[0], pair[1], cfg)
    ms = (time.perf_counter() - t0) * 1e3
    chk = path.centerline_checks
    rows = evaluate_at_positions(scene, boundary, cfg, path, positions,
                                 float(chk.get("d_safe", 0.30)))
    return {"path": path, "rows": rows, "ms": ms,
            "gain": float(chk.get("clearance_gain_mean", 0.0)),
            "offset_max": float(chk.get("max_offset_m", 0.0)),
            "label_status": path.label_status, "accepted": bool(path.accepted),
            "solver": [path.solver_success, path.solver_message,
                       path.solver_max_violation]}


# --------------------------------------------------------------------------- #
def experiment_a(scene, pair, cfg2, cfg21, boundary):
    p_rows = run_one(scene, pair, {**cfg21, "centerline": {**cfg21["centerline"],
                                                           "r_max_m": 1e-9}}, boundary, POS)
    v2 = run_one(scene, pair, cfg2, boundary, POS)
    v21 = run_one(scene, pair, cfg21, boundary, POS)
    table = []
    for i, s in enumerate(POS):
        table.append({
            "s": float(s),
            "clearance_p": float(np.interp(float(s), norm_arc(v2["path"].p_xy),
                                           scene.distance(v2["path"].p_xy))),
            "clearance_v2": v2["rows"][i]["clearance"],
            "clearance_v21": v21["rows"][i]["clearance"],
            "area_v2": v2["rows"][i]["area"], "area_v21": v21["rows"][i]["area"],
            "offset_v2": float(np.linalg.norm(v2["rows"][i]["q_xy"] - np.asarray(
                at_position(v2["path"].p_xy, float(s))[0]))),
            "offset_v21": float(np.linalg.norm(v21["rows"][i]["q_xy"] - np.asarray(
                at_position(v21["path"].p_xy, float(s))[0]))),
            "curvature_v2": v2["rows"][i]["curvature"],
            "curvature_v21": v21["rows"][i]["curvature"],
            "status_v2": v2["rows"][i]["status"], "status_v21": v21["rows"][i]["status"],
            "ratio": (v21["rows"][i]["area"] / v2["rows"][i]["area"]
                      if v2["rows"][i]["area"] > 1e-9 else None),
        })
    ratios = [t["ratio"] for t in table if t["ratio"]]
    summary = {
        "area_p50_v2": float(np.median([t["area_v2"] for t in table])),
        "area_p50_v21": float(np.median([t["area_v21"] for t in table])),
        "ratio_median": float(np.median(ratios)) if ratios else None,
        "n_worse": int(sum(1 for r in ratios if r < 1.0)),
        "mean_clearance_v2": float(np.mean([t["clearance_v2"] for t in table])),
        "mean_clearance_v21": float(np.mean([t["clearance_v21"] for t in table])),
        "mean_clearance_p": float(np.mean([t["clearance_p"] for t in table])),
    }
    return {"table": table, "summary": summary, "run_v2": {
        "gain": v2["gain"], "area_p50_method": float(np.median(
            [r["area"] for r in v2["rows"]])), "ms": v2["ms"]}, "run_v21": {
        "gain": v21["gain"], "ms": v21["ms"], "label_status": v21["label_status"]}}


def experiment_b(scene, pair, cfg21, boundary):
    cc = cfg21["centerline"]
    variants = {
        "A_astar_only": dict(multi_start_count=1, init_offsets_m=[0.5, 1.0]),
        "B_plus_0.5": dict(multi_start_count=2, init_offsets_m=[0.5]),
        "C_plus_1.0": dict(multi_start_count=2, init_offsets_m=[1.0]),
        "D_pm_0.5": dict(multi_start_count=3, init_offsets_m=[0.5, -0.5]),
        "E_pm_1.0": dict(multi_start_count=3, init_offsets_m=[1.0, -1.0]),
    }
    out = {}
    for name, over in variants.items():
        cfg = json.loads(json.dumps(cfg21))
        cfg["centerline"].update(over)
        r = run_one(scene, pair, cfg, boundary, POS)
        out[name] = {"gain": r["gain"], "ms": r["ms"],
                     "area_p50": float(np.median([x["area"] for x in r["rows"]])),
                     "solver_violation": r["solver"][2],
                     "label_status": r["label_status"]}
    return out


def experiment_c(scene, pair, cfg21, boundary):
    cc = cfg21["centerline"]
    variants = {
        "full_stations": dict(max_optimised_stations=100000),
        "knots_32": dict(max_optimised_stations=32),
        "knots_24": dict(max_optimised_stations=24),
        "knots_16": dict(max_optimised_stations=16),
        "no_projection": dict(max_optimised_stations=32, _skip_projection=True),
        "no_repair": dict(max_optimised_stations=32, _skip_repair=True),
    }
    out = {}
    for name, over in variants.items():
        cfg = json.loads(json.dumps(cfg21))
        cfg["centerline"].update(over)
        r = run_one(scene, pair, cfg, boundary, POS)
        out[name] = {"gain": r["gain"], "ms": r["ms"], "label_status": r["label_status"],
                     "accepted": r["accepted"],
                     "area_p50": float(np.median([x["area"] for x in r["rows"]])),
                     "n_knots": (len(C.simplify_stations(scene, r["path"].p_xy, cfg,
                                                        int(over.get("max_optimised_stations",
                                                                     32))))
                                 if "max_optimised_stations" in over else None)}
    return out


def experiment_d(scene, pair, cfg2, cfg21, boundary):
    v2 = run_one(scene, pair, cfg2, boundary, GOAL_POS)
    v21 = run_one(scene, pair, cfg21, boundary, GOAL_POS)
    cc = cfg21["centerline"]
    variants = {
        "tangent_0.25": dict(station_tangent_limit_ratio=0.25),
        "tangent_0.10": dict(station_tangent_limit_ratio=0.10),
        "tangent_0.50": dict(station_tangent_limit_ratio=0.50),
        "w_smooth_0.05": dict(w_smooth=0.05),
        "w_smooth_0.40": dict(w_smooth=0.40),
        "w_pull_0.005": dict(w_pull=0.005),
    }
    abl = {}
    for name, over in variants.items():
        cfg = json.loads(json.dumps(cfg21))
        cfg["centerline"].update(over)
        r = run_one(scene, pair, cfg, boundary, GOAL_POS)
        abl[name] = {"area": [x["area"] for x in r["rows"]],
                     "gain": r["gain"], "accepted": r["accepted"],
                     "area_p50": float(np.median([x["area"] for x in r["rows"]]))}
    table = []
    for i, s in enumerate(GOAL_POS):
        table.append({
            "s": float(s),
            "clearance_p": float(np.interp(float(s), norm_arc(v2["path"].p_xy),
                                           scene.distance(v2["path"].p_xy))),
            "clearance_v2": v2["rows"][i]["clearance"],
            "clearance_v21": v21["rows"][i]["clearance"],
            "offset_v2": float(np.linalg.norm(v2["rows"][i]["q_xy"] - np.asarray(
                at_position(v2["path"].p_xy, float(s))[0]))),
            "offset_v21": float(np.linalg.norm(v21["rows"][i]["q_xy"] - np.asarray(
                at_position(v21["path"].p_xy, float(s))[0]))),
            "curvature": v21["rows"][i]["curvature"],
            "area_v2": v2["rows"][i]["area"], "area_v21": v21["rows"][i]["area"],
            "tangent_margin_v21": None,
        })
    return {"table": table, "ablation": abl, "v21_gain": v21["gain"], "v2_gain": v2["gain"]}


def fig_experiment(name, exp_a, exp_d, viz_dir, title):
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.6))
    ax = axes[0][0]
    ax.plot(POS, [t["clearance_p"] for t in exp_a["table"]], "-s", ms=3, label="A* P")
    ax.plot(POS, [t["clearance_v2"] for t in exp_a["table"]], "-o", ms=3, label="V2")
    ax.plot(POS, [t["clearance_v21"] for t in exp_a["table"]], "-o", ms=3, label="V2.1")
    ax.set_xlabel("normalised arc length"); ax.set_ylabel("clearance [m]")
    ax.set_title("A: clearance at the SAME positions (labels regenerated)")
    ax.legend(); ax.grid(alpha=0.3)
    ax = axes[0][1]
    ax.plot(POS, [t["area_v2"] for t in exp_a["table"]], "-o", ms=3, label="V2")
    ax.plot(POS, [t["area_v21"] for t in exp_a["table"]], "-o", ms=3, label="V2.1")
    ax.set_xlabel("normalised arc length"); ax.set_ylabel("regenerated label area [m2]")
    ax.set_title(f"A: area, median ratio {exp_a['summary']['ratio_median']:.2f}, "
                 f"{exp_a['summary']['n_worse']}/{len(POS)} positions worse")
    ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1][0]
    ax.bar(range(len(exp_a["table"])), [t["ratio"] or 0 for t in exp_a["table"]])
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_xticks(range(len(POS))); ax.set_xticklabels([f"{s:.2f}" for s in POS], fontsize=7)
    ax.set_ylabel("area ratio V2.1/V2"); ax.set_title("A: per-position ratio")
    ax.grid(alpha=0.3)
    ax = axes[1][1]
    ax.plot(GOAL_POS, [t["clearance_v2"] for t in exp_d["table"]], "-o", ms=3, label="V2")
    ax.plot(GOAL_POS, [t["clearance_v21"] for t in exp_d["table"]], "-o", ms=3, label="V2.1")
    ax2 = ax.twinx()
    ax2.plot(GOAL_POS, [t["area_v2"] for t in exp_d["table"]], "--s", ms=3, color="tab:orange",
             alpha=0.6, label="V2 area")
    ax2.plot(GOAL_POS, [t["area_v21"] for t in exp_d["table"]], "--s", ms=3, color="tab:blue",
             alpha=0.6, label="V2.1 area")
    ax.set_xlabel("normalised arc length (goal end)"); ax.set_ylabel("clearance [m]")
    ax2.set_ylabel("area [m2]"); ax.set_title("D: goal neighbourhood")
    ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    dst = os.path.join(viz_dir, f"{name}.png")
    fig.savefig(dst, dpi=110)
    plt.close(fig)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-v2", default=os.path.join(PROJECT_ROOT, "configs",
                                                        "data_generation_v2.yaml"))
    ap.add_argument("--config-v21", default=os.path.join(PROJECT_ROOT, "configs",
                                                         "data_generation_v2_1.yaml"))
    args = ap.parse_args()
    cfg2 = load_cfg(args.config_v2)
    cfg21 = load_cfg(args.config_v21)
    viz = os.path.join(cfg21["paths"]["viz_dir"], "stage3")
    os.makedirs(viz, exist_ok=True)
    out: Dict[str, object] = {}

    # ---------- S channel (synthetic, fixed pair)
    occ, start, goal = build_layout("s_channel")
    scene_s = make_scene(occ, cfg21, "s_channel")
    bnd_s = scene_s.obstacle_boundary_cells()
    pair_s = (start, goal)
    a_s = experiment_a(scene_s, pair_s, cfg2, cfg21, bnd_s)
    b_s = experiment_b(scene_s, pair_s, cfg21, bnd_s)
    c_s = experiment_c(scene_s, pair_s, cfg21, bnd_s)
    d_s = experiment_d(scene_s, pair_s, cfg2, cfg21, bnd_s)
    print(f"[S channel] A: median ratio {a_s['summary']['ratio_median']:.2f} "
          f"({a_s['summary']['n_worse']}/{len(POS)} worse); areas "
          f"{a_s['summary']['area_p50_v2']:.3f} -> {a_s['summary']['area_p50_v21']:.3f} m2; "
          f"clearance {a_s['summary']['mean_clearance_v2']:.2f} -> "
          f"{a_s['summary']['mean_clearance_v21']:.2f} m")
    print(f"[S channel] B init ablation: " +
          ", ".join(f"{k}:{v['area_p50']:.3f}/{v['ms']:.0f}ms" for k, v in b_s.items()))
    print(f"[S channel] C multi-res: " +
          ", ".join(f"{k}:{v['area_p50']:.3f}/{v['ms']:.0f}ms/{v['label_status']}"
                    for k, v in c_s.items()))
    out["s_channel"] = {"A": a_s, "B": b_s, "C": c_s, "D": d_s}
    fig_experiment("s_channel", a_s, d_s, viz, "Stage 3 - S channel")

    # ---------- Berlin (real map, goal-neighbourhood degeneration)
    rec = next(r for r in build_map_catalog(cfg21["paths"]["map_dir"], 0.25)
               if r.parsed_ok and r.file_name == "Berlin_2_512_part_0_0.map")
    src, meta = load_moving_ai_map(os.path.join(cfg21["paths"]["map_dir"], rec.rel_path))
    r0 = float(meta["resolution"] or 0.25)
    crop = int(round(48.0 / r0))
    scene_b = build_scene(src, r0, (247, 209), crop, 256, 48.0, 0.25, 0.04, 4,
                          "berlin_s3", rec.file_name)
    bnd_b = scene_b.obstacle_boundary_cells()
    pair_b = sample_start_goal(scene_b, np.random.default_rng(42), cfg21, [])
    a_b = experiment_a(scene_b, pair_b, cfg2, cfg21, bnd_b)
    d_b = experiment_d(scene_b, pair_b, cfg2, cfg21, bnd_b)
    out["berlin"] = {"A": a_b, "D": d_b}
    print(f"[Berlin]    A: median ratio {a_b['summary']['ratio_median']:.2f} "
          f"({a_b['summary']['n_worse']}/{len(POS)} worse); areas "
          f"{a_b['summary']['area_p50_v2']:.2f} -> {a_b['summary']['area_p50_v21']:.2f} m2")
    print("[Berlin]    D goal neighbourhood:")
    for t in d_b["table"]:
        print(f"             s={t['s']:.2f} clearance {t['clearance_v2']:.2f}->"
              f"{t['clearance_v21']:.2f} m | offset {t['offset_v2']:.2f}->{t['offset_v21']:.2f} "
              f"| area {t['area_v2']:.2f}->{t['area_v21']:.2f} m2")
    print("[Berlin]    D ablation: " +
          ", ".join(f"{k}:{v['area_p50']:.2f}" for k, v in d_b["ablation"].items()))
    fig_experiment("berlin", a_b, d_b, viz, "Stage 3 - Berlin (goal neighbourhood)")

    dst = os.path.join(cfg21["paths"]["report_dir"], "stage3_experiments.json")
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\n-> {dst}\nfigures -> {viz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
