"""V2.1 stage 3: Berlin paired comparison at matched normalised arc positions.

V2 and V2.1 produce different station/anchor counts, so pairing by index is
misleading. This samples both centrelines at the same normalised arc positions
and compares clearance, offset and labelled ellipse area there.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from pathsafe_ellipse.map_io import build_map_catalog, load_moving_ai_map  # noqa: E402
from pathsafe_ellipse.v2.centerline import build_path  # noqa: E402
from pathsafe_ellipse.v2.ellipse import make_label  # noqa: E402
from pathsafe_ellipse.v2.pipeline import anchor_indices, sample_start_goal  # noqa: E402
from pathsafe_ellipse.v2.scene import build_scene  # noqa: E402
from v2_1_scenarios import load_cfg, normalised_arc, sample_at_normalised  # noqa: E402

POS = np.linspace(0.05, 0.95, 10)


def run(scene, start, goal, cfg, boundary):
    path = build_path(scene, start, goal, cfg)
    anchors = anchor_indices(scene, path, cfg["anchor"])
    labels = [make_label(scene, path.q_xy[i], boundary, cfg) for i in anchors]
    areas = np.array([l.area for l in labels])
    s = normalised_arc(path.q_xy)
    s_a = s[np.asarray(anchors)]
    return {
        "path": path,
        "clearance": sample_at_normalised(path.clearance_q, s, POS),
        "offset": sample_at_normalised(np.linalg.norm(path.q_xy - path.p_xy, axis=1), s, POS),
        "area": np.interp(POS, s_a, areas) if len(areas) > 1 else np.zeros_like(POS),
        "area_p50": float(np.median(areas)),
        "n_anchors": int(len(anchors)),
    }


def main() -> int:
    cfg2 = load_cfg(os.path.join(PROJECT_ROOT, "configs", "data_generation_v2.yaml"))
    cfg21 = load_cfg(os.path.join(PROJECT_ROOT, "configs", "data_generation_v2_1.yaml"))
    rec = next(r for r in build_map_catalog(cfg21["paths"]["map_dir"], 0.25)
               if r.parsed_ok and r.file_name == "Berlin_2_512_part_0_0.map")
    src, meta = load_moving_ai_map(os.path.join(cfg21["paths"]["map_dir"], rec.rel_path))
    r0 = float(meta["resolution"] or 0.25)
    crop = int(round(48.0 / r0))
    scene = build_scene(src, r0, (247, 209), crop, 256, 48.0, 0.25, 0.04, 4,
                        "berlin_paired", rec.file_name)
    boundary = scene.obstacle_boundary_cells()
    rng = np.random.default_rng(42)
    pair = sample_start_goal(scene, rng, cfg21, [])

    a = run(scene, pair[0], pair[1], cfg2, boundary)
    b = run(scene, pair[0], pair[1], cfg21, boundary)

    print("paired comparison at matched normalised arc positions")
    print(f"{'s_norm':>7} {'D(q) V2':>9} {'D(q) V2.1':>10} {'off V2':>7} {'off V2.1':>9} "
          f"{'area V2':>8} {'area V2.1':>10} {'ratio':>6}")
    ratios = []
    for i, s in enumerate(POS):
        r = b["area"][i] / a["area"][i] if a["area"][i] > 1e-9 else np.nan
        ratios.append(r)
        print(f"{s:7.2f} {a['clearance'][i]:9.2f} {b['clearance'][i]:10.2f} "
              f"{a['offset'][i]:7.2f} {b['offset'][i]:9.2f} {a['area'][i]:8.2f} "
              f"{b['area'][i]:10.2f} {r:6.2f}")
    ratios = np.asarray(ratios, dtype=float)
    print(f"\nclearance gain: V2 {np.mean(a['clearance']-a['clearance']):+.2f} vs "
          f"V2.1 {(b['clearance']-a['clearance']).mean():+.2f} m "
          f"(mean over positions: V2 {a['clearance'].mean():.2f} -> V2.1 {b['clearance'].mean():.2f})")
    print(f"area ratio V2.1/V2 at matched positions: median {np.nanmedian(ratios):.2f}, "
          f"min {np.nanmin(ratios):.2f}, max {np.nanmax(ratios):.2f}; "
          f"positions with ratio < 1: {int(np.nansum(ratios < 1))}/{len(ratios)}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(POS, a["clearance"], "-o", ms=3, label="V2")
    axes[0].plot(POS, b["clearance"], "-o", ms=3, label="V2.1")
    axes[0].set_xlabel("normalised arc length"); axes[0].set_ylabel("D(q) [m]")
    axes[0].set_title("clearance at matched arc positions"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(POS, a["offset"], "-o", ms=3, label="V2")
    axes[1].plot(POS, b["offset"], "-o", ms=3, label="V2.1")
    axes[1].axhline(4.0, color="k", ls="--", lw=0.8, label="R_max")
    axes[1].set_xlabel("normalised arc length"); axes[1].set_ylabel("|q-p| [m]")
    axes[1].set_title("offset"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].plot(POS, a["area"], "-o", ms=3, label="V2")
    axes[2].plot(POS, b["area"], "-o", ms=3, label="V2.1")
    axes[2].set_xlabel("normalised arc length"); axes[2].set_ylabel("ellipse area [m2]")
    axes[2].set_title("label area"); axes[2].legend(); axes[2].grid(alpha=0.3)
    fig.suptitle("V2.1 stage 3 - Berlin paired comparison (matched normalised arc length)")
    fig.tight_layout()
    viz = os.path.join(cfg21["paths"]["viz_dir"], "berlin_case")
    os.makedirs(viz, exist_ok=True)
    dst_fig = os.path.join(viz, "paired_normalised.png")
    fig.savefig(dst_fig, dpi=115)
    plt.close(fig)

    dst = os.path.join(cfg21["paths"]["report_dir"], "berlin_paired.json")
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump({"positions": POS.tolist(),
                   "v2": {k: np.asarray(v).tolist() for k, v in a.items()
                          if isinstance(v, np.ndarray)},
                   "v2_1": {k: np.asarray(v).tolist() for k, v in b.items()
                            if isinstance(v, np.ndarray)},
                   "area_ratio_median": float(np.nanmedian(ratios)),
                   "n_ratio_below_one": int(np.nansum(ratios < 1)),
                   "n_positions": len(POS)}, fh, indent=2)
    print(f"-> {dst}\n-> {dst_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
