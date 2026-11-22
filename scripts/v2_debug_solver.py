"""Debug one cap-active ellipse: why does the barrier stop short of CLARABEL?"""
from __future__ import annotations

import os
import sys

import numpy as np
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from pathsafe_ellipse.map_io import build_map_catalog, load_moving_ai_map  # noqa: E402
from pathsafe_ellipse.v2.centerline import build_path  # noqa: E402
from pathsafe_ellipse.v2.ellipse import (  # noqa: E402
    _det_min, _pack, exact_clearance, make_label, solve_s_barrier, solve_s_cvxpy,
)
from pathsafe_ellipse.v2.pipeline import anchor_indices, sample_start_goal  # noqa: E402
from pathsafe_ellipse.v2.scene import build_scene  # noqa: E402


def main() -> int:
    with open(os.path.join(PROJECT_ROOT, "configs", "data_generation_v2.yaml"), encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["paths"]["map_dir"] = os.path.normpath(os.path.join(PROJECT_ROOT, cfg["paths"]["map_dir"]))
    rng = np.random.default_rng(7)
    records = [r for r in build_map_catalog(cfg["paths"]["map_dir"], 0.25) if r.parsed_ok]
    rec = records[int(rng.integers(0, len(records)))]
    src, meta = load_moving_ai_map(os.path.join(cfg["paths"]["map_dir"], rec.rel_path))
    r0 = float(meta["resolution"] or 0.25)
    crop = int(round(64.0 / r0))
    h, w = src.shape
    x0, y0 = int(rng.integers(0, w - crop)), int(rng.integers(0, h - crop))
    scene = build_scene(src, r0, (x0, y0), crop, 256, 64.0, 0.25, 0.04, 4, "dbg", rec.file_name)
    boundary = scene.obstacle_boundary_cells()
    pair = sample_start_goal(scene, rng, cfg, [])
    path = build_path(scene, pair[0], pair[1], cfg)
    anchors = anchor_indices(scene, path, cfg["anchor"])[:12]
    target = int(anchors[5])
    lab = make_label(scene, path.q_xy[target], boundary, cfg)
    n = np.stack([h_.n for h_ in lab.halfspaces])
    b2 = np.array([h_.d2 for h_ in lab.halfspaces])

    S_bar, st = solve_s_barrier(n, b2, cfg["ellipse"]["min_eig_m2"],
                                cfg["ellipse"]["r_axis_max_m"] ** 2)
    S_cv, st_cv = solve_s_cvxpy(n, b2, cfg["ellipse"]["min_eig_m2"],
                                cfg["ellipse"]["r_axis_max_m"] ** 2)
    print(f"m={len(n)} barrier={st} cvxpy={st_cv}")
    for name, S in (("barrier", S_bar), ("cvxpy", S_cv)):
        eig = np.sort(np.linalg.eigvalsh(S))[::-1]
        slack = b2 - np.einsum("ij,jk,ik->i", n, S, n)
        k = int(np.argmin(slack))
        print(f"  {name}: axes={np.sqrt(eig).round(4)} det={np.linalg.det(S):.4f} "
              f"min_slack={slack.min():.3e} (constraint {k}, b={b2[k]:.4f})")
    print(f"  area barrier={np.pi*np.sqrt(np.linalg.det(S_bar)):.3f} "
          f"cvxpy={np.pi*np.sqrt(np.linalg.det(S_cv)):.3f}")
    # which direction is the major axis, and what binds it
    eig_b, vec_b = np.linalg.eigh(S_bar)
    print(f"  barrier major dir={vec_b[:,1].round(3)} minor dir={vec_b[:,0].round(3)}")
    ok_b, mq_b, _, _ = exact_clearance(S_bar, lab.q, scene.blocked.astype(np.uint8), scene.r_scene)
    ok_c, mq_c, _, _ = exact_clearance(S_cv, lab.q, scene.blocked.astype(np.uint8), scene.r_scene)
    print(f"  exact: barrier clear={ok_b} minq={mq_b:.3f} | cvxpy clear={ok_c} minq={mq_c:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
