"""V2.1 stage 1 audits: homotopy coverage and boundary/mask/distance consistency.

Audit A - homotopy coverage
    A single discrete sweep is not a continuous proof, so this re-checks every
    accepted centreline on a *much finer* lambda grid (and with a denser segment
    sampling) and reports:
      * configured vs audited lambda_max and segment step,
      * whether the fine sweep ever finds a violation the production sweep missed,
      * the feasible-set structure over lambda (contiguous intervals, monotone
        prefix or islands) - a non-monotone structure is exactly what a coarse
        sweep could jump over,
      * the minimum clearance observed along the deformation (margin to d_safe).

Audit B - boundary / valid_mask / distance-field consistency
    * blocked == (dist <= r_robot + margin) | (valid_mask == 0)   (exact count)
    * is_free(x) == dist(x) > free_distance_m
    * disagreement between the continuous test and the legacy cell-centre lookup
    * border integrity and out-of-field behaviour (must be blocked)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from pathsafe_ellipse.map_io import build_map_catalog, load_moving_ai_map  # noqa: E402
from pathsafe_ellipse.v2.centerline import (  # noqa: E402
    build_path, path_samples, segment_path_free,
)
from pathsafe_ellipse.v2.scene import build_scene  # noqa: E402
from v2_1_scenarios import build_layout, load_cfg, make_scene  # noqa: E402

FINE_LAM_CELLS = 0.02       # 12.5x finer than the configured 0.25
FINE_SEG_CELLS = 0.05       # 5x denser than the configured 0.25


def feasible_at(scene, p, q, lam, seg_cells):
    ql = (1.0 - lam) * p + lam * q
    if not bool(np.all(scene.is_free(ql))):
        return False, float(np.min(scene.distance(ql)))
    pts = path_samples(ql, seg_cells * scene.r_scene)
    d = scene.distance(pts)
    return bool(np.all(d > scene.free_distance_m)), float(np.min(d))


def audit_homotopy(scene, p, q, cfg):
    cc = cfg["centerline"]
    seg_cfg = float(cc.get("segment_check_step_cells", 0.25))
    lam_cfg = float(cc.get("homotopy_check_step_cells", 0.25))
    d_safe = float(cfg["robot"]["radius_m"]) + float(cfg["robot"]["safety_margin_m"])
    disp = float(np.max(np.linalg.norm(q - p, axis=1))) if len(p) else 0.0

    def sweep(lam_cells, seg_cells, lam_max_guess=1.0):
        n_states = max(2, int(np.ceil(disp / max(lam_cells * scene.r_scene, 1e-9))) + 1)
        lams = np.linspace(0.0, 1.0, n_states)
        flags, margins = [], []
        for lam in lams:
            ok, mn = feasible_at(scene, p, q, lam, seg_cells)
            flags.append(ok)
            margins.append(mn)
        flags = np.asarray(flags)
        lm = 0.0
        for lam, ok in zip(lams, flags):
            if not ok:
                break
            lm = float(lam)
        return {"lam_cells": lam_cells, "seg_cells": seg_cells, "n_states": int(n_states),
                "lambda_max": lm, "all_feasible": bool(flags.all()),
                "min_margin_m": float(np.min(margins)), "lams": lams, "flags": flags,
                "margins": margins}

    prod = sweep(lam_cfg, seg_cfg)
    fine = sweep(FINE_LAM_CELLS, FINE_SEG_CELLS)

    # interval structure of the feasible set on the fine grid
    f = fine["flags"].astype(int)
    changes = np.flatnonzero(np.diff(f) != 0)
    intervals = []
    start = 0
    cur = f[0]
    for k in changes:
        intervals.append({"from": float(fine["lams"][start]), "to": float(fine["lams"][k]),
                          "feasible": bool(cur)})
        start = k + 1
        cur = f[k + 1]
    intervals.append({"from": float(fine["lams"][start]), "to": float(fine["lams"][-1]),
                      "feasible": bool(cur)})
    n_feasible_intervals = sum(1 for iv in intervals if iv["feasible"])
    monotone_prefix = bool(n_feasible_intervals <= 1 and intervals[0]["feasible"])
    return {
        "disp_m": disp,
        "production": {k: prod[k] for k in ("lam_cells", "seg_cells", "n_states", "lambda_max",
                                            "all_feasible", "min_margin_m")},
        "audit": {k: fine[k] for k in ("lam_cells", "seg_cells", "n_states", "lambda_max",
                                       "all_feasible", "min_margin_m")},
        "discretisation_risk": bool(fine["lambda_max"] < prod["lambda_max"] - 1e-9),
        "n_feasible_intervals": int(n_feasible_intervals),
        "monotone_prefix": monotone_prefix,
        "intervals": intervals,
        "margin_to_d_safe_m": float(fine["min_margin_m"] - d_safe),
        "seg_samples_per_m": 1.0 / max(FINE_SEG_CELLS * scene.r_scene, 1e-9),
    }


def audit_consistency(scene, name, n_probe=4000, rng=None):
    rng = rng or np.random.default_rng(0)
    thr = float(scene.free_distance_m)
    dist = scene.dist_m
    mismatch_blocked = int(np.sum(scene.blocked != ((dist <= thr) | (scene.valid_mask == 0))))
    lo, hi = 0.0, scene.size * scene.r_scene
    pts = rng.uniform(-2.0, hi + 2.0, size=(n_probe, 2))
    d = scene.distance(pts)
    free = scene.is_free(pts)
    free_cmp = d > thr
    isfree_mismatch = int(np.sum(free != free_cmp))
    outside = (pts[:, 0] < lo) | (pts[:, 0] > hi) | (pts[:, 1] < lo) | (pts[:, 1] > hi)
    outside_free = int(np.sum(free[outside])) if np.any(outside) else 0
    # continuous vs legacy cell-centre lookup
    rc = np.rint(scene.cell_of(pts)).astype(int)
    rc[:, 0] = np.clip(rc[:, 0], 0, scene.size - 1)
    rc[:, 1] = np.clip(rc[:, 1], 0, scene.size - 1)
    legacy_free = ~scene.blocked[rc[:, 1], rc[:, 0]]
    dis = int(np.sum(legacy_free != free))
    border_blocked = bool(scene.blocked[0, :].all() and scene.blocked[-1, :].all()
                          and scene.blocked[:, 0].all() and scene.blocked[:, -1].all())
    # every free cell centre must be reported free by the continuous test
    rr, cc_ = np.nonzero(scene.free)
    idx = rng.choice(len(rr), size=min(2000, len(rr)), replace=False) if len(rr) else []
    centres = np.column_stack([cc_[idx], rr[idx]]) * scene.r_scene if len(rr) else np.zeros((0, 2))
    centres_bad = int(np.sum(~scene.is_free(centres))) if len(centres) else 0
    return {
        "scene": name,
        "blocked_vs_dist_mask_mismatch": mismatch_blocked,
        "is_free_vs_distance_mismatch": isfree_mismatch,
        "legacy_cell_lookup_vs_continuous_disagreement": dis,
        "disagreement_rate": round(dis / n_probe, 5),
        "outside_points_judged_free": outside_free,
        "border_fully_blocked": border_blocked,
        "free_cell_centres_judged_free_violations": centres_bad,
        "threshold_m": thr,
        "valid_mask_fraction": float(scene.valid_mask.mean()),
        "border_required": bool(name in ("wall_one_side", "narrow_corridor", "s_channel",
                                         "right_angle", "u_loop")),
        "pass": bool(mismatch_blocked == 0 and isfree_mismatch == 0 and outside_free == 0
                     and centres_bad == 0
                     and (border_blocked or name.startswith("real:"))),
    }


def main() -> int:
    cfg = load_cfg(os.path.join(PROJECT_ROOT, "configs", "data_generation_v2_1.yaml"))
    out = {"homotopy": {}, "consistency": []}

    # ---- scenarios
    for name in ("wall_one_side", "narrow_corridor", "s_channel", "right_angle", "u_loop"):
        occ, start, goal = build_layout(name)
        scene = make_scene(occ, cfg, name)
        path = build_path(scene, start, goal, cfg)
        if path is None:
            continue
        h = audit_homotopy(scene, path.p_xy, path.q_xy, cfg)
        h["status"] = path.status
        h["n_stations"] = int(len(path.p_xy))
        h["segment_step_m"] = round(float(cfg["centerline"]["segment_check_step_cells"])
                                    * scene.r_scene, 5)
        out["homotopy"][name] = h
        print(f"[homo {name:15s}] stations {h['n_stations']:3d} disp {h['disp_m']:5.2f} m | "
              f"prod {h['production']['n_states']:4d} states (step "
              f"{h['production']['seg_cells']*scene.r_scene:.3f} m) lam_max "
              f"{h['production']['lambda_max']:.2f} | audit {h['audit']['n_states']:5d} states "
              f"(step {h['audit']['seg_cells']*scene.r_scene:.3f} m) lam_max "
              f"{h['audit']['lambda_max']:.2f} | risk {h['discretisation_risk']} | "
              f"intervals {h['n_feasible_intervals']} mono {h['monotone_prefix']} | "
              f"margin {h['margin_to_d_safe_m']:+.2f} m")
        out["consistency"].append(audit_consistency(scene, name))

    # ---- two real map scenes
    records = [r for r in build_map_catalog(cfg["paths"]["map_dir"], 0.25) if r.parsed_ok]
    rng = np.random.default_rng(3)
    for k in range(2):
        rec = records[int(rng.integers(0, len(records)))]
        src, meta = load_moving_ai_map(os.path.join(cfg["paths"]["map_dir"], rec.rel_path))
        r0 = float(meta["resolution"] or 0.25)
        field_m = 64.0
        crop = int(round(field_m / r0))
        h, w = src.shape
        x0, y0 = int(rng.integers(0, max(1, w - crop))), int(rng.integers(0, max(1, h - crop)))
        scene = build_scene(src, r0, (x0, y0), crop, 256, field_m,
                            float(cfg["robot"]["radius_m"]), float(cfg["robot"]["safety_margin_m"]),
                            int(cfg["robot"]["edt_upsample"]), f"real{k}", rec.file_name)
        out["consistency"].append(audit_consistency(scene, f"real:{rec.file_name}"))

    print("\nconsistency audit:")
    for c in out["consistency"]:
        print(f"  {c['scene'][:34]:36s} blocked/mask {c['blocked_vs_dist_mask_mismatch']} "
              f"| is_free/dist {c['is_free_vs_distance_mismatch']} "
              f"| legacy-vs-continuous {c['disagreement_rate']:.3%} "
              f"| outside-free {c['outside_points_judged_free']} "
              f"| border {c['border_fully_blocked']} | free-centre-bad "
              f"{c['free_cell_centres_judged_free_violations']} | pass {c['pass']}")

    dst = os.path.join(cfg["paths"]["report_dir"], "audits.json")
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n-> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
