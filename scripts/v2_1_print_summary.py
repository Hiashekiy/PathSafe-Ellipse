"""Print a pilot summary: per-path detail plus the eight reconciled counters.

Reads ``reports_v2_1/pilot_<tag>_summary.json`` (written by ``v2_1_pilot.py``).
Safety and continuity are printed as separate counts - never merged.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "reports_v2_1")


def load(tag: str) -> dict:
    dst = os.path.join(REP, f"pilot_{tag}_summary.json")
    if not os.path.exists(dst):
        legacy = os.path.join(REP, "v2_pilot_summary.json")
        if os.path.exists(legacy):
            print(f"!! {dst} not found; the legacy V2 summary format has no per-path "
                  f"detail - rerun scripts/v2_1_pilot.py --tag {tag}")
        raise SystemExit(f"missing {dst}")
    with open(dst, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="stage2smoke")
    ap.add_argument("--variant", default="v21", choices=["v21", "v2"])
    args = ap.parse_args()
    s = load(args.tag)
    v = args.variant

    print(f"pilot {s['tag']}  scenes {s['n_scenes']}  paths/scene {s['paths_per_scene']}  "
          f"label_version {s['label_version']}  commit {s['git_commit']}")
    print(f"{'scene':44s} {'paths':>7} {'labels':>7} {'safe':>6} {'cont':>6} {'both':>6} "
          f"{'conv':>5} {'fnc':>4} {'fb':>3} {'fail':>5} {'area p50':>9} {'ell ms':>7}")
    tot = {k: 0 for k in ("paths", "labels", "safe", "cont", "both", "conv", "fnc", "fb",
                          "fail")}
    for row in s["scenes"]:
        if v not in row or "n_anchors" not in row[v]:
            print(f"{row['scene_id'][:44]:44s}  (no {v} result: {row.get(v, {}).get('error', '?')})")
            continue
        d = row[v]
        tot["paths"] += int(d["n_paths_accepted"])
        tot["labels"] += int(d["n_anchors"])
        tot["safe"] += int(d["n_ellipse_safe"])
        tot["cont"] += int(d["n_continuity_verified"])
        tot["both"] += int(d["n_both_verified"])
        tot["conv"] += int(d["n_solver_converged"])
        tot["fnc"] += int(d["n_feasible_not_converged"])
        tot["fb"] += int(d["n_fallback"])
        tot["fail"] += int(d["n_failed"])
        print(f"{row['scene_id'][:44]:44s} "
              f"{d['n_paths_accepted']:3d}/{d['n_paths']:<3d} {d['n_anchors']:7d} "
              f"{d['n_ellipse_safe']:6d} {d['n_continuity_verified']:6d} "
              f"{d['n_both_verified']:6d} {d['n_solver_converged']:5d} "
              f"{d['n_feasible_not_converged']:4d} {d['n_fallback']:3d} {d['n_failed']:5d} "
              f"{(d['areas_p50'] or 0):9.2f} {(d['solve_ms_median'] or 0):7.1f}")
    print(f"{'TOTAL':44s} {tot['paths']:7d} {tot['labels']:7d} {tot['safe']:6d} "
          f"{tot['cont']:6d} {tot['both']:6d} {tot['conv']:5d} {tot['fnc']:4d} "
          f"{tot['fb']:3d} {tot['fail']:5d}")

    print("\nper-path detail")
    print(f"{'scene':34s} {'p':>2} {'status':24s} {'stn':>4} {'anc':>4} {'lab':>4} "
          f"{'gain':>6} {'off':>5} {'cl_ms':>7} {'homo':>5}")
    for row in s["scenes"]:
        if v not in row or "paths" not in row[v]:
            continue
        for p in row[v]["paths"]:
            gain = p.get("clearance_gain_mean")
            print(f"{row['scene_id'][:34]:34s} {p['path_id']:2d} "
                  f"{str(p.get('label_status', ''))[:24]:24s} {p.get('n_stations', 0):4d} "
                  f"{p.get('n_anchors', 0):4d} {p.get('n_labels', 0):4d} "
                  f"{(f'{gain:+.2f}' if gain is not None else '   -  '):>6} "
                  f"{(p.get('max_offset_m') or 0):5.2f} {(p.get('centerline_ms') or 0):7.0f} "
                  f"{str(p.get('homotopy_verified')):>5}")

    print("\nper-path ellipse time p50/p95 (ms)")
    times = [(p.get("ellipse_ms_median"), p.get("ellipse_ms_p95"))
             for row in s["scenes"] if v in row and "paths" in row[v] for p in row[v]["paths"]]
    p50 = [a for a, _ in times if a is not None]
    p95 = [b for _, b in times if b is not None]
    if p50:
        print(f"  p50 median {np.median(p50):.1f}  p95 median {np.median(p95):.1f}  "
              f"paths {len(p50)}")
    if "paired" in s:
        print("\npaired summary (V2 vs V2.1):")
        for k, val in s["paired"].items():
            print(f"  {k:28s} {val}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
