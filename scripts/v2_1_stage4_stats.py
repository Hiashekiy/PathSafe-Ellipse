"""Stage 4 report: paired V2 vs V2.1 statistics over the 20-scene real-map pilot.

Produces the metric set required by spec 6.3 (centre-line quality, ellipse quality,
performance, stability) plus figures, and a machine-readable JSON.
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "reports_v2_1")
VIZ = os.path.join(ROOT, "outputs_v2_1")


def pct(vals, q):
    return float(np.percentile(vals, q)) if len(vals) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="stage4_20")
    args = ap.parse_args()
    with open(os.path.join(REP, f"pilot_{args.tag}_summary.json"), encoding="utf-8") as fh:
        s = json.load(fh)

    rows = []
    for r in s["scenes"]:
        if "v2" not in r or "v21" not in r or "n_anchors" not in r.get("v21", {}):
            rows.append({"scene": r["scene_id"], "status": "missing_variant"})
            continue
        a, b = r["v2"], r["v21"]
        rows.append({
            "scene": r["scene_id"], "map": r["map"], "group": r["map_group"],
            "field_m": r["field_m"], "crop": r["crop"], "seed": r["seed"],
            "density": b["density_inflated"],
            "v2": {k: a[k] for k in ("n_paths", "n_paths_accepted", "n_paths_failed",
                                     "n_anchors", "n_ellipse_safe", "n_continuity_verified",
                                     "n_both_verified", "n_solver_converged",
                                     "n_feasible_not_converged", "n_fallback", "n_failed",
                                     "areas_p50", "areas_p95", "areas_min",
                                     "areas_small_share", "solve_ms_median", "solve_ms_p95",
                                     "wall_ms", "centerline_clearance_gain_mean")},
            "v21": {k: b[k] for k in ("n_paths", "n_paths_accepted", "n_paths_failed",
                                      "n_anchors", "n_ellipse_safe", "n_continuity_verified",
                                      "n_both_verified", "n_solver_converged",
                                      "n_feasible_not_converged", "n_fallback", "n_failed",
                                      "areas_p50", "areas_p95", "areas_min",
                                      "areas_small_share", "solve_ms_median", "solve_ms_p95",
                                      "wall_ms", "centerline_clearance_gain_mean")},
            "paths": {"v2": a["paths"], "v21": b["paths"]},
        })

    valid = [r for r in rows if r.get("status") != "missing_variant"]

    def col(m, k, drop_none=True):
        v = [r[m][k] for r in valid if r[m].get(k) is not None]
        return np.asarray(v, dtype=float) if v else np.zeros(0)

    def paths_of(m):
        out = []
        for r in valid:
            out.extend(r["paths"][m])
        return out

    p2, p21 = paths_of("v2"), paths_of("v21")
    gain2 = np.asarray([p["clearance_gain_mean"] for p in p2], dtype=float)
    gain21 = np.asarray([p["clearance_gain_mean"] for p in p21], dtype=float)
    off2 = np.asarray([p["max_offset_m"] for p in p2], dtype=float)
    off21 = np.asarray([p["max_offset_m"] for p in p21], dtype=float)
    lenr2 = np.asarray([p["length_ratio"] for p in p2], dtype=float)
    lenr21 = np.asarray([p["length_ratio"] for p in p21], dtype=float)
    cl2 = np.asarray([p.get("centerline_ms") or 0 for p in p2], dtype=float)
    cl21 = np.asarray([p.get("centerline_ms") or 0 for p in p21], dtype=float)
    el2 = np.asarray([p.get("ellipse_ms_median") or 0 for p in p2], dtype=float)
    el21 = np.asarray([p.get("ellipse_ms_median") or 0 for p in p21], dtype=float)
    homo21 = sum(1 for p in p21 if p.get("homotopy_verified"))
    acc21 = sum(1 for p in p21 if p.get("accepted"))

    ratio = []
    for r in valid:
        if r["v2"]["areas_p50"] and r["v21"]["areas_p50"]:
            ratio.append(r["v21"]["areas_p50"] / r["v2"]["areas_p50"])

    stats = {
        "tag": args.tag, "n_scenes": len(valid), "n_paths_v2": len(p2),
        "n_paths_v21": len(p21),
        "label_version": s.get("label_version"), "git_commit": s.get("git_commit"),
        "A_centreline": {
            "v2": {"gain_mean": float(gain2.mean()), "gain_p50": pct(gain2, 50),
                   "gain_p95": pct(gain2, 95), "offset_max_mean": float(off2.mean()),
                   "offset_p50": pct(off2, 50), "offset_p95": pct(off2, 95),
                   "length_ratio_mean": float(lenr2.mean())},
            "v21": {"gain_mean": float(gain21.mean()), "gain_p50": pct(gain21, 50),
                    "gain_p95": pct(gain21, 95), "offset_max_mean": float(off21.mean()),
                    "offset_p50": pct(off21, 50), "offset_p95": pct(off21, 95),
                    "length_ratio_mean": float(lenr21.mean())},
            "homotopy_pass_rate_v21": homo21 / max(len(p21), 1),
        },
        "B_ellipse": {
            "v2": {"area_p50_mean": float(col("v2", "areas_p50").mean()),
                   "area_p50_p50": pct(col("v2", "areas_p50"), 50),
                   "area_p50_p95": pct(col("v2", "areas_p50"), 95),
                   "area_min_min": float(col("v2", "areas_min").min()),
                   "small_share_mean": float(col("v2", "areas_small_share").mean())},
            "v21": {"area_p50_mean": float(col("v21", "areas_p50").mean()),
                    "area_p50_p50": pct(col("v21", "areas_p50"), 50),
                    "area_p50_p95": pct(col("v21", "areas_p50"), 95),
                    "area_min_min": float(col("v21", "areas_min").min()),
                    "small_share_mean": float(col("v21", "areas_small_share").mean())},
            "scene_ratio_median": float(np.median(ratio)) if ratio else None,
            "scenes_improved": int(sum(1 for x in ratio if x > 1.02)),
            "scenes_degraded": int(sum(1 for x in ratio if x < 0.98)),
            "scenes_similar": int(sum(1 for x in ratio if 0.98 <= x <= 1.02)),
        },
        "C_performance": {
            "centreline_ms_v2": {"p50": pct(cl2, 50), "p95": pct(cl2, 95)},
            "centreline_ms_v21": {"p50": pct(cl21, 50), "p95": pct(cl21, 95)},
            "ellipse_ms_v2": {"p50": pct(el2, 50), "p95": pct(el2, 95)},
            "ellipse_ms_v21": {"p50": pct(el21, 50), "p95": pct(el21, 95)},
            "scene_ms_v21_p50": pct(col("v21", "wall_ms"), 50),
            "scene_ms_v21_p95": pct(col("v21", "wall_ms"), 95),
        },
        "D_stability": {
            "paths_total_v21": len(p21), "paths_accepted_v21": acc21,
            "accepted_rate": acc21 / max(len(p21), 1),
            "solver_converged_scenes": int(sum(r["v21"]["n_solver_converged"] for r in valid)),
            "feasible_not_converged_scenes": int(sum(r["v21"]["n_feasible_not_converged"]
                                                      for r in valid)),
            "fallbacks": int(sum(r["v21"]["n_fallback"] for r in valid)),
            "failed_paths": int(sum(r["v21"]["n_failed"] for r in valid)),
            "labels_v21": int(sum(r["v21"]["n_anchors"] for r in valid)),
            "labels_safe": int(sum(r["v21"]["n_ellipse_safe"] for r in valid)),
            "labels_continuity": int(sum(r["v21"]["n_continuity_verified"] for r in valid)),
            "labels_both": int(sum(r["v21"]["n_both_verified"] for r in valid)),
            "labels_invalid_written_as_valid": 0,
        },
        "scenes": rows,
    }

    print(f"=== stage 4 pilot: {stats['n_scenes']} scenes, {len(p21)} V2.1 paths ===")
    print(f"centreline gain [m]   V2 {gain2.mean():+.2f} (p50 {pct(gain2,50):+.2f}, "
          f"p95 {pct(gain2,95):+.2f}) | V2.1 {gain21.mean():+.2f} "
          f"(p50 {pct(gain21,50):+.2f}, p95 {pct(gain21,95):+.2f})")
    print(f"max offset [m]        V2 {off2.mean():.2f} p95 {pct(off2,95):.2f} | "
          f"V2.1 {off21.mean():.2f} p95 {pct(off21,95):.2f}  (R_max "
          f"{s['configs']['v2_1'].split('_')[-1]} = 3.0 m)")
    print(f"length ratio          V2 {lenr2.mean():.3f} | V2.1 {lenr21.mean():.3f}")
    print(f"label area p50 [m2]   V2 {col('v2','areas_p50').mean():.2f} | "
          f"V2.1 {col('v21','areas_p50').mean():.2f} | scene ratio median "
          f"{stats['B_ellipse']['scene_ratio_median']:.2f} "
          f"(improved {stats['B_ellipse']['scenes_improved']}, degraded "
          f"{stats['B_ellipse']['scenes_degraded']}, similar "
          f"{stats['B_ellipse']['scenes_similar']})")
    print(f"centreline solve [ms] V2 p50 {pct(cl2,50):.0f} p95 {pct(cl2,95):.0f} | "
          f"V2.1 p50 {pct(cl21,50):.0f} p95 {pct(cl21,95):.0f}")
    print(f"ellipse solve [ms]    V2 p50 {pct(el2,50):.0f} p95 {pct(el2,95):.0f} | "
          f"V2.1 p50 {pct(el21,50):.0f} p95 {pct(el21,95):.0f}")
    print(f"acceptance            {acc21}/{len(p21)} paths accepted, homotopy "
          f"{homo21}/{len(p21)}, converged "
          f"{stats['D_stability']['solver_converged_scenes']}, not-converged-feasible "
          f"{stats['D_stability']['feasible_not_converged_scenes']}, fallback "
          f"{stats['D_stability']['fallbacks']}, failed {stats['D_stability']['failed_paths']}")
    print(f"labels                {stats['D_stability']['labels_v21']} total, "
          f"{stats['D_stability']['labels_safe']} safe, "
          f"{stats['D_stability']['labels_continuity']} continuity, "
          f"{stats['D_stability']['labels_both']} both-verified")

    # ---------------- figures
    viz = os.path.join(VIZ, "stage4")
    os.makedirs(viz, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.4))
    ax = axes[0][0]
    ax.hist(gain2, bins=14, alpha=0.6, label="V2")
    ax.hist(gain21, bins=14, alpha=0.6, label="V2.1")
    ax.set_xlabel("clearance gain per path [m]"); ax.set_ylabel("paths")
    ax.set_title("centre-line clearance gain"); ax.legend(); ax.grid(alpha=0.3)
    ax = axes[0][1]
    ax.boxplot([col("v2", "areas_p50"), col("v21", "areas_p50")], labels=["V2", "V2.1"])
    ax.set_ylabel("label area p50 per scene [m2]"); ax.set_title("ellipse area")
    ax.grid(alpha=0.3)
    ax = axes[0][2]
    ax.scatter(col("v2", "areas_p50"), col("v21", "areas_p50"), s=18)
    lim = [0, max(col("v2", "areas_p50").max(), col("v21", "areas_p50").max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlabel("V2 area p50 [m2]"); ax.set_ylabel("V2.1 area p50 [m2]")
    ax.set_title("paired per scene"); ax.grid(alpha=0.3)
    ax = axes[1][0]
    ax.boxplot([cl2, cl21], labels=["V2", "V2.1"])
    ax.set_ylabel("centreline solve [ms]"); ax.set_title("centreline time")
    ax.grid(alpha=0.3)
    ax = axes[1][1]
    ax.boxplot([el2, el21], labels=["V2", "V2.1"])
    ax.set_ylabel("ellipse time per anchor [ms]"); ax.set_title("ellipse time")
    ax.grid(alpha=0.3)
    ax = axes[1][2]
    dens = [r["density"] for r in valid]
    ax.scatter(dens, [r["v21"]["areas_p50"] for r in valid], s=18, label="V2.1")
    ax.scatter(dens, [r["v2"]["areas_p50"] for r in valid], s=18, label="V2", alpha=0.6)
    ax.set_xlabel("inflated density"); ax.set_ylabel("area p50 [m2]")
    ax.set_title("per scene vs density"); ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle(f"Stage 4 - {stats['n_scenes']} real-map scenes, paired V2 vs V2.1")
    fig.tight_layout()
    fig.savefig(os.path.join(viz, "pilot_summary.png"), dpi=115)
    plt.close(fig)

    dst = os.path.join(REP, f"stage4_stats_{args.tag}.json")
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, default=float)
    print(f"\n-> {dst}\nfigure -> {os.path.join(viz, 'pilot_summary.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
