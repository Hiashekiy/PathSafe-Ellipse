"""Dataset card: aggregate statistics over a generated V2.1 pilot/dataset run."""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dataset_v21")
    args = ap.parse_args()
    p = os.path.join(ROOT, "reports_v2_1", f"pilot_{args.tag}_summary.json")
    s = json.load(open(p, encoding="utf-8"))
    tot = {k: 0 for k in ("paths", "acc", "fail", "labels", "safe", "cont", "both",
                          "conv", "fnc", "fb", "failed", "inserted")}
    dens, area50, ell, cl_ms, wall = [], [], [], [], []
    groups, fields = set(), {}
    for r in s["scenes"]:
        v = r.get("v21")
        if not v or "n_anchors" not in v:
            continue
        tot["paths"] += v["n_paths"]
        tot["acc"] += v["n_paths_accepted"]
        tot["fail"] += v["n_paths_failed"]
        tot["labels"] += v["n_anchors"]
        tot["safe"] += v["n_ellipse_safe"]
        tot["cont"] += v["n_continuity_verified"]
        tot["both"] += v["n_both_verified"]
        tot["conv"] += v["n_solver_converged"]
        tot["fnc"] += v["n_feasible_not_converged"]
        tot["fb"] += v["n_fallback"]
        tot["failed"] += v["n_failed"]
        tot["inserted"] += v["n_inserted"]
        dens.append(v["density_inflated"])
        fields[int(r["field_m"])] = fields.get(int(r["field_m"]), 0) + 1
        groups.add(r["map_group"])
        if v.get("areas_p50") is not None:
            area50.append(v["areas_p50"])
        if v.get("solve_ms_median") is not None:
            ell.append(v["solve_ms_median"])
        if v.get("wall_ms") is not None:
            wall.append(v["wall_ms"])
        for pr in v["paths"]:
            if pr.get("centerline_ms") is not None:
                cl_ms.append(pr["centerline_ms"])
    out = {
        "tag": args.tag, "label_version": s.get("label_version"),
        "git_commit": s.get("git_commit"), "n_scenes": len(s["scenes"]),
        "n_map_groups": len(groups), "field_histogram": fields,
        "paths": tot["paths"], "paths_accepted": tot["acc"],
        "paths_failed": tot["fail"], "labels": tot["labels"],
        "labels_safe": tot["safe"], "labels_continuity": tot["cont"],
        "labels_both": tot["both"], "labels_invalid": tot["labels"] - tot["both"],
        "solver_converged": tot["conv"], "feasible_not_converged": tot["fnc"],
        "fallbacks": tot["fb"], "failed_paths": tot["failed"],
        "inserted_anchors": tot["inserted"],
        "density_min": float(min(dens)), "density_p50": float(np.median(dens)),
        "density_max": float(max(dens)),
        "scene_area_p50_median": float(np.median(area50)) if area50 else None,
        "ellipse_ms_p50": float(np.median(ell)) if ell else None,
        "ellipse_ms_p95": float(np.percentile(ell, 95)) if ell else None,
        "centreline_ms_p50": float(np.median(cl_ms)) if cl_ms else None,
        "centreline_ms_p95": float(np.percentile(cl_ms, 95)) if cl_ms else None,
        "scene_ms_p50": float(np.median(wall)) if wall else None,
        "valid_label_rate": tot["both"] / max(tot["labels"], 1),
        "acceptance_rate": tot["acc"] / max(tot["paths"], 1),
    }
    for k, v in out.items():
        print(f"{k:26s} {v}")
    dst = os.path.join(ROOT, "reports_v2_1", f"dataset_card_{args.tag}.json")
    json.dump(out, open(dst, "w", encoding="utf-8"), indent=2)
    print(f"\n-> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
