"""Verify a pilot run: NPZ completeness, cross-count reconciliation, retrievability."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from pathsafe_ellipse.v2.pilot import LABEL_FIELDS, labels_of_path, load_scene_npz, path_polyline  # noqa: E402

REQUIRED_META = ("label_version", "git_commit", "r_max_m", "d_safe_m",
                 "segment_check_step_cells", "homotopy_check_step_cells", "seed",
                 "source_map_id", "source_rel_path", "scene_id", "r_scene_m",
                 "crop_origin", "robot_radius_m", "safety_margin_m", "lidar_rays",
                 "anchor_spacing", "rho_max")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="stage2smoke")
    ap.add_argument("--root", default=os.path.join(PROJECT_ROOT, "data", "processed_v2_1"))
    args = ap.parse_args()
    base = os.path.join(args.root, f"pilot_{args.tag}")
    with open(os.path.join(base, "index.json"), encoding="utf-8") as fh:
        index = json.load(fh)

    checks = []

    def chk(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    n_files = 0
    meta_ok = labels_ok = counts_ok = retrieve_ok = valid_ok = shapes_ok = True
    versions = set()
    for e in index["entries"]:
        p = os.path.join(base, e.get("dir") or e["variant"].replace("v21", "v2_1"), e["npz"])
        if not os.path.exists(p):
            meta_ok = False
            continue
        n_files += 1
        d = load_scene_npz(p)
        meta = d.get("meta", {})
        versions.add((e["variant"], str(meta.get("label_version"))))
        missing = [k for k in REQUIRED_META if k not in meta]
        if missing:
            meta_ok = False
            print(f"   missing meta in {e['npz']}: {missing}")
        missing_fields = [k for k in LABEL_FIELDS if k not in d and k != "point_id"]
        if missing_fields:
            labels_ok = False
            print(f"   missing label fields in {e['npz']}: {missing_fields}")
        n = int(d["path_id"].shape[0]) if "path_id" in d else 0
        if n != int(e["n_anchors"]):
            counts_ok = False
        # per-path counts must sum to the file total
        if "path_n_labels" in d:
            per_path = int(np.nansum(d["path_n_labels"]))
            if per_path != n:
                counts_ok = False
                print(f"   count mismatch in {e['npz']}: labels {n} vs sum(paths) {per_path}")
        if n:
            # point ids unique + contiguous
            pid = d["point_id"]
            if len(np.unique(pid)) != n or not np.array_equal(np.sort(pid), np.arange(n)):
                shapes_ok = False
            # label_valid == safety AND continuity
            both = (d["safety_verified"].astype(bool) & d["continuity_verified"].astype(bool))
            if not np.array_equal(d["label_valid"].astype(bool), both):
                valid_ok = False
            # ellipse / lidar shapes
            if d["ellipse_ab"].shape != (n, 2) or d["ellipse_S"].shape != (n, 4):
                shapes_ok = False
            rays = int(meta.get("lidar_rays", 0))
            if d["lidar_ranges"].shape != (n, rays):
                shapes_ok = False
            if not np.all(np.isfinite(d["ellipse_area"])):
                shapes_ok = False
            # retrievability by path_id
            for pid_k in np.unique(d["path_id"])[:2]:
                sub = labels_of_path(d, int(pid_k))
                if len(sub["q_xy"]) == 0:
                    retrieve_ok = False
                poly = path_polyline(d, int(pid_k), "q")
                if len(poly) < 2:
                    retrieve_ok = False

    chk("all indexed NPZ files load", n_files == len(index["entries"]),
        f"{n_files}/{len(index['entries'])}")
    chk("metadata complete (version/git/r_max/d_safe/sampling/seed/map/scene)", meta_ok)
    chk("label fields complete", labels_ok)
    chk("label counts reconcile (file total == sum over paths == index)", counts_ok)
    chk("array shapes and finiteness (ellipse, lidar rays)", shapes_ok)
    chk("point_id unique and contiguous", shapes_ok)
    chk("label_valid == safety_verified AND continuity_verified", valid_ok)
    chk("retrievable by scene_id + path_id", retrieve_ok)
    chk("V2 and V2.1 stored in separate directories with different version tags",
        os.path.isdir(os.path.join(base, "v2")) and os.path.isdir(os.path.join(base, "v2_1"))
        and versions <= {("v2", "v2"), ("v21", "v2.1")} and bool(versions),
        f"version tags seen: {sorted(versions)}")

    n_pass = sum(1 for c in checks if c["pass"])
    print(f"\n{n_pass}/{len(checks)} checks passed")
    out_dir = os.path.join(PROJECT_ROOT, "reports_v2_1_followup")
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, f"stage2_verify_{args.tag}.json")
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump({"tag": args.tag, "n_files": n_files, "checks": checks,
                   "n_pass": n_pass, "n_total": len(checks)}, fh, indent=2)
    print(f"-> {dst}")
    return 0 if n_pass == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
