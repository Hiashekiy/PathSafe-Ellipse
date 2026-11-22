"""Merge sharded pilot/dataset runs into one summary + index."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "reports_v2_1")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "processed_v2_1"))
    args = ap.parse_args()

    shard_files = sorted(glob.glob(os.path.join(REP, f"pilot_{args.tag}_shard*of*_summary.json")))
    if not shard_files:
        raise SystemExit(f"no shard summaries for tag {args.tag}")
    scenes: List[Dict] = []
    index: List[Dict] = []
    base = None
    for f in shard_files:
        d = json.load(open(f, encoding="utf-8"))
        base = base or {k: d.get(k) for k in ("tag", "seed", "paths_per_scene",
                                              "label_version", "git_commit", "configs", "dirs")}
        scenes.extend(d["scenes"])
        # shard summaries live in reports_v2_1/, the matching index next to the npz
        shard = os.path.basename(f).replace(f"pilot_{args.tag}_", "").replace("_summary.json", "")
        idx_path = os.path.join(args.root, f"pilot_{args.tag}", f"index_{shard}.json")
        if os.path.exists(idx_path):
            index.extend(json.load(open(idx_path, encoding="utf-8"))["entries"])
        else:
            print(f"  !! missing index for {shard}: {idx_path}")

    def agg(tag: str, key: str):
        vals = [r[tag][key] for r in scenes if tag in r and key in r[tag]
                and r[tag][key] is not None and not isinstance(r[tag][key], list)]
        return float(np.mean(vals)) if vals else None

    summary = {**base, "tag": args.tag, "n_scenes": len(scenes),
               "n_shards": len(shard_files), "scenes": scenes,
               "paired": {
                   "mean_gain_v2": agg("v2", "centerline_clearance_gain_mean"),
                   "mean_gain_v21": agg("v21", "centerline_clearance_gain_mean"),
                   "mean_area_p50_v2": agg("v2", "areas_p50"),
                   "mean_area_p50_v21": agg("v21", "areas_p50"),
                   "total_labels_v2": int(sum(r["v2"]["n_anchors"] for r in scenes if "v2" in r)),
                   "total_labels_v21": int(sum(r["v21"]["n_anchors"] for r in scenes if "v21" in r)),
                   "total_both_verified_v2": int(sum(r["v2"]["n_both_verified"] for r in scenes
                                                    if "v2" in r)),
                   "total_both_verified_v21": int(sum(r["v21"]["n_both_verified"] for r in scenes
                                                      if "v21" in r)),
               }}
    dst = os.path.join(REP, f"pilot_{args.tag}_summary.json")
    json.dump(summary, open(dst, "w", encoding="utf-8"), indent=2, default=float)
    idx_dst = os.path.join(args.root, f"pilot_{args.tag}", "index.json")
    json.dump({"n_files": len(index), "entries": index},
              open(idx_dst, "w", encoding="utf-8"), indent=2)
    tot = sum(r["v21"]["n_anchors"] for r in scenes if "v21" in r)
    both = sum(r["v21"]["n_both_verified"] for r in scenes if "v21" in r)
    print(f"merged {len(shard_files)} shards: {len(scenes)} scenes, "
          f"{len(index)} npz, {tot} labels ({both} both-verified)")
    print(f"-> {dst}\n-> {idx_dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
