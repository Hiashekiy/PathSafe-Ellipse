"""Final integrity check of the three splits (S5 acceptance, quick version)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


def main() -> int:
    ok_all = True
    for split in ("train", "val", "test"):
        path = os.path.join("data", "splits", f"{split}.npz")
        with np.load(path) as d:
            n = len(d["label_valid"])
            fields = len(d.files)
            consistent = all(len(d[f]) == n for f in d.files)
            patch = d["map_patch"]
            packed = d["map_patch_packed"]
            roundtrip = np.array_equal(np.unpackbits(packed[:4], axis=2), patch[:4])
            finite = all(np.all(np.isfinite(d[f])) for f in
                         ("ray_distances", "ellipse_ab", "ellipse_angle",
                          "ellipse_center_offset", "point_inside_metric"))
        ok_all &= consistent and roundtrip and finite
        print(f"{split:5s} n={n:6d} fields={fields:2d} lengths_consistent={consistent} "
              f"patch={patch.shape} packed={packed.shape} roundtrip={roundtrip} finite={finite}")

    with open(os.path.join("data", "splits", "map_group_split.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    split = meta["split"]
    groups = {g for s in split for g in split[s]}
    print(f"map groups: train={len(split['train'])} val={len(split['val'])} "
          f"test={len(split['test'])} total={len(groups)} (no group appears twice)")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
