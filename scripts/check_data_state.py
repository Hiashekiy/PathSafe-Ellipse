"""Report what label data currently exists on disk."""
from __future__ import annotations

import glob
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)


def main() -> int:
    print("=== final labelled dataset (data/splits) ===")
    total = 0
    for split in ("train", "val", "test"):
        path = f"data/splits/{split}.npz"
        if not os.path.exists(path):
            print(f"  {split:5s} MISSING")
            continue
        with np.load(path) as d:
            n = len(d["label_valid"])
            total += n
            len_ok = all(len(d[f]) == n for f in d.files)
            finite_fields = ("ray_distances", "ellipse_ab", "ellipse_angle",
                             "ellipse_center_offset", "point_inside_metric")
            finite_ok = all(np.all(np.isfinite(d[f])) for f in finite_fields)
            a = d["ellipse_ab"][:, 0]
            b = d["ellipse_ab"][:, 1]
            all_valid = int(d["label_valid"].sum()) == n
            mtime = os.path.getmtime(path)
            import datetime
            ts = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            print(f"  {split:5s} n={n:6d} fields={len(d.files):2d} len_ok={len_ok} "
                  f"finite={finite_ok} all_label_valid={all_valid} "
                  f"a_p50={np.median(a):.2f}m b_p50={np.median(b):.2f}m "
                  f"{os.path.getsize(path) / 1e6:.1f}MB  (mtime {ts})")
    print(f"  total samples: {total}")
    print()
    print("=== intermediate per-map label cache (data/intermediate/samples) ===")
    npz = sorted(glob.glob("data/intermediate/samples/*.npz"))
    print(f"  files: {len(npz)}")
    for f in npz:
        import datetime
        with np.load(f) as d:
            ts = datetime.datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d %H:%M")
            print(f"    {os.path.basename(f)}: n={len(d['label_valid'])} (mtime {ts})")
    print()
    print("=== other inputs / caches ===")
    print(f"  raw maps            : {len(glob.glob('data/street-map/**/*.map', recursive=True))}")
    print(f"  inflated map caches : {len(glob.glob('data/processed/inflated_maps/*.npz'))}")
    print(f"  A* path caches      : {len(glob.glob('data/intermediate/paths/*.npz'))}")
    print(f"  map_index.json      : {os.path.exists('data/labels/map_index.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
