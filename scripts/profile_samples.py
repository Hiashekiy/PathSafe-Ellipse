"""Profile the per-point label pipeline on one map (dev tool)."""
from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pathsafe_ellipse.config import load_config, project_paths  # noqa: E402
from pathsafe_ellipse.dataset_builder import build_samples_for_map, get_solver, load_inflated  # noqa: E402
from pathsafe_ellipse.map_io import build_map_catalog, map_key  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-paths", type=int, default=4)
    ap.add_argument("--map", default=None)
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    pp = project_paths(cfg)
    records = [r for r in build_map_catalog(pp.map_dir, cfg["map"]["resolution"]) if r.parsed_ok]
    rec = None
    for r in records:
        if args.map and args.map not in r.file_name:
            continue
        if os.path.exists(os.path.join(pp.paths_dir, f"{map_key(r)}.npz")):
            rec = r
            break
    if rec is None:
        print("no cached paths found")
        return 2
    key = map_key(rec)
    dirs = {"map_dir": pp.map_dir, "inflated_dir": pp.inflated_maps_dir,
            "paths_dir": pp.paths_dir, "samples_dir": pp.samples_dir}
    inflated, raw = load_inflated(rec, cfg, dirs)
    npz = np.load(os.path.join(pp.paths_dir, f"{key}.npz"))
    flat, offs = npz["flat_paths"], npz["path_offsets"]
    paths = [flat[offs[i]:offs[i + 1]] for i in range(len(offs) - 1)][: args.n_paths]
    solver = get_solver(cfg)
    print(f"map={key} paths={len(paths)} points~{sum(len(p) for p in paths)}")

    t0 = time.time()
    if args.profile:
        pr = cProfile.Profile()
        pr.enable()
        arrays, info = build_samples_for_map(inflated, raw, {"paths": paths}, cfg, solver, 0)
        pr.disable()
        dt = time.time() - t0
        st = pstats.Stats(pr)
        st.sort_stats("cumulative").print_stats(18)
    else:
        arrays, info = build_samples_for_map(inflated, raw, {"paths": paths}, cfg, solver, 0)
        dt = time.time() - t0
    n = len(arrays["label_valid"])
    print(f"\nsamples={n} elapsed={dt:.1f}s -> {dt / max(n,1)*1e3:.1f} ms/sample")
    print("candidate_stats:", info["candidate_stats"])
    print("reasons:", info["reasons"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
