"""Step S0/S1: environment report, map inventory, inflation previews."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pathsafe_ellipse.config import load_config, project_paths, resolve_path, setup_logging  # noqa: E402
from pathsafe_ellipse.map_io import build_map_catalog, discover_map_files, map_group_id  # noqa: E402
from pathsafe_ellipse.map_preprocess import preprocess_map  # noqa: E402


def environment_report() -> dict:
    import cvxpy
    import scipy

    info = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "cvxpy": cvxpy.__version__,
        "cvxpy_solvers": sorted(cvxpy.installed_solvers()),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["torch_device"] = torch.cuda.get_device_name(0)
    except Exception:  # pragma: no cover - torch is optional for data generation
        info["torch"] = "not installed"
    try:
        import clarabel

        info["clarabel"] = clarabel.__version__
    except Exception:  # pragma: no cover
        info["clarabel"] = "not installed"
    try:
        import scs

        info["scs"] = scs.__version__
    except Exception:  # pragma: no cover
        info["scs"] = "not installed"
    try:
        import matplotlib

        info["matplotlib"] = matplotlib.__version__
    except Exception:  # pragma: no cover
        info["matplotlib"] = "not installed"
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect maps: inventory, inflation, metadata")
    parser.add_argument("--config", default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--max-maps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg, "inspect_maps")
    pp = project_paths(cfg)

    env = environment_report()
    print("== environment ==")
    for key, value in env.items():
        print(f"  {key}: {value}")

    files = discover_map_files(pp.map_dir)
    print(f"\n== maps ==\n  found {len(files)} .map files under {pp.map_dir}")
    if not files:
        print("ERROR: no .map files found")
        return 2

    records = build_map_catalog(pp.map_dir, cfg["map"]["resolution"])
    if args.max_maps:
        records = records[: args.max_maps]

    bad = [r for r in records if not r.parsed_ok]
    for rec in bad:
        print(f"  PARSE ERROR: {rec.file_name}: {rec.error}")

    rows = [r.as_dict() for r in records]
    inventory_csv = os.path.join(pp.reports_dir, "map_inventory.csv")
    with open(inventory_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    groups: dict[str, list[str]] = {}
    for rec in records:
        if rec.parsed_ok:
            groups.setdefault(rec.group_id, []).append(rec.file_name)

    sizes = {(r.width, r.height) for r in records if r.parsed_ok}
    resolutions = sorted({r.resolution for r in records if r.parsed_ok})
    ratios = np.array([r.obstacle_ratio for r in records if r.parsed_ok], dtype=float)
    summary = {
        "map_dir": pp.map_dir,
        "n_files": len(files),
        "n_parsed_ok": sum(1 for r in records if r.parsed_ok),
        "n_parse_errors": len(bad),
        "header_sizes": sorted([list(s) for s in sizes]),
        "resolutions": resolutions,
        "n_groups": len(groups),
        "groups": {k: sorted(v) for k, v in sorted(groups.items())},
        "group_sizes": {k: len(v) for k, v in sorted(groups.items())},
        "obstacle_ratio_mean": float(ratios.mean()) if ratios.size else None,
        "obstacle_ratio_min": float(ratios.min()) if ratios.size else None,
        "obstacle_ratio_max": float(ratios.max()) if ratios.size else None,
        "provided_split_counts": {
            tag: sum(1 for r in records if r.provided_split == tag)
            for tag in ("train", "val", "test", "unknown")
        },
        "environment": env,
    }
    summary_path = os.path.join(pp.reports_dir, "map_inventory.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(f"\n  parsed ok: {summary['n_parsed_ok']} / {len(records)}")
    print(f"  header sizes: {summary['header_sizes']}  resolutions: {resolutions}")
    print(f"  map groups: {len(groups)}")
    print(f"  obstacle ratio: mean={summary['obstacle_ratio_mean']:.3f} "
          f"min={summary['obstacle_ratio_min']:.3f} max={summary['obstacle_ratio_max']:.3f}")
    print(f"  provided folder tags: {summary['provided_split_counts']}")
    print(f"  wrote {inventory_csv}")
    print(f"  wrote {summary_path}")

    # Inflate every map once and cache it; verify idempotency of the cache.
    print("\n== inflation ==")
    robot = cfg["robot"]
    kernel_radius = int(robot["inflation_kernel"])
    iterations = int(robot["inflation_iterations"])
    for idx, rec in enumerate(records):
        if not rec.parsed_ok:
            continue
        inflated = preprocess_map(
            rec, pp.map_dir, pp.inflated_maps_dir,
            robot_radius=float(robot["radius"]),
            kernel_radius=kernel_radius,
            iterations=iterations,
            use_cache=not args.no_cache,
        )
        if idx < 3 or (idx + 1) % 40 == 0:
            extra = int(np.count_nonzero(inflated.occupancy_inflated)) - int(np.count_nonzero(inflated.occupancy_raw))
            print(f"  [{idx + 1}/{len(records)}] {rec.file_name}: "
                  f"{inflated.occupancy_raw.shape} added_obstacle_cells={extra}")
    print(f"  cached inflated maps -> {pp.inflated_maps_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
