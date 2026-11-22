"""End-to-end dataset generation (S2-S5): paths -> samples -> filter -> split -> NPZ."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pathsafe_ellipse.config import load_config, project_paths, setup_logging  # noqa: E402
from pathsafe_ellipse.dataset_builder import paths_worker, samples_worker  # noqa: E402
from pathsafe_ellipse.dataset_filter import filter_arrays  # noqa: E402
from pathsafe_ellipse.dataset_split import group_split, save_split  # noqa: E402
from pathsafe_ellipse.map_io import build_map_catalog, map_key  # noqa: E402

ARRAY_KEYS = (
    "map_patch", "ray_distances", "ray_hits", "path_tangent",
    "offset_direction", "offset_ratio", "ellipse_ab", "ellipse_angle",
    "ellipse_center_offset", "ellipse_area", "point_inside_metric", "angle_valid",
    "label_valid", "point_xy", "map_id", "path_id", "point_id", "clearance",
    "ray_distances_raw", "n_candidates", "n_candidates_valid", "ellipse_iters",
    "ellipse_converged", "clearance_class",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate the PathSafe-Ellipse dataset")
    p.add_argument("--config", default=None)
    p.add_argument("--max-maps", type=int, default=None, help="limit number of maps (dev runs)")
    p.add_argument("--paths-per-map", type=int, default=None, help="override paths_per_map")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--map-filter", default=None, help="comma separated substrings to keep")
    p.add_argument("--stage", default="all", choices=["paths", "samples", "assemble", "all"])
    p.add_argument("--shard", type=int, default=0, help="this shard index (filesystem sharding)")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--tag", default="", help="suffix for the split files (dev runs)")
    return p.parse_args()


def run_stage(payloads: List[Dict], worker, workers: int, label: str) -> List[Dict]:
    results: List[Dict] = []
    if workers > 1:
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(worker, p): p for p in payloads}
                done = 0
                for fut in as_completed(futures):
                    payload = futures[fut]
                    try:
                        results.append(fut.result())
                    except Exception as exc:  # keep going, report the broken map
                        print(f"  [{label}] FAILED {payload['record'].file_name}: {exc}", flush=True)
                        results.append({"map_key": map_key(payload["record"]), "error": str(exc)})
                    done += 1
                    if done % 10 == 0:
                        print(f"  [{label}] {done}/{len(payloads)} maps", flush=True)
            return results
        except (PermissionError, OSError) as exc:
            print(f"  [{label}] process pool unavailable ({exc}); falling back to serial", flush=True)
            results = []
    for i, payload in enumerate(payloads):
        try:
            results.append(worker(payload))
        except Exception as exc:
            print(f"  [{label}] FAILED {payload['record'].file_name}: {exc}", flush=True)
            traceback.print_exc()
            results.append({"map_key": map_key(payload["record"]), "error": str(exc)})
        if (i + 1) % 5 == 0:
            print(f"  [{label}] {i + 1}/{len(payloads)} maps", flush=True)
    return results


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg, "generate_dataset")
    pp = project_paths(cfg)
    rng_seed = int(args.seed if args.seed is not None else cfg["project"]["random_seed"])
    if args.paths_per_map is not None:
        cfg["path_search"]["paths_per_map"] = int(args.paths_per_map)
    workers = int(args.workers if args.workers is not None else cfg["runtime"]["num_workers"])

    records = [r for r in build_map_catalog(pp.map_dir, cfg["map"]["resolution"]) if r.parsed_ok]
    if args.map_filter:
        needles = [s.strip() for s in args.map_filter.split(",") if s.strip()]
        records = [r for r in records if any(n in r.file_name for n in needles)]
    if args.max_maps:
        records = records[: args.max_maps]
    records.sort(key=lambda r: r.file_name)
    if not records:
        print("ERROR: no maps selected")
        return 2
    if args.num_shards > 1:
        records = [r for i, r in enumerate(records) if i % args.num_shards == args.shard]
        print(f"shard {args.shard}/{args.num_shards}: {len(records)} maps")
    print(f"maps selected: {len(records)}   workers={workers}   "
          f"paths_per_map={cfg['path_search']['paths_per_map']}   seed={rng_seed}")

    dirs = {
        "map_dir": pp.map_dir,
        "inflated_dir": pp.inflated_maps_dir,
        "paths_dir": pp.paths_dir,
        "samples_dir": pp.samples_dir,
    }
    base_payload = {"cfg": cfg, "dirs": dirs}

    t0 = time.time()
    if args.stage == "assemble":
        # all per-map artefacts must already exist; go straight to filter/split
        keys = [map_key(r) for r in build_map_catalog(pp.map_dir, cfg["map"]["resolution"])]
        return assemble(cfg, pp, keys, group_map_from_catalog(pp, cfg), args, t0)

    # ---------------------------------------------------------------- paths
    path_payloads = []
    for i, rec in enumerate(records):
        key = map_key(rec)
        npz = os.path.join(pp.paths_dir, f"{key}.npz")
        if os.path.exists(npz) and not args.overwrite:
            continue
        path_payloads.append({**base_payload, "record": rec,
                              "seed": rng_seed + 1000 + i})
    if path_payloads:
        print(f"\n== S2: A* paths ({len(path_payloads)} maps) ==")
        path_results = run_stage(path_payloads, paths_worker, workers, "paths")
        with open(os.path.join(pp.reports_dir, "path_generation.json"), "w", encoding="utf-8") as fh:
            json.dump([r for r in path_results if "error" not in r], fh, indent=2)
        print(f"  paths written: {sum(r.get('n_paths', 0) for r in path_results)}")
    else:
        print("\n== S2: A* paths == (cache complete)")

    if args.stage == "paths":
        return 0

    # -------------------------------------------------------------- samples
    sample_payloads = []
    for i, rec in enumerate(records):
        key = map_key(rec)
        npz = os.path.join(pp.samples_dir, f"{key}.npz")
        if os.path.exists(npz) and not args.overwrite:
            continue
        if not os.path.exists(os.path.join(pp.paths_dir, f"{key}.npz")):
            print(f"  skip {key}: no cached paths")
            continue
        sample_payloads.append({**base_payload, "record": rec,
                                "seed": rng_seed + 2000 + i})
    sample_infos: List[Dict] = []
    if sample_payloads:
        print(f"\n== S3/S4: samples + ellipse labels ({len(sample_payloads)} maps) ==")
        sample_infos = run_stage(sample_payloads, samples_worker, workers, "samples")
        with open(os.path.join(pp.reports_dir, "sample_generation.json"), "w", encoding="utf-8") as fh:
            json.dump([r for r in sample_infos if "error" not in r], fh, indent=2)
        print(f"  elapsed {time.time() - t0:.1f}s")
    else:
        print("\n== S3/S4: samples == (cache complete)")

    if args.stage == "samples":
        return 0

    group_of = {map_key(r): r.group_id for r in records}
    keys = [map_key(r) for r in records]
    return assemble(cfg, pp, keys, group_of, args, t0)


def group_map_from_catalog(pp, cfg) -> Dict[str, str]:
    return {map_key(r): r.group_id
            for r in build_map_catalog(pp.map_dir, cfg["map"]["resolution"]) if r.parsed_ok}


def assemble(cfg, pp, keys: List[str], group_of: Dict[str, str],
             args, t0: float) -> int:
    """Filter per-map samples, build the group split and write train/val/test NPZ."""
    print("\n== S5: filter, split, NPZ ==")
    split_path = os.path.join(pp.splits_dir, f"map_group_split{args.tag}.json")

    per_map: Dict[str, Dict[str, np.ndarray]] = {}
    filter_reasons: Dict[str, int] = {}
    n_raw = 0
    for key in keys:
        npz = os.path.join(pp.samples_dir, f"{key}.npz")
        if not os.path.exists(npz):
            continue
        with np.load(npz) as data:
            arrays = {k: data[k] for k in data.files}
        n_raw += len(arrays["label_valid"])
        filtered, reasons, _ = filter_arrays(arrays, cfg)
        for k, v in reasons.items():
            filter_reasons[k] = filter_reasons.get(k, 0) + int(v)
        n_valid = len(filtered["label_valid"])
        if n_valid == 0:
            continue
        # keep only the fields that go into the split, with the patch bit-packed
        # (unpacked 128x128 patches would need ~16 kB x 350k samples in RAM)
        keep = {f: filtered[f] for f in ARRAY_KEYS if f in filtered and f != "map_patch"}
        keep["map_patch_packed"] = np.packbits(
            filtered["map_patch"].reshape(n_valid, 128, 128), axis=2)
        keep["map_id"] = np.full(n_valid, keys.index(key), dtype=np.int32)
        per_map[key] = keep
        del arrays, filtered

    if not per_map:
        print("ERROR: no valid samples produced")
        return 3

    weights = {g: float(sum(len(per_map[k]["label_valid"]) for k in per_map if group_of[k] == g))
               for g in sorted(set(group_of[k] for k in per_map))}
    split_cfg = {"train_ratio": cfg["dataset"]["train_ratio"],
                 "val_ratio": cfg["dataset"]["val_ratio"],
                 "test_ratio": cfg["dataset"]["test_ratio"]}
    if os.path.exists(split_path) and not args.overwrite:
        from pathsafe_ellipse.dataset_split import load_split
        split = load_split(split_path)
        print(f"  reusing existing split {split_path}")
    else:
        split = group_split(list(weights.keys()), weights, split_cfg,
                            int(cfg["dataset"]["split_seed"]))
        save_split(split, split_path, {
            "strategy": "map_group",
            "group_definition": "city_index (all _part_ tiles of one mother map stay together)",
            "reason": "avoid adjacent-tile leakage between train/val/test",
            "group_weights_samples": weights,
            "ratios": split_cfg,
            "seed": int(cfg["dataset"]["split_seed"]),
        })
        print(f"  wrote split {split_path}")

    split_of_group = {g: s for s, groups in split.items() for g in groups}
    outputs = {}
    fields = [f for f in ARRAY_KEYS if f != "map_patch"] + ["map_patch_packed"]
    for name in ("train", "val", "test"):
        groups = set(split.get(name, []))
        chosen = [k for k in per_map if group_of[k] in groups]
        if not chosen:
            print(f"  WARNING: split '{name}' has no samples")
            np.savez_compressed(os.path.join(pp.splits_dir, f"{name}.npz"))
            outputs[name] = {"n_samples": 0, "n_maps": 0, "n_paths": 0, "maps": [],
                             "file": os.path.join(pp.splits_dir, f"{name}.npz"), "size_mb": 0.0}
            continue
        merged = {field: np.concatenate([per_map[k][field] for k in chosen], axis=0)
                  for field in fields if field in per_map[chosen[0]]}
        # unpacked twin for downstream users that expect the documented 128x128 patch
        merged["map_patch"] = np.unpackbits(merged["map_patch_packed"], axis=2)
        out_npz = os.path.join(pp.splits_dir, f"{name}.npz")
        np.savez_compressed(out_npz, **merged)
        labels = merged.get("label_valid", np.zeros(0))
        outputs[name] = {
            "n_samples": int(len(labels)),
            "n_maps": len(chosen),
            "n_paths": int(len({(m, p) for m, p in zip(merged.get("map_id", []),
                                                       merged.get("path_id", []))})),
            "maps": sorted(chosen),
            "file": out_npz,
            "size_mb": round(os.path.getsize(out_npz) / 1e6, 2),
        }
        print(f"  {name}: {outputs[name]['n_samples']} samples, "
              f"{outputs[name]['n_maps']} maps, {outputs[name]['size_mb']} MB -> {out_npz}")
        del merged

    summary = {
        "n_maps_selected": len(keys),
        "n_maps_with_samples": len(per_map),
        "n_samples_raw": n_raw,
        "n_samples_filtered": int(sum(o["n_samples"] for o in outputs.values())),
        "filter_reasons": filter_reasons,
        "splits": outputs,
        "split_file": split_path,
        "paths_per_map": cfg["path_search"]["paths_per_map"],
        "seed": int(cfg["project"]["random_seed"]),
        "elapsed_s": round(time.time() - t0, 1),
        "coordinate_convention": "(x, y) = (col, row); patch centre = (64, 64)",
        "units": {"patch": "cells", "ellipse_ab": "m", "ellipse_center_offset": "m",
                  "ray_distances": "normalised by ray_max_range", "clearance": "cells"},
    }
    os.makedirs(pp.reports_dir, exist_ok=True)
    with open(os.path.join(pp.reports_dir, "generation_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    with open(os.path.join(pp.labels_dir, "map_index.json"), "w", encoding="utf-8") as fh:
        json.dump({i: k for i, k in enumerate(keys)}, fh, indent=2)
    print(f"\n  summary -> {os.path.join(pp.reports_dir, 'generation_summary.json')}")
    print(f"  total elapsed {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
