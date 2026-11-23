"""Task 2 - build the signed distance field data version (never touches V3 data).

For every *unique* blocked map of the V3 trajectory dataset this writes

    data/processed_v3sdf/sdf/<map_key>.npz
        signed_distance_m (H,W) float32   metres, signed, cell-centre aligned
        blocked           (H,W) uint8      the exact map the SDF was derived from
        valid_mask        (H,W) uint8      1 = pixel comes from the source map
        resolution_m      scalar

plus ``index.jsonl`` (one line per trajectory NPZ, joining it to its map) and
``manifest.json`` (generation parameters, sign/unit statistics, verification).

The V3 trajectory NPZ files are neither modified nor copied.

    python scripts/v3_build_sdf.py --src data/processed_v3/trajectories/train_v3 \
        --out data/processed_v3sdf --workers 12 --verify 40
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from psen_v3.data.sdf import build_and_check, map_key, signed_distance_m  # noqa: E402

SDF_VERSION = "v3sdf-1"


def read_header(path: str) -> Dict[str, object]:
    """Map key + the meta fields the index needs (one NPZ read)."""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"].item()))
        occ = np.asarray(z["occupancy"], np.uint8)
        n = int(np.asarray(z["path_xy"]).shape[0])
    res = float(meta["r_scene"])
    return {"map_key": map_key(occ, res), "resolution_m": res,
            "crop_px": int(meta.get("crop_px", -1)),
            "scene_id": str(meta.get("scene_id", "")),
            "map_group": str(meta.get("map_group", "")),
            "path_id": int(meta.get("path_id", -1)),
            "n_points": n}


def read_occ(path: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        return np.asarray(z["occupancy"], np.uint8)


def valid_mask_all_ones(occ: np.ndarray) -> np.ndarray:
    """Coverage mask of the scene resampling, stored explicitly.

    ``scene.conservative_resize`` maps every source cell onto a target cell (area
    max-pool downwards, nearest neighbour upwards), so no target cell is ever left
    uncovered for both ``crop_px >= 256`` and ``crop_px <= 256`` (the audit checks
    this empirically): the mask is all ones, and the network never has to guess.
    """
    return np.ones_like(occ, dtype=np.uint8)


def write_map(out_dir: str, key: str, occ: np.ndarray, res: float) -> Dict[str, object]:
    path = os.path.join(out_dir, "sdf", f"{key}.npz")
    sdf, stats = build_and_check(occ, res)
    if not os.path.exists(path):
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp,
                            signed_distance_m=sdf,
                            blocked=np.asarray(occ, np.uint8),
                            valid_mask=valid_mask_all_ones(occ),
                            resolution_m=np.float64(res))
        os.replace(tmp, path)
    stats["bytes"] = os.path.getsize(path)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed_v3/trajectories/train_v3")
    ap.add_argument("--out", default="data/processed_v3sdf")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verify", type=int, default=40,
                    help="re-read N random stored maps and compare with a rebuild")
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    out_dir = os.path.abspath(args.out)
    os.makedirs(os.path.join(out_dir, "sdf"), exist_ok=True)
    files = sorted(os.path.join(src, n) for n in os.listdir(src) if n.endswith(".npz"))
    if args.limit:
        files = files[: args.limit]
    print(f"{len(files)} trajectory NPZ in {src}", flush=True)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        heads: List[Dict[str, object]] = list(ex.map(read_header, files, chunksize=64))
    t_hdr = time.perf_counter() - t0
    print(f"headers read in {t_hdr:.1f}s", flush=True)

    rep: Dict[str, str] = {}
    res_of: Dict[str, float] = {}
    for f, h in zip(files, heads):
        key = str(h["map_key"])
        rep.setdefault(key, f)
        res_of[key] = float(h["resolution_m"])
    print(f"unique maps {len(rep)}", flush=True)

    t_map = time.perf_counter()
    stats: List[Dict[str, object]] = []
    for i, (key, f) in enumerate(rep.items()):
        stats.append(write_map(out_dir, key, read_occ(f), res_of[key]))
        if (i + 1) % 4000 == 0:
            print(f"  {i + 1}/{len(rep)} maps", flush=True)
    build_s = time.perf_counter() - t_map
    print(f"SDF written in {build_s:.1f}s", flush=True)

    index_path = os.path.join(out_dir, "index.jsonl")
    with open(index_path, "w", encoding="utf-8") as fh:
        for f, h in zip(files, heads):
            fh.write(json.dumps({
                "trajectory": os.path.relpath(f, src).replace("\\", "/"),
                "map_key": h["map_key"],
                "resolution_m": h["resolution_m"],
                "scene_id": h["scene_id"],
                "map_group": h["map_group"],
                "path_id": h["path_id"],
                "n_points": h["n_points"],
            }, ensure_ascii=False) + "\n")
    print(f"index -> {index_path} ({os.path.getsize(index_path) / 1e6:.1f} MB)", flush=True)

    rng = np.random.default_rng(0)
    rep_list = list(rep.items())
    verify_idx = rng.choice(len(rep_list), size=min(args.verify, len(rep_list)), replace=False)
    v_err, v_mask_ok = [], 0
    for j in verify_idx:
        key = rep_list[int(j)][0]
        with np.load(os.path.join(out_dir, "sdf", f"{key}.npz"), allow_pickle=False) as z:
            stored = np.asarray(z["signed_distance_m"], np.float64)
            blocked = np.asarray(z["blocked"], np.uint8)
            res = float(z["resolution_m"])
            mask = np.asarray(z["valid_mask"], np.uint8)
        v_err.append(float(np.abs(stored - signed_distance_m(blocked, res)).max()))
        v_mask_ok += int(bool(mask.all()))

    negatives = np.asarray([s["negative_fraction"] for s in stats], np.float64)
    manifest = {
        "sdf_version": SDF_VERSION,
        "source_trajectory_dir": src,
        "n_trajectory_files": len(files),
        "n_unique_maps": len(rep),
        "resolution_m_variants": sorted({round(v, 6) for v in res_of.values()}),
        "convention": {
            "free": "+distance to the nearest blocked cell area [m]",
            "blocked": "-distance to the nearest free cell area [m]",
            "outside_map": "blocked (border padded as blocked)",
            "half_cell_offset": 0.5,
            "cell_centre": "cell (row=i, col=j) sits at (x, y) = (j*r, i*r)",
            "sdf_outside_value_m": -1.0,
        },
        "sdf_stats": {
            "min": float(min(s["min"] for s in stats)),
            "max": float(max(s["max"] for s in stats)),
            "sign_mismatch_max": float(max(s["sign_mismatch_fraction"] for s in stats)),
            "negative_fraction": {"mean": float(negatives.mean()),
                                  "min": float(negatives.min()),
                                  "max": float(negatives.max())},
            "total_bytes": int(sum(int(s["bytes"]) for s in stats)),
        },
        "verify": {"n": len(v_err),
                   "max_abs_err_m": float(max(v_err) if v_err else 0.0),
                   "valid_mask_all_ones": v_mask_ok},
        "timing_s": {"headers": round(t_hdr, 1), "sdf_build": round(build_s, 1)},
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False)[:1600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
