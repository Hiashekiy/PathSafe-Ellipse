"""Cached map access: parsing ``.map`` text costs 1-3 s per file.

The collector visits the same 180 source maps thousands of times, and every shard
process also parses the whole catalogue at start-up.  Both are memoised on disk:

    <cache_dir>/maps/<rel_path flattened>.npz      parsed occupancy + header
    <cache_dir>/map_catalog.json                   MapRecord list

Cache entries carry the source file size and mtime, so an edited map is
re-parsed automatically.  Writes are atomic (tmp + replace).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np

from ..map_io import (MapRecord, build_map_catalog, discover_map_files,
                      load_moving_ai_map)


def _slug(rel_path: str) -> str:
    return rel_path.replace("\\", "/").replace("/", "__").removesuffix(".map")


def map_cache_file(cache_dir: str, rel_path: str) -> str:
    return os.path.join(cache_dir, "maps", f"{_slug(rel_path)}.npz")


def load_map_cached(map_dir: str, rel_path: str,
                    cache_dir: str) -> Tuple[np.ndarray, Dict[str, object]]:
    """Parsed ``(occupancy uint8, header dict)`` of one source map, memoised."""
    src = os.path.join(map_dir, rel_path)
    st = os.stat(src)
    dst = map_cache_file(cache_dir, rel_path)
    if os.path.exists(dst):
        try:
            with np.load(dst, allow_pickle=False) as z:
                payload = json.loads(str(z["meta_json"].item()))
                if (payload.get("source_size") == st.st_size
                        and payload.get("source_mtime") == int(st.st_mtime)):
                    return z["occ"], dict(payload["map"])
        except Exception:                     # stale or unreadable -> re-parse
            pass
    occ, meta = load_moving_ai_map(src)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = f"{dst[:-4]}.{os.getpid()}.tmp.npz"
    payload = {"source_size": st.st_size, "source_mtime": int(st.st_mtime), "map": meta}
    np.savez_compressed(tmp, occ=occ,
                        meta_json=np.asarray(json.dumps(payload), dtype="U"))
    os.replace(tmp, dst)
    return occ, meta


def load_catalog_cached(map_dir: str, default_resolution: float,
                        cache_dir: str) -> List[MapRecord]:
    """``build_map_catalog`` with the result memoised (it parses every map)."""
    dst = os.path.join(cache_dir, "map_catalog.json")
    files = discover_map_files(map_dir)
    newest = max((os.path.getmtime(f) for f in files), default=0.0)
    if os.path.exists(dst) and os.path.getmtime(dst) >= newest:
        try:
            with open(dst, encoding="utf-8") as fh:
                rows = json.load(fh)
            if len(rows) == len(files):
                return [MapRecord(**r) for r in rows]
        except Exception:
            pass
    records = build_map_catalog(map_dir, default_resolution)
    os.makedirs(cache_dir, exist_ok=True)
    tmp = f"{dst}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump([r.as_dict() for r in records], fh)
    os.replace(tmp, dst)
    return records
