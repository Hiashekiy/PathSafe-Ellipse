"""Scene-slot selection shared by the collection scripts (map + crop + field).

A slot is one 256x256 training scene: a source ``.map``, a random crop inside it
and the physical field of view (48 / 64 / 80 m) that fixes ``r_scene``.
"""
from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import yaml

from ..map_io import MapRecord, map_group_id

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def load_cfg(path: str) -> Dict:
    """Load a YAML config, resolving ``./`` paths against the project root."""
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for key, value in list(cfg.get("paths", {}).items()):
        if isinstance(value, str) and value.startswith("./"):
            cfg["paths"][key] = os.path.normpath(os.path.join(PROJECT_ROOT, value))
    cfg["_config_path"] = os.path.abspath(path)
    return cfg


def pick_scenes(records: List[MapRecord], n: int, seed: int) -> List[MapRecord]:
    """Stratified by map group and filename scale to avoid near-duplicates."""
    rng = np.random.default_rng(seed)
    by_group: Dict[str, List[MapRecord]] = {}
    for rec in records:
        by_group.setdefault(map_group_id(rec.file_name), []).append(rec)
    groups = sorted(by_group)
    rng.shuffle(groups)
    out: List[MapRecord] = []
    gi = 0
    while len(out) < n and groups:
        g = groups[gi % len(groups)]
        pool = by_group[g]
        rec = pool[int(rng.integers(0, len(pool)))]
        if all(r.file_name != rec.file_name for r in out):
            out.append(rec)
        gi += 1
        if gi > 40 * n:
            break
    return out


def build_slots(records: List[MapRecord], n_scenes: int, crops_per_map: int, seed: int,
                fields_m: List[float]) -> List[Dict[str, object]]:
    """Map/crop/field slots; several crops per map grow the dataset without repeats."""
    records = sorted(records, key=lambda r: r.file_name)
    if crops_per_map <= 1:
        picked = pick_scenes(records, min(n_scenes, len(records)), seed)
        return [{"rec": r, "crop": 0, "field_m": float(fields_m[i % len(fields_m)]),
                 "seed": int(seed + i)} for i, r in enumerate(picked)]
    slots: List[Dict[str, object]] = []
    k = 0
    while len(slots) < n_scenes:
        for rec in records:
            for c in range(crops_per_map):
                slots.append({
                    "rec": rec, "crop": c,
                    "field_m": float(fields_m[(k + c) % len(fields_m)]),
                    "seed": int(seed + 977 * k + 131 * c),
                })
                if len(slots) >= n_scenes:
                    break
            k += 1
            if len(slots) >= n_scenes:
                break
    return slots
