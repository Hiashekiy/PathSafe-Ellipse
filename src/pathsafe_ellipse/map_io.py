"""Moving AI ``.map`` parsing with strict header / body validation."""
from __future__ import annotations

import glob
import hashlib
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np

# Characters treated as blocked, matching Neural-IRIS' load_moving_ai_map().
OBSTACLE_CHARS = {"@", "O", "T", "W"}
FREE_CHARS = {".", "G", "S"}


class MapParseError(RuntimeError):
    """Raised when a ``.map`` file cannot be parsed unambiguously."""


@dataclass
class MapRecord:
    file_name: str
    rel_path: str
    width: int
    height: int
    resolution: float
    map_type: str
    free_cells: int
    obstacle_cells: int
    obstacle_ratio: float
    md5: str
    group_id: str
    provided_split: str
    parsed_ok: bool
    error: str = ""

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def discover_map_files(map_dir: str) -> List[str]:
    """Recursively find every ``.map`` file below ``map_dir`` (sorted, deduped)."""
    pattern = os.path.join(map_dir, "**", "*.map")
    files = sorted({os.path.normpath(p) for p in glob.glob(pattern, recursive=True)})
    return files


def map_group_id(file_name: str) -> str:
    """Group id for split construction.

    ``Berlin_1_1024_part_0_1.map`` -> ``Berlin_1``: all ``_part_`` tiles of the
    same mother map share one group, so adjacent tiles never straddle splits.
    """
    stem = os.path.splitext(os.path.basename(file_name))[0]
    head = stem.split("_part_")[0]
    tokens = head.split("_")
    # Drop a trailing tile-size token (256 / 512 / 1024 / ...) if present.
    if len(tokens) > 1 and tokens[-1].isdigit() and int(tokens[-1]) >= 128:
        tokens = tokens[:-1]
    return "_".join(tokens) if tokens else stem


def provided_split_tag(rel_path: str) -> str:
    parts = rel_path.replace("\\", "/").split("/")
    for tag in ("train", "val", "test"):
        if tag in parts[:-1]:
            return tag
    return "unknown"


def load_moving_ai_map(filepath: str) -> Tuple[np.ndarray, Dict[str, object]]:
    """Parse a Moving AI map into a ``uint8`` occupancy grid (0 free / 1 obstacle).

    Raises :class:`MapParseError` with the offending file name on any
    inconsistency instead of silently truncating.
    """
    with open(filepath, "r", encoding="utf-8", errors="strict") as fh:
        lines = fh.read().splitlines()

    height = width = None
    resolution = None
    map_type = "octile"
    map_start = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        key, _, rest = stripped.partition(" ")
        if key == "height":
            height = int(rest)
        elif key == "width":
            width = int(rest)
        elif key == "resolution":
            resolution = float(rest)
        elif key == "cell_length_m":
            resolution = float(rest)
        elif key == "type":
            map_type = rest.strip()
        elif key == "map":
            map_start = idx + 1
            break

    if map_start is None:
        raise MapParseError(f"{filepath}: no 'map' header line found")
    if height is None or width is None:
        raise MapParseError(f"{filepath}: missing height/width header")
    if height <= 0 or width <= 0:
        raise MapParseError(f"{filepath}: non-positive height/width ({height}x{width})")

    body = lines[map_start:map_start + height]
    if len(body) < height:
        raise MapParseError(
            f"{filepath}: header declares height={height} but only {len(body)} map rows present"
        )

    grid = np.zeros((height, width), dtype=np.uint8)
    for row, line in enumerate(body):
        if len(line) != width:
            raise MapParseError(
                f"{filepath}: row {row} has length {len(line)}, header width={width}"
            )
        unknown = set(line) - OBSTACLE_CHARS - FREE_CHARS
        if unknown:
            raise MapParseError(f"{filepath}: row {row} contains unknown chars {sorted(unknown)}")
        obstacle = np.fromiter((c in OBSTACLE_CHARS for c in line), dtype=bool, count=width)
        grid[row] = obstacle.astype(np.uint8)

    meta = {
        "width": width,
        "height": height,
        "resolution": resolution,
        "map_type": map_type,
    }
    return grid, meta


def build_map_catalog(map_dir: str, default_resolution: float) -> List[MapRecord]:
    """Parse every map under ``map_dir`` and return one metadata record per file."""
    records: List[MapRecord] = []
    for filepath in discover_map_files(map_dir):
        rel = os.path.relpath(filepath, map_dir)
        with open(filepath, "rb") as fh:
            md5 = hashlib.md5(fh.read()).hexdigest()
        name = os.path.basename(filepath)
        try:
            grid, meta = load_moving_ai_map(filepath)
        except (MapParseError, ValueError) as exc:
            records.append(
                MapRecord(
                    file_name=name, rel_path=rel, width=0, height=0,
                    resolution=default_resolution, map_type="unknown",
                    free_cells=0, obstacle_cells=0, obstacle_ratio=float("nan"),
                    md5=md5, group_id=map_group_id(name),
                    provided_split=provided_split_tag(rel), parsed_ok=False, error=str(exc),
                )
            )
            continue
        free = int(np.count_nonzero(grid == 0))
        obstacle = int(grid.size - free)
        records.append(
            MapRecord(
                file_name=name,
                rel_path=rel,
                width=int(meta["width"]),
                height=int(meta["height"]),
                resolution=float(meta["resolution"] or default_resolution),
                map_type=str(meta["map_type"]),
                free_cells=free,
                obstacle_cells=obstacle,
                obstacle_ratio=obstacle / float(grid.size),
                md5=md5,
                group_id=map_group_id(name),
                provided_split=provided_split_tag(rel),
                parsed_ok=True,
            )
        )
    return records


def map_key(record: MapRecord) -> str:
    """Stable identifier for a map (filename stem)."""
    return os.path.splitext(record.file_name)[0]
