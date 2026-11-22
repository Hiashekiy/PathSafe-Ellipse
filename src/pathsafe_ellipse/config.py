"""Config loading + path resolution helpers."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def resolve_path(path: str) -> str:
    """Resolve a config path relative to the project root."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(PROJECT_ROOT, path))


def load_config(path: str | None = None) -> Dict[str, Any]:
    cfg_path = path or os.path.join(PROJECT_ROOT, "configs", "data_generation.yaml")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = cfg_path
    return cfg


@dataclass(frozen=True)
class ProjectPaths:
    """Concrete, absolute directories used by the pipeline."""

    map_dir: str
    processed_dir: str
    map_metadata_dir: str
    inflated_maps_dir: str
    intermediate_dir: str
    paths_dir: str
    samples_dir: str
    labels_dir: str
    splits_dir: str
    reports_dir: str
    outputs_dir: str
    previews_dir: str
    path_viz_dir: str
    ellipse_viz_dir: str
    logs_dir: str

    def ensure(self) -> "ProjectPaths":
        for value in self.__dict__.values():
            os.makedirs(value, exist_ok=True)
        return self


def project_paths(cfg: Dict[str, Any]) -> ProjectPaths:
    p = cfg["paths"]
    processed = resolve_path(p["processed_dir"])
    intermediate = resolve_path(p["intermediate_dir"])
    outputs = resolve_path(p["outputs_dir"])
    paths = ProjectPaths(
        map_dir=resolve_path(p["map_dir"]),
        processed_dir=processed,
        map_metadata_dir=os.path.join(processed, "map_metadata"),
        inflated_maps_dir=os.path.join(processed, "inflated_maps"),
        intermediate_dir=intermediate,
        paths_dir=os.path.join(intermediate, "paths"),
        samples_dir=os.path.join(intermediate, "samples"),
        labels_dir=resolve_path(p["labels_dir"]),
        splits_dir=resolve_path(p["splits_dir"]),
        reports_dir=resolve_path(p["reports_dir"]),
        outputs_dir=outputs,
        previews_dir=os.path.join(outputs, "previews"),
        path_viz_dir=os.path.join(outputs, "path_visualizations"),
        ellipse_viz_dir=os.path.join(outputs, "ellipse_visualizations"),
        logs_dir=resolve_path(p["logs_dir"]),
    )
    return paths.ensure()


def setup_logging(cfg: Dict[str, Any], log_name: str) -> None:
    import logging

    level = getattr(logging, str(cfg.get("runtime", {}).get("log_level", "INFO")).upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().setLevel(level)
