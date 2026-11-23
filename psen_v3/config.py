"""Config loading for ``psen_v3`` (yaml in, plain dict out, paths resolved)."""
from __future__ import annotations

import os
from typing import Dict, Optional

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "psen_v3", "configs", "v3.yaml")


def _resolve(value: str) -> str:
    return os.path.normpath(os.path.join(PROJECT_ROOT, value))


def load_config(path: Optional[str] = None, **overrides) -> Dict:
    """Load the yaml config; ``./...`` paths are resolved against the project root.

    Keyword overrides are applied as ``section.key=value`` via ``overrides``
    (``load_config(path, train_batch_size=8)`` -> ``cfg['train']['batch_size']=8``).
    """
    path = path or DEFAULT_CONFIG
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for key, value in list(cfg["data"].items()):
        if isinstance(value, str) and value.startswith("./"):
            cfg["data"][key] = _resolve(value)
    out_dir = cfg["train"].get("out_dir", "")
    if isinstance(out_dir, str) and out_dir.startswith("./"):
        cfg["train"]["out_dir"] = _resolve(out_dir)
    for k, v in overrides.items():
        if "__" not in k:
            raise KeyError(f"override '{k}' must look like section__key")
        section, key = k.split("__", 1)
        if section not in cfg:
            raise KeyError(f"unknown config section '{section}'")
        cfg[section][key] = v
    return cfg


def git_commit() -> str:
    """Current commit (best effort, '' when git is unavailable)."""
    import subprocess

    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip()
    except Exception:
        return ""
