"""Map-group / path-group split construction (section 10.2).

Samples of one map group always land in the same split, and samples of one path
never straddle splits, so adjacent tiles and neighbouring path points cannot
leak between train and test.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np


def group_split(
    groups: Sequence[str],
    group_weights: Dict[str, float],
    ratios: Dict[str, float],
    seed: int,
) -> Dict[str, List[str]]:
    """Assign whole map groups to train/val/test, matching the target ratios."""
    names = sorted(set(groups))
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(names)))
    shuffled = [names[i] for i in order]
    total = sum(group_weights[g] for g in shuffled) or 1.0

    target = {"train": float(ratios["train_ratio"]), "val": float(ratios["val_ratio"]),
              "test": float(ratios["test_ratio"])}
    accum = {"train": 0.0, "val": 0.0, "test": 0.0}
    assignment: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for g in shuffled:
        share = group_weights[g] / total
        # greedy: put the group into the split that is furthest below its target
        deficits = {k: target[k] - accum[k] for k in accum}
        split = max(deficits, key=deficits.get)
        assignment[split].append(g)
        accum[split] += share

    for split in assignment:
        assignment[split] = sorted(assignment[split])
    return assignment


def path_group_split(path_keys: Sequence[Tuple[str, int]], weights: Dict[str, float],
                     seed: int) -> Dict[str, List[str]]:
    """Fallback split when there are too few map groups: keep paths together."""
    keys = sorted(set(path_keys))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(keys))
    n_val = max(1, int(round(len(keys) * weights["val_ratio"])))
    n_test = max(1, int(round(len(keys) * weights["test_ratio"])))
    val = [keys[i] for i in idx[:n_val]]
    test = [keys[i] for i in idx[n_val:n_val + n_test]]
    train = [keys[i] for i in idx[n_val + n_test:]]
    return {
        "train": sorted(f"{m}#{p}" for m, p in train),
        "val": sorted(f"{m}#{p}" for m, p in val),
        "test": sorted(f"{m}#{p}" for m, p in test),
    }


def save_split(split: Dict[str, List[str]], path: str, meta: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"split": split, "meta": meta}, fh, indent=2, ensure_ascii=False)


def load_split(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)["split"]
