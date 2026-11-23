"""SDF sign / unit / boundary tests on hand-built maps (plan section 12.3 first item).

    python tests/test_sdf.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ellipsenet.data.sdf import SDF_OUTSIDE_M, build_and_check, map_key, signed_distance_m

FAILS = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}{(' :: ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def test_empty_map() -> None:
    """No obstacle inside: every value is the distance to the map border."""
    n, res = 8, 1.0
    sdf = signed_distance_m(np.zeros((n, n), np.uint8), res)
    check("empty map: all positive", bool((sdf > 0).all()),
          f"min {sdf.min():.3f}")
    # centre cell (3,3): nearest blocked ring centre is 4 cells away -> (4-0.5)*res
    check("empty map: centre == (4-0.5)*res", abs(sdf[3, 3] - 3.5) < 1e-6,
          f"{sdf[3, 3]:.4f} vs 3.5")
    check("empty map: corner == (1-0.5)*res", abs(sdf[0, 0] - 0.5) < 1e-6,
          f"{sdf[0, 0]:.4f} vs 0.5")


def test_single_wall() -> None:
    """One blocked column: distances on both sides match, wall cells are negative."""
    n, res = 9, 0.5
    occ = np.zeros((n, n), np.uint8)
    occ[:, 4] = 1
    sdf = signed_distance_m(occ, res)
    check("wall: symmetric", abs(sdf[4, 2] - sdf[4, 6]) < 1e-6,
          f"{sdf[4, 2]:.4f} vs {sdf[4, 6]:.4f}")
    check("wall: 2 cells away == (2-0.5)*res", abs(sdf[4, 2] - 0.75) < 1e-6,
          f"{sdf[4, 2]:.4f} vs 0.75")
    check("wall: adjacent free == 0.5*res", abs(sdf[4, 3] - 0.25) < 1e-6,
          f"{sdf[4, 3]:.4f} vs 0.25")
    check("wall: wall cell negative", bool((sdf[:, 4] < 0).all()),
          f"max {sdf[:, 4].max():.4f}")
    check("wall: wall cell == -0.5*res", abs(sdf[4, 4] + 0.25) < 1e-6,
          f"{sdf[4, 4]:.4f} vs -0.25")
    check("wall: sign matches blocked mask",
          bool(((sdf < 0) == occ.astype(bool)).all()))


def test_narrow_corridor() -> None:
    """Two parallel walls 3 free cells apart, metric units."""
    res = 0.25
    occ = np.zeros((12, 12), np.uint8)
    occ[:, 5] = 1
    occ[:, 9] = 1
    sdf = signed_distance_m(occ, res)
    corridor = sdf[6, 6:9]
    check("corridor: free cells positive", bool((corridor > 0).all()),
          str(np.round(corridor, 3)))
    check("corridor: centre of 3-free-cell gap == (2-0.5)*res",
          abs(sdf[6, 7] - 0.375) < 1e-6, f"{sdf[6, 7]:.4f} vs 0.375")
    check("corridor: walls negative", bool((sdf[:, [5, 9]] < 0).all()))


def test_map_border() -> None:
    """A free cell next to the border sees the border as a blocked wall."""
    res = 0.5
    occ = np.zeros((10, 10), np.uint8)
    sdf = signed_distance_m(occ, res)
    check("border: first row == 0.5*res", abs(sdf[0, 5] - 0.25) < 1e-6,
          f"{sdf[0, 5]:.4f} vs 0.25")
    check("border: no NaN/Inf", bool(np.isfinite(sdf).all()))


def test_all_blocked() -> None:
    sdf = signed_distance_m(np.ones((6, 6), np.uint8), 0.25)
    check("all blocked: never positive", bool((sdf <= 0).all()), f"max {sdf.max():.3f}")


def test_monotone_and_stats() -> None:
    """Approaching an obstacle strictly decreases the free-space value.

    Rows 26..38 only: closer to the wall the field must grow by exactly one cell
    step (0.25 m); the map border and the wall symmetric side are checked apart.
    """
    res = 0.25
    occ = np.zeros((48, 48), np.uint8)
    occ[40, :] = 1
    sdf = signed_distance_m(occ, res)
    column = sdf[26:39, 24][::-1]               # rows 38..26, from the wall upwards
    check("monotone: grows by exactly one cell step",
          bool(np.allclose(np.diff(column), res, atol=1e-6)),
          str(np.round(np.diff(column), 4)))
    _, stats = build_and_check(occ, res)
    check("stats: sign == blocked mask",
          stats["sign_mismatch_fraction"] == 0.0, str(stats["sign_mismatch_fraction"]))
    check("stats: resolution recorded", stats["resolution_m"] == res)


def test_key_stability() -> None:
    occ = np.zeros((8, 8), np.uint8)
    occ[3, 3] = 1
    k1 = map_key(occ, 0.25)
    k2 = map_key(occ.copy(), 0.25)
    occ2 = occ.copy()
    occ2[5, 5] = 1
    check("key: identical maps share a key", k1 == k2, k1)
    check("key: different maps differ", k1 != map_key(occ2, 0.25))


def test_outside_constant() -> None:
    check("outside constant is negative (blocked)", SDF_OUTSIDE_M < 0.0, str(SDF_OUTSIDE_M))


def main() -> int:
    for fn in (test_empty_map, test_single_wall, test_narrow_corridor, test_map_border,
               test_all_blocked, test_monotone_and_stats, test_key_stability,
               test_outside_constant):
        print(fn.__name__)
        fn()
    print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
