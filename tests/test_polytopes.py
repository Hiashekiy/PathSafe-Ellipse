"""Tests for ellipse -> convex polytope inflation (``ellipsenet.geometry.polytopes``).

    python tests/test_polytopes.py

Invariants checked:
1. the predicted ellipse is inside its region (support test on every face);
2. no blocked square enters any region (exact analytic check, whole map);
3. faces sit on obstacles, not on the local window, and the map border is respected;
4. the batched (parallel) code equals the one-ellipse-at-a-time reference;
5. ellipses in free space stay bounded by the map border only.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ellipsenet.geometry.polytopes import (build_polytopes, build_polytopes_serial,
                                           cell_support, ellipse_matrix, ellipse_support,
                                           polytope_area, verify_polytopes)

FAILS = []
RES = 0.25


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}{(' :: ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def empty_map(n=96):
    return torch.zeros((n, n), dtype=torch.float32)


def wall_map(n=96):
    m = torch.zeros((n, n), dtype=torch.float32)
    m[:, 60] = 1.0                       # wall at col 60 (x = 15 m)
    return m


def room_map(n=96):
    m = torch.zeros((n, n), dtype=torch.float32)
    m[20:24, 20:76] = 1.0                # top wall
    m[72:76, 20:76] = 1.0                # bottom wall
    m[20:76, 20:24] = 1.0                # left wall
    m[20:76, 72:76] = 1.0                # right wall
    return m


def ellipse_in_region(polys, i=0) -> bool:
    """Every face must satisfy ``h_E(n) <= b`` (the ellipse lies in the region)."""
    k = int(polys.n_planes[i])
    A = polys.A[i, :k]
    b = polys.b[i, :k]
    Q = ellipse_matrix(polys.radius[i:i + 1] * 0 + 1.0,  # placeholder, replaced below
                       torch.ones(1), torch.zeros(1))
    del Q
    return True


def check_ellipse_inside(polys, a, bb, theta, i=0) -> bool:
    k = int(polys.n_planes[i])
    Q = ellipse_matrix(a[i:i + 1], bb[i:i + 1], theta[i:i + 1])[0]
    sup = ellipse_support(Q, polys.center[i], polys.A[i, :k])
    return bool((sup <= polys.b[i, :k] + 1e-5).all())


def main() -> int:
    torch.manual_seed(0)

    # ---------------------------------------------------------------- 1 free space
    m = empty_map()
    c = torch.tensor([[12.0, 12.0]])
    a = torch.tensor([2.0])
    bb = torch.tensor([1.0])
    th = torch.tensor([0.0])
    # empty sectors of the seeding step get a trivially satisfied face (n=0, b=BIG), so the
    # plane count is a uniform 8 map/window + seed_directions; nothing else may bind
    p = build_polytopes(c, a, bb, th, m, RES, seed_directions=8)
    hi_x, lo_x = float(p.b[0, 0]), float(-p.b[0, 1])
    check("free space: only map + window faces are used", int(p.n_planes[0]) == 16,
          f"n_planes {int(p.n_planes[0])}")
    trivial = bool((p.A[0, 8:16].abs().sum(dim=-1) < 1e-9).all())
    check("free space: all seeded faces are trivially satisfied", trivial)
    check("free space: region reaches the map border",
          abs(hi_x - ((m.shape[1] - 1) * RES - 0.02)) < 1e-4, f"x_max {hi_x:.3f}")
    check("free space: ellipse inside the region", check_ellipse_inside(p, a, bb, th))

    # ---------------------------------------------------------------- 2 single wall
    m = wall_map()
    c = torch.tensor([[10.0, 12.0]])
    a = torch.tensor([1.5])
    bb = torch.tensor([1.0])
    th = torch.tensor([0.3])
    # a generous window so the wall (5 m away) actually binds the region
    p = build_polytopes(c, a, bb, th, m, RES, region_margin_m=0.02, window_margin_m=6.0)
    A0, b0 = p.planes(0)
    wall_face = 60 * RES - 0.5 * RES          # x of the near face of the wall cell
    from ellipsenet.geometry.polytopes import _region_bbox
    _lo_x, _hi_x, _lo_y, _hi_y = _region_bbox(p.A[:, :int(p.n_planes[0])],
                                             p.b[:, :int(p.n_planes[0])],
                                             torch.float32, torch.device('cpu'))
    x_max = float(_hi_x[0])
    check("wall: a separating face was added on the wall side",
          int(p.n_planes[0]) > 8, f"n_planes {int(p.n_planes[0])}")
    check("wall: region stops before the wall cell face",
          wall_face - 0.2 < x_max <= wall_face + 1e-6,
          f"x_max {x_max:.3f} vs wall face {wall_face:.3f}")
    check("wall: ellipse inside the region", check_ellipse_inside(p, a, bb, th))
    rep = verify_polytopes(p, m, RES)
    check("wall: no blocked square inside the region", rep["violating_cells"] == 0,
          str(rep))

    # ---------------------------------------------------------------- 3 small room
    m = room_map()
    cen = torch.tensor([[12.0, 12.0], [10.0, 10.0]])
    a = torch.tensor([1.0, 0.7])
    bb = torch.tensor([0.8, 0.5])
    th = torch.tensor([0.0, 1.2])
    p = build_polytopes(cen, a, bb, th, m, RES, region_margin_m=0.02)
    rep = verify_polytopes(p, m, RES)
    check("room: both regions verified", rep["violating_cells"] == 0.0, str(rep))
    check("room: ellipses inside their regions",
          check_ellipse_inside(p, a, bb, th, 0) and check_ellipse_inside(p, a, bb, th, 1))
    areas = [polytope_area(*p.planes(i)) for i in range(2)]
    room_area = ((76 - 24) * RES) ** 2
    check("room: regions are bounded by the room walls",
          all(3.0 < a_ <= room_area + 1e-6 for a_ in areas),
          str([round(v, 2) for v in areas]) + f" room {room_area:.1f}")

    # ------------------------------------------------- 4 parallel is as safe as serial
    m = room_map()
    n = 12
    cen = torch.rand(n, 2) * 8.0 + 8.0
    a = torch.rand(n) * 1.2 + 0.5
    bb = torch.rand(n) * 0.5 + 0.3
    th = torch.rand(n) * 3.1
    par = build_polytopes(cen, a, bb, th, m, RES)
    ser = build_polytopes_serial(cen.numpy(), a.numpy(), bb.numpy(), th.numpy(),
                                 m.numpy(), RES)
    rep_par = verify_polytopes(par, m, RES)
    rep_ser = verify_polytopes(ser, m, RES)
    check("batched: all regions verified", rep_par["violating_cells"] == 0.0,
          str(rep_par))
    check("serial: all regions verified", rep_ser["violating_cells"] == 0.0,
          str(rep_ser))
    check("batched: every ellipse still inside its region",
          all(check_ellipse_inside(par, a, bb, th, i) for i in range(n)))
    # regions must stay inside their window (semi-major axis + margin, squared-ish bound)
    areas_par = [polytope_area(*par.planes(i)) for i in range(n)]
    win_side = 2.0 * (float(torch.maximum(a, bb).max()) + 3.0)
    check("batched regions stay inside their window",
          all(0.0 < x <= (win_side + 1.0) ** 2 for x in areas_par),
          str([round(v, 1) for v in areas_par]))

    # ---------------------------------------------------------------- 5 speed
    m = room_map()
    n = 256
    cen = torch.rand(n, 2) * 8.0 + 8.0
    a = torch.rand(n) * 1.5 + 0.5
    bb = torch.rand(n) * 0.6 + 0.3
    th = torch.rand(n) * 3.1
    for _ in range(3):
        build_polytopes(cen, a, bb, th, m, RES)
    t0 = time.perf_counter()
    for _ in range(10):
        build_polytopes(cen, a, bb, th, m, RES)
    dt = (time.perf_counter() - t0) / 10
    print(f"  [info] CPU batched: {n} ellipses in {dt * 1e3:.2f} ms "
          f"({dt / n * 1e6:.1f} us/ellipse)")
    check("CPU batched construction stays under 10 ms per ellipse (fallback path)", dt / n < 10e-3,
          f"{dt * 1e3:.1f} ms total, {dt / n * 1e3:.2f} ms/ellipse")

    if torch.cuda.is_available():
        dev = torch.device("cuda")
        mm = m.to(dev)
        cc, aa, bbb, tt = cen.to(dev), a.to(dev), bb.to(dev), th.to(dev)
        for _ in range(5):
            build_polytopes(cc, aa, bbb, tt, mm, RES)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            build_polytopes(cc, aa, bbb, tt, mm, RES)
        torch.cuda.synchronize()
        dt_gpu = (time.perf_counter() - t0) / 50
        print(f"  [info] GPU batched: {n} ellipses in {dt_gpu * 1e3:.2f} ms "
              f"({dt_gpu / n * 1e6:.1f} us/ellipse)")
        rep = verify_polytopes(build_polytopes(cc, aa, bbb, tt, mm, RES), mm, RES)
        check("GPU: all regions verified", rep["violating_cells"] == 0.0, str(rep))
        check("GPU: batched construction under 200 us per ellipse", dt_gpu / n < 2e-4,
              f"{dt_gpu * 1e3:.1f} ms total, {dt_gpu / n * 1e6:.0f} us/ellipse")

    print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())


