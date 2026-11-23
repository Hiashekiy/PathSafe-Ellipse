"""Ellipse -> convex polytope (safe region) inflation, batched and exact.

Method borrowed from Neural-IRIS (``src/neural_iris/safe_region.py``): for one ellipse
only **local** obstacle cells are inspected, they are visited in the ellipse metric
``(x-c)^T Q (x-c)`` ascending, and each visit adds one separating half-space
``n . x <= b``.

Two deliberate upgrades over that reference
-------------------------------------------
* the obstacle is treated as the whole blocked **square** (half-width ``h = r/2``) and
  the offset uses its exact support function ``n . o - h(|n_x| + |n_y|)`` instead of
  ``n . o - margin``.  The face therefore touches the obstacle facet: the region is as
  large as the separating plane allows and can never cut into the square.
* the region is **verified analytically** at the end: a blocked square lies inside
  ``{n.x <= b}`` iff ``n . o + h(|n_x| + |n_y|) <= b``, so "is any blocked cell inside
  my region" is a closed-form test over the cells of the region's bounding box (the box
  comes from the plane pair intersections - no sampling, no full-map scan per ellipse).

Parallelism
-----------
All ellipses of a batch go through the same vectorised kernels; there is no per-ellipse
python loop in :func:`build_polytopes`.  :func:`build_polytopes_serial` is the reference
implementation (one ellipse at a time) used by the tests to prove the two agree.

Half-space convention: ``A`` is ``(N, P, 2)`` outward unit normals, ``b`` is ``(N, P)``,
row ``k`` valid while ``k < n_planes``; padded rows are ``n = 0, b = BIG`` (always true).
The region is ``{x : A_k . x <= b_k}``, an intersection of half-planes that contains the
predicted ellipse and no blocked cell.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

BIG = 1e9


# --------------------------------------------------------------------------- #
# batched region container
# --------------------------------------------------------------------------- #
@dataclass
class Polytopes:
    A: torch.Tensor          # (N, P, 2) outward unit normals (padded)
    b: torch.Tensor          # (N, P)   offsets (padded)
    n_planes: torch.Tensor   # (N,)     valid rows
    converged: torch.Tensor  # (N,) bool: no blocked cell inside any region
    n_rounds: torch.Tensor   # (N,)     inflation planes added from the local window
    center: torch.Tensor     # (N, 2)   ellipse centre [m]
    radius: torch.Tensor     # (N,)     semi-major axis [m] (diagnostics)
    feasible: torch.Tensor   # (N,) bool: no added face cut the ellipse itself

    def planes(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        k = int(self.n_planes[i])
        return (self.A[i, :k].detach().cpu().numpy(),
                self.b[i, :k].detach().cpu().numpy())

    def contains(self, points: torch.Tensor, tol: float = 0.0) -> torch.Tensor:
        """``points (N, M, 2)`` -> ``(N, M)`` inside test."""
        return ((points @ self.A.transpose(1, 2)) <= (self.b[:, None, :] + tol)).all(-1)


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def ellipse_matrix(a: torch.Tensor, b: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """``Q = R diag(1/a^2, 1/b^2) R^T``, i.e. ``{x : (x-c)^T Q (x-c) <= 1}``."""
    c, s = torch.cos(theta), torch.sin(theta)
    R = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
    D = torch.zeros(*a.shape, 2, 2, dtype=a.dtype, device=a.device)
    D[..., 0, 0] = 1.0 / a.clamp_min(1e-6) ** 2
    D[..., 1, 1] = 1.0 / b.clamp_min(1e-6) ** 2
    return R @ D @ R.transpose(-1, -2)


def ellipse_support(Q: torch.Tensor, c: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """Support function of the ellipse ``max_{x in E} n . x`` (``n`` may be unnormalised)."""
    S = torch.linalg.inv(Q)
    quad = torch.einsum("...i,...ij,...j->...", n, S, n).clamp_min(0.0)
    return (n * c).sum(-1) + torch.sqrt(quad)


def cell_support(n: torch.Tensor, centres: torch.Tensor, half: float) -> torch.Tensor:
    """Support function of an axis-aligned square (exact: ``half*(|nx|+|ny|)``)."""
    return (n * centres).sum(-1) + half * n.abs().sum(-1)


def cell_face(n: torch.Tensor, centres: torch.Tensor, half: float) -> torch.Tensor:
    """Offset of the plane that separates the region from a square: its **near** face.

    ``n . o - h(|n_x| + |n_y|)`` is the smallest value of ``n . x`` over the square, so a
    half-space ``n . x <= that`` leaves the whole square on the outside.  Using the
    support (``+h``) instead would cut *through* the square - that is the bug this
    helper exists to avoid.
    """
    return (n * centres).sum(-1) - half * n.abs().sum(-1)


# --------------------------------------------------------------------------- #
# local obstacle gathering (never touches the whole map)
# --------------------------------------------------------------------------- #
def _local_cells(blocked: torch.Tensor, centres_cell: torch.Tensor, window: int,
                 resolution_m: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``window x window`` block of cells around each centre.

    Returns ``(cell_centres (N,K,2) [m], is_blocked (N,K), in_map (N,K))`` where ``K =
    window^2``.  Cells outside the map are reported as blocked (outside == obstacle).
    """
    H, W = blocked.shape
    pad = window // 2
    padded = torch.zeros((H + 2 * pad, W + 2 * pad), dtype=torch.float32,
                         device=blocked.device)
    padded[pad:pad + H, pad:pad + W] = blocked.float()
    dy, dx = torch.meshgrid(
        torch.arange(-pad, pad + 1, device=blocked.device),
        torch.arange(-pad, pad + 1, device=blocked.device), indexing="ij")
    rows = centres_cell[:, 0:1] + pad + dy.reshape(1, -1)
    cols = centres_cell[:, 1:2] + pad + dx.reshape(1, -1)
    inside_map = ((rows - pad) >= 0) & ((rows - pad) < H) & \
                 ((cols - pad) >= 0) & ((cols - pad) < W)
    vals = padded[rows.clamp(0, H + 2 * pad - 1), cols.clamp(0, W + 2 * pad - 1)]
    is_blocked = (vals > 0.5) | (~inside_map)
    cy = (rows - pad).float() * resolution_m
    cx = (cols - pad).float() * resolution_m
    return torch.stack([cx, cy], dim=-1), is_blocked, inside_map


def _map_planes(blocked: torch.Tensor, resolution_m: float, margin_m: float,
                dtype, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Four faces just inside the map border (outside counts as blocked)."""
    H, W = blocked.shape
    A = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
                     dtype=dtype, device=device)
    b = torch.tensor([(W - 1) * resolution_m - margin_m, -margin_m,
                      (H - 1) * resolution_m - margin_m, -margin_m],
                     dtype=dtype, device=device)
    return A, b


# --------------------------------------------------------------------------- #
# closed-form checks
# --------------------------------------------------------------------------- #
def _overlap_mask(A: torch.Tensor, b: torch.Tensor, n_planes: torch.Tensor,
                  cells: torch.Tensor, half: float, eps: float = 1e-9) -> torch.Tensor:
    """``(N, K)`` bool: the square **intersects** the region (exact).

    A square and a polytope overlap iff one of the square's four corners lies inside the
    polytope, or the polytope is entirely inside the square (caught by
    :func:`_vertex_hits`).  Corner membership is the honest criterion: the pure support
    test only sees *full containment* and misses a corner poking in.
    """
    N, P, _two = A.shape
    base = (A.unsqueeze(2) * cells.unsqueeze(1)).sum(-1)                # (N,P,K)
    absA = A.abs()                                                     # (N,P,2)
    valid = (torch.arange(P, device=A.device)[None, :] < n_planes[:, None])
    lim = (b.unsqueeze(2) - eps)
    res = torch.zeros(base.shape, dtype=torch.bool, device=A.device)
    for sx, sy in ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)):
        off = half * (sx * absA[:, :, 0] + sy * absA[:, :, 1])          # (N,P)
        inside = (base + off[:, :, None]) <= lim
        res |= inside
    inside_all = torch.where(valid.unsqueeze(2), res, torch.ones_like(res))
    return inside_all.all(dim=1)


def _inside_mask(A: torch.Tensor, b: torch.Tensor, n_planes: torch.Tensor,
                 cells: torch.Tensor, half: float) -> torch.Tensor:
    """``(N, K)`` bool: the whole square lies inside the region (support test)."""
    sup = cell_support(A.unsqueeze(2), cells.unsqueeze(1), half)       # (N,P,K)
    valid = (torch.arange(A.shape[1], device=A.device)[None, :] < n_planes[:, None])
    inside = sup <= (b[:, :, None] + 1e-9)
    return (inside | ~valid[:, :, None]).all(dim=1)


def _vertex_hits(A: torch.Tensor, b: torch.Tensor, blocked: torch.Tensor,
                 resolution_m: float) -> torch.Tensor:
    """``(N,)`` bool: is a region vertex inside a blocked cell?

    Complements :func:`_overlap_mask` for the (rare) case where the region is so small
    that it sits entirely inside one cell: then no cell corner is inside the region.
    """
    device = A.device
    P = A.shape[1]
    i, j = torch.triu_indices(P, P, offset=1, device=device)
    A1, A2 = A[:, i], A[:, j]
    b1, b2 = b[:, i], b[:, j]
    det = A1[..., 0] * A2[..., 1] - A1[..., 1] * A2[..., 0]
    ok = det.abs() > 1e-9
    det_safe = torch.where(ok, det, torch.ones_like(det))
    x = (b1 * A2[..., 1] - b2 * A1[..., 1]) / det_safe
    y = (A1[..., 0] * b2 - A2[..., 0] * b1) / det_safe
    pts = torch.stack([x, y], dim=-1)
    feasible = ((pts @ A.transpose(1, 2)) <= (b[:, None, :] + 1e-6)).all(-1) & ok
    H, W = blocked.shape
    res = float(resolution_m)
    cols = torch.round(pts[..., 0] / res).long().clamp(0, W - 1)
    rows = torch.round(pts[..., 1] / res).long().clamp(0, H - 1)
    hit = (blocked[rows, cols] > 0.5) & feasible
    return hit.any(dim=1)


def _region_bbox(A: torch.Tensor, b: torch.Tensor,
                 dtype, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact bounding box of the region, from all feasible plane-pair intersections."""
    P = A.shape[1]
    i, j = torch.triu_indices(P, P, offset=1, device=device)
    A1, A2 = A[:, i], A[:, j]                     # (N,M,2)
    b1, b2 = b[:, i], b[:, j]                     # (N,M)
    det = A1[..., 0] * A2[..., 1] - A1[..., 1] * A2[..., 0]
    ok = det.abs() > 1e-9
    det_safe = torch.where(ok, det, torch.ones_like(det))
    x = (b1 * A2[..., 1] - b2 * A1[..., 1]) / det_safe
    y = (A1[..., 0] * b2 - A2[..., 0] * b1) / det_safe
    pts = torch.stack([x, y], dim=-1)             # (N,M,2)
    feasible = ((pts @ A.transpose(1, 2)) <= (b[:, None, :] + 1e-6)).all(-1) & ok
    mask = feasible.unsqueeze(-1)
    big = torch.where(mask, pts, torch.full_like(pts, float("inf")))
    lo = big.amin(dim=1)                          # (N,2)
    big_hi = torch.where(mask, pts, torch.full_like(pts, -float("inf")))
    hi = big_hi.amax(dim=1)
    lo = torch.where(torch.isfinite(lo), lo, torch.zeros_like(lo))
    hi = torch.where(torch.isfinite(hi), hi, torch.zeros_like(hi))
    return lo[:, 0], hi[:, 0], lo[:, 1], hi[:, 1]


def _first_inside_cell(A: torch.Tensor, b: torch.Tensor, n_planes: torch.Tensor,
                       blocked: torch.Tensor, resolution_m: float, half: float
                       ) -> torch.Tensor:
    """Flat index of the first blocked cell inside each region (-1 = verified).

    Scans only the region bounding box (derived from the planes), not the map.
    """
    N = A.shape[0]
    H, W = blocked.shape
    res = float(resolution_m)
    lo_x, hi_x, lo_y, hi_y = _region_bbox(A, b, A.dtype, A.device)
    c0 = torch.clamp(torch.floor(lo_x / res), 0, W - 1).long()
    c1 = torch.clamp(torch.ceil(hi_x / res), 0, W - 1).long()
    r0 = torch.clamp(torch.floor(lo_y / res), 0, H - 1).long()
    r1 = torch.clamp(torch.ceil(hi_y / res), 0, H - 1).long()
    span_c = (c1 - c0 + 1).clamp_min(1)
    span_r = (r1 - r0 + 1).clamp_min(1)
    K = int((span_c * span_r).max().item())
    K = max(min(K, H * W), 1)
    k = torch.arange(K, device=A.device)[None, :]
    cols = c0[:, None] + (k % span_c[:, None])
    rows = r0[:, None] + (k // span_c[:, None])
    valid = (cols <= c1[:, None]) & (rows <= r1[:, None])
    cols_c = cols.clamp(0, W - 1)
    rows_c = rows.clamp(0, H - 1)
    cand = valid & (blocked[rows_c, cols_c] > 0.5)
    cells = torch.stack([cols_c.float() * res, rows_c.float() * res], dim=-1)
    bad = cand & _inside_mask(A, b, n_planes, cells, half)
    has = bad.any(dim=1)
    first = torch.where(has, bad.float().argmax(dim=1), torch.zeros((N,), dtype=torch.long,
                                                                   device=A.device))
    rr = rows_c.gather(1, first[:, None]).squeeze(1)
    cc = cols_c.gather(1, first[:, None]).squeeze(1)
    return torch.where(has, rr * W + cc, torch.full((N,), -1, dtype=torch.long,
                                                    device=A.device))


# --------------------------------------------------------------------------- #
# batched inflation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def build_polytopes(centres: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                    theta: torch.Tensor, blocked: torch.Tensor, resolution_m: float,
                    region_margin_m: float = 0.02, max_planes: int = 16,
                    window_margin_m: float = 3.0,
                    planes_per_iter: int = 1,
                    seed_directions: int = 8,
                    refine: bool = True) -> Polytopes:
    """Inflate one convex region per predicted ellipse, all in parallel.

    ``centres (N,2)`` metres, ``a/b/theta (N,)``, ``blocked (H,W)`` 0/1 grid,
    ``resolution_m`` metres per cell, ``region_margin_m`` keeps every face slightly off
    the obstacle.

    The region is, **by construction**, contained in the ellipse's own window
    (semi-major axis + ``window_margin_m``).  The four window faces are added as hard
    planes, which is what makes the whole procedure local: the search is a fixed-size
    neighbourhood around each ellipse, the greedy provably terminates, and the exact
    verification only has to look at the same window cells.  Pass a larger
    ``window_margin_m`` if a bigger region is preferred at the cost of speed.
    """
    device = centres.device
    dtype = torch.float32
    N = centres.shape[0]
    res = float(resolution_m)
    half = 0.5 * res
    centres = centres.to(dtype=dtype)
    a = a.to(dtype=dtype)
    b = b.to(dtype=dtype)
    theta = theta.to(dtype=dtype)
    blocked = blocked.to(device)
    Q = ellipse_matrix(a, b, theta)
    H, W = blocked.shape

    # ---- local window = ellipse bbox + margin (never the whole map)
    radius = float(torch.maximum(a, b).max()) + float(window_margin_m)
    half_cells = max(int(np.ceil(radius / res)), 4)                   # region cap half-size
    win_cells = 2 * half_cells + 5                                    # gathered cell patch
    # (row, col) = (y/res, x/res): the grid is indexed [row=y, col=x], so the centre
    # must be reordered here - swapping it silently transposes every local patch.
    centres_rc = torch.round(centres / res).long()[:, [1, 0]]
    cells, is_blocked, _in_map = _local_cells(blocked, centres_rc, win_cells, res)
    diff = cells - centres[:, None, :]
    metric = torch.einsum("nki,nij,nkj->nk", diff, Q, diff)
    metric = metric.masked_fill(~is_blocked, float("inf"))

    # ---- compact the window to its *blocked* cells, already sorted by the ellipse
    # metric.  On a real scene only ~4% of the window is blocked (measured 214 of 5929),
    # so this cuts every later pass by >4x while staying exact: no blocked cell of the
    # window is dropped, they are only re-ordered and padded.
    m_cells = max(int(is_blocked.sum(dim=1).max().item()), 1)
    cand_metric, idx = torch.topk(metric, m_cells, dim=1, largest=False)
    cand = torch.gather(cells, 1, idx[..., None].expand(-1, -1, 2))    # (N,M,2)
    cand_ok = torch.isfinite(cand_metric)                              # padded tail = False

    # ---- planes: map border + window border (both keep the region provably local).
    # The cap is centred on the ellipse's cell and is >= its semi-major axis, so it can
    # never cut the ellipse itself.
    A0, b0 = _map_planes(blocked, res, region_margin_m, dtype, device)
    cx = centres_rc[:, 1].float() * res                              # col -> x
    cy = centres_rc[:, 0].float() * res                              # row -> y
    cap = half_cells * res - region_margin_m
    wx_lo = (cx - cap).clamp_min(region_margin_m)
    wx_hi = (cx + cap).clamp_max((W - 1) * res - region_margin_m)
    wy_lo = (cy - cap).clamp_min(region_margin_m)
    wy_hi = (cy + cap).clamp_max((H - 1) * res - region_margin_m)
    P = max_planes + 8
    A = torch.zeros((N, P, 2), dtype=dtype, device=device)
    off = torch.full((N, P), BIG, dtype=dtype, device=device)
    A[:, :4] = A0
    off[:, :4] = b0
    winA = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
                        dtype=dtype, device=device)
    A[:, 4:8] = winA
    off[:, 4] = wx_hi
    off[:, 5] = -wx_lo
    off[:, 6] = wy_hi
    off[:, 7] = -wy_lo
    n_planes = torch.full((N,), 8, dtype=torch.long, device=device)
    ar = torch.arange(N, device=device)
    feasible = torch.ones((N,), dtype=torch.bool, device=device)

    # ---- seed: one face per angular sector, all sectors in a single step.
    # This is the batched version of the classic "N sectors -> N constraints" start-up: the
    # exact loop below then only has to close the corners between sectors (1-2 rounds
    # instead of one round per obstacle).  Empty sectors get a trivially satisfied face.
    if seed_directions and seed_directions > 1:
        D = int(seed_directions)
        rel = cand - centres[:, None, :]                              # (N,M,2)
        ang = torch.atan2(rel[..., 1], rel[..., 0])                   # (N,M) in (-pi, pi]
        sector = torch.floor((ang + np.pi) / (2 * np.pi) * D).long().clamp_(0, D - 1)
        metric = torch.where(cand_ok, cand_metric, torch.full_like(cand_metric, float("inf")))
        # per (ellipse, sector) minimum of the ellipse metric
        best = torch.full((N, D), float("inf"), dtype=dtype, device=device)
        best.scatter_reduce_(1, sector, metric, reduce="amin", include_self=True)
        # the cell that realises that minimum, per sector
        same = sector.unsqueeze(-1) == torch.arange(D, device=device)[None, None, :]
        is_min = metric.unsqueeze(-1) == best.unsqueeze(1)             # (N,M,D)
        sel = same & is_min & cand_ok.unsqueeze(-1)
        chosen = sel.float().argmax(dim=1)                            # (N,D)
        has = sel.any(dim=1)                                          # (N,D)
        obs = cand[ar[:, None], chosen]                               # (N,D,2)
        normal = torch.einsum("nij,ndj->ndi", Q, obs - centres[:, None, :])
        normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        offset = cell_face(normal, obs, half) - region_margin_m       # (N,D)
        cuts = ellipse_support(Q.unsqueeze(1), centres.unsqueeze(1), normal) > (offset + 1e-6)
        feasible = feasible & ~(has & cuts).any(dim=1)
        A[:, 8:8 + D] = torch.where(has[..., None], normal, torch.zeros_like(normal))
        off[:, 8:8 + D] = torch.where(has, offset, torch.full_like(offset, BIG))
        n_planes = torch.full((N,), 8 + D, dtype=torch.long, device=device)

    # ---- single exact inflation loop.
    # The overlap test below is the honest one (a square poking one corner into the region
    # counts), so the loop *terminates only when the region is provably obstacle free* -
    # that termination condition IS the verification; there is no separate check pass that
    # could disagree with the construction.
    # The whole call is latency bound rather than compute bound (a 30-ellipse call and a
    # 256-ellipse call take the same wall time), so several faces are added per iteration:
    # fewer iterations is the only thing that actually matters here.
    for _ in range(P if refine else 0):
        bad = _overlap_mask(A, off, n_planes, cand, half) & cand_ok
        any_bad = bad.any(dim=1) & feasible & (n_planes < P)
        if not bool(any_bad.any()):
            break
        n_add = int(min(planes_per_iter, P))
        vals, idxs = torch.topk(bad.float(), k=n_add, dim=1)          # metric-ordered
        for j in range(n_add):
            add = any_bad & ((n_planes + j) < P) & (vals[:, j] > 0.5)
            if not bool(add.any()):
                continue
            obs = cand[ar, idxs[:, j]]                                # (N,2)
            normal = torch.einsum("nij,nj->ni", Q, obs - centres)
            normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            offset = cell_face(normal, obs, half) - region_margin_m
            # a face that cuts the ellipse means this ellipse (or this margin) is
            # infeasible: no obstacle-free region can contain it
            cuts = ellipse_support(Q, centres, normal) > (offset + 1e-6)
            feasible = feasible & ~(add & cuts)
            slot = (n_planes + j).clamp_max(P - 1)
            A[ar[add], slot[add]] = normal[add]
            off[ar[add], slot[add]] = offset[add]
            n_planes = n_planes + add.long()

    still = (_overlap_mask(A, off, n_planes, cand, half) & cand_ok).any(dim=1)
    converged = ~(still | _vertex_hits(A[:, :P], off, blocked, res))

    return Polytopes(A=A, b=off, n_planes=n_planes, converged=converged,
                     n_rounds=(n_planes - 8), center=centres,
                     radius=torch.maximum(a, b), feasible=feasible)


# --------------------------------------------------------------------------- #
# diagnostics (python loops are acceptable here - tests, plots, acceptance)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def verify_polytopes(polys: Polytopes, blocked: torch.Tensor, resolution_m: float,
                     margin_m: float = 0.0) -> Dict[str, float]:
    """Exact whole-map check: how many blocked squares enter a region."""
    H, W = blocked.shape
    res = float(resolution_m)
    half = 0.5 * res
    rr, cc = torch.nonzero(blocked > 0.5, as_tuple=True)
    if rr.numel() == 0:
        return {"n_cells": 0, "violating_regions": 0.0, "violating_cells": 0.0}
    cells = torch.stack([cc.float() * res, rr.float() * res], dim=-1).to(blocked.device)
    bad_regions = 0
    bad_cells = 0
    for i in range(polys.A.shape[0]):
        k = int(polys.n_planes[i])
        overlap = _overlap_mask(polys.A[i:i + 1, :k], polys.b[i:i + 1, :k] - margin_m,
                                torch.tensor([k], device=blocked.device),
                                cells.unsqueeze(0), half)
        vertex_bad = _vertex_hits(polys.A[i:i + 1, :k], polys.b[i:i + 1, :k] - margin_m,
                                  blocked, res)
        n_bad = int(overlap[0].sum()) + int(bool(vertex_bad[0]))
        bad_cells += n_bad
        bad_regions += int(n_bad > 0)
    return {"n_cells": int(rr.numel()), "violating_regions": float(bad_regions),
            "violating_cells": float(bad_cells)}


def polytope_area(A: np.ndarray, b: np.ndarray) -> float:
    """Area of ``{A x <= b}`` via the convex hull of feasible plane intersections."""
    from scipy.spatial import ConvexHull

    n = A.shape[0]
    pts = []
    for i in range(n):
        for j in range(i + 1, n):
            M = np.stack([A[i], A[j]])
            if abs(np.linalg.det(M)) < 1e-9:
                continue
            p = np.linalg.solve(M, np.array([b[i], b[j]]))
            if np.all(A @ p <= b + 1e-6):
                pts.append(p)
    if len(pts) < 3:
        return 0.0
    try:
        return float(ConvexHull(np.asarray(pts)).volume)
    except Exception:
        return 0.0


def build_polytopes_serial(centres: np.ndarray, a: np.ndarray, b: np.ndarray,
                           theta: np.ndarray, blocked: np.ndarray, resolution_m: float,
                           device: Optional[torch.device] = None, **kwargs) -> Polytopes:
    """Reference implementation, one ellipse at a time (tests only)."""
    dev = device or torch.device("cpu")
    blk = torch.as_tensor(blocked, dtype=torch.float32, device=dev)
    parts = [build_polytopes(torch.tensor(centres[i:i + 1], dtype=torch.float32, device=dev),
                             torch.tensor(a[i:i + 1], dtype=torch.float32, device=dev),
                             torch.tensor(b[i:i + 1], dtype=torch.float32, device=dev),
                             torch.tensor(theta[i:i + 1], dtype=torch.float32, device=dev),
                             blk, resolution_m, **kwargs)
             for i in range(len(centres))]
    P = max(int(p.n_planes.max()) for p in parts)
    return Polytopes(A=torch.cat([p.A[:, :P] for p in parts]),
                     b=torch.cat([p.b[:, :P] for p in parts]),
                     n_planes=torch.cat([p.n_planes for p in parts]),
                     converged=torch.cat([p.converged for p in parts]),
                     n_rounds=torch.cat([p.n_rounds for p in parts]),
                     center=torch.cat([p.center for p in parts]),
                     radius=torch.cat([p.radius for p in parts]),
                     feasible=torch.cat([p.feasible for p in parts]))






