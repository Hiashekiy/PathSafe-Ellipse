"""Evaluation metrics for the argmax-selected ellipses (plan section 14).

Two levels:

* **tensor level** (fast, no NumPy loop): SDF clearance of the selected centre,
  ellipse area, SDF-sampled safety rate, direction entropy / max probability,
  agreement between the argmax direction and the minimum-loss candidate, and the
  comparison with the stored GT labels (diagnostic only - never a training label);
* **exact level** (NumPy): the analytic ellipse-vs-blocked-square test of
  ``pathsafe_ellipse.v2.ellipse.exact_clearance`` plus boundary coverage between
  adjacent ellipses.  The SDF sampling above is a differentiable *proxy*, so the
  collision rate that is reported as the acceptance number is this exact one.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from psen_v3.geometry.sampling import (ellipse_points, ellipse_rotation,  # noqa: E402
                                       quadratic_form, sample_sdf, unit_ellipse_template)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def quantiles(values, prefix: str = "") -> Dict[str, Optional[float]]:
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {f"{prefix}mean": None, f"{prefix}p10": None, f"{prefix}p50": None,
                f"{prefix}p90": None, f"{prefix}p95": None, f"{prefix}n": 0}
    return {f"{prefix}mean": float(v.mean()),
            f"{prefix}p10": float(np.percentile(v, 10)),
            f"{prefix}p50": float(np.percentile(v, 50)),
            f"{prefix}p90": float(np.percentile(v, 90)),
            f"{prefix}p95": float(np.percentile(v, 95)),
            f"{prefix}n": int(v.size)}


def ellipse_S(a: torch.Tensor, b: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """``R diag(a^2, b^2) R^T`` for ``(B,N)`` inputs -> ``(B,N,2,2)``."""
    c, s = torch.cos(theta), torch.sin(theta)
    R = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
    d = torch.stack([a * a, b * b], -1)
    D = torch.zeros(*d.shape[:-1], 2, 2, dtype=d.dtype, device=d.device)
    D[..., 0, 0] = d[..., 0]
    D[..., 1, 1] = d[..., 1]
    return R @ D @ R.transpose(-1, -2)


def ellipse_iou_np(c1, a1, b1, t1, c2, a2, b2, t2, n: int = 48) -> float:
    """Rasterised IoU of two ellipses (union bounding box, ``n x n`` grid)."""
    pad = 0.05 * max(a1, b1, a2, b2, 1e-3)
    lo = np.minimum(np.asarray(c1) - max(a1, b1), np.asarray(c2) - max(a2, b2)) - pad
    hi = np.maximum(np.asarray(c1) + max(a1, b1), np.asarray(c2) + max(a2, b2)) + pad
    xs = np.linspace(lo[0], hi[0], n)
    ys = np.linspace(lo[1], hi[1], n)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel()], axis=-1)

    def inside(c, a, b, t):
        d = pts - np.asarray(c)[None, :]
        dx = np.cos(t) * d[:, 0] + np.sin(t) * d[:, 1]
        dy = -np.sin(t) * d[:, 0] + np.cos(t) * d[:, 1]
        return (dx / a) ** 2 + (dy / b) ** 2 <= 1.0

    m1, m2 = inside(c1, a1, b1, t1), inside(c2, a2, b2, t2)
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(inter / max(union, 1))


# --------------------------------------------------------------------------- #
# exact (NumPy) checks
# --------------------------------------------------------------------------- #
def exact_trajectory(center: np.ndarray, a: np.ndarray, b: np.ndarray, theta: np.ndarray,
                     occupancy: np.ndarray, r_scene: float, rho_max: float = 0.85
                     ) -> Dict[str, object]:
    """Exact safety and continuity of one predicted trajectory (NumPy)."""
    from pathsafe_ellipse.v2.ellipse import exact_clearance

    n = len(center)
    S = ellipse_S(torch.from_numpy(a.astype(np.float64)),
                  torch.from_numpy(b.astype(np.float64)),
                  torch.from_numpy(theta.astype(np.float64))).numpy()
    H, W = occupancy.shape
    safe = np.zeros(n, dtype=bool)
    min_q = np.full(n, np.nan)
    center_blocked = np.zeros(n, dtype=bool)
    for i in range(n):
        ok, mq, _bad, _idx = exact_clearance(S[i], center[i].astype(np.float64),
                                             occupancy, r_scene)
        safe[i] = bool(ok)
        min_q[i] = float(mq) if np.isfinite(mq) else 1e9
        cc = int(round(center[i, 0] / r_scene))
        rr = int(round(center[i, 1] / r_scene))
        center_blocked[i] = (0 <= rr < H and 0 <= cc < W and occupancy[rr, cc] > 0)

    # adjacent-pair continuity: boundary of each ellipse inside the other one
    t = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    unit = np.stack([np.cos(t), np.sin(t)], axis=-1)                  # (32,2)
    pair_ok, pair_cover = [], []
    for i in range(n - 1):
        ci, cj = center[i], center[i + 1]
        bi = _boundary(ci, a[i], b[i], theta[i], unit)
        bj = _boundary(cj, a[i + 1], b[i + 1], theta[i + 1], unit)
        rho_i_in_j = _rho(bi, cj, a[i + 1], b[i + 1], theta[i + 1])
        rho_j_in_i = _rho(bj, ci, a[i], b[i], theta[i])
        cover = 0.5 * (np.mean(rho_i_in_j <= 1.0) + np.mean(rho_j_in_i <= 1.0))
        pair_cover.append(float(cover))
        m = 0.5 * (ci + cj)
        r0 = float(np.sqrt(max(_rho(m[None], ci, a[i], b[i], theta[i])[0], 0.0)))
        r1 = float(np.sqrt(max(_rho(m[None], cj, a[i + 1], b[i + 1], theta[i + 1])[0], 0.0)))
        pair_ok.append(bool(r0 <= rho_max and r1 <= rho_max))
    return {
        "n": n,
        "center": center, "a": a, "b": b, "theta": theta,
        "safe": safe,
        "min_quadratic": min_q,
        "center_blocked": center_blocked,
        "pair_ok": np.asarray(pair_ok, dtype=bool),
        "pair_cover": np.asarray(pair_cover, dtype=np.float64),
        "border": _border(center, a, b, theta, H, W, r_scene),
    }


def _boundary(c, a, b, t, unit):
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    local = unit * np.array([[a, b]])
    return c[None, :] + local @ R.T


def _rho(x, c, a, b, t):
    d = np.asarray(x) - np.asarray(c)[None, :]
    dx = np.cos(t) * d[:, 0] + np.sin(t) * d[:, 1]
    dy = -np.sin(t) * d[:, 0] + np.cos(t) * d[:, 1]
    return (dx / a) ** 2 + (dy / b) ** 2


def _border(center, a, b, theta, H, W, res, margin: float = 0.0) -> np.ndarray:
    """Per-point flag: does the ellipse leave the map rectangle?"""
    c, s = np.cos(theta), np.sin(theta)
    rx = np.sqrt((a * c) ** 2 + (b * s) ** 2) + margin
    ry = np.sqrt((a * s) ** 2 + (b * c) ** 2) + margin
    return ((center[:, 0] - rx < 0) | (center[:, 0] + rx > (W - 1) * res)
            | (center[:, 1] - ry < 0) | (center[:, 1] + ry > (H - 1) * res))


# --------------------------------------------------------------------------- #
# tensor-level diagnostics for one batch
# --------------------------------------------------------------------------- #
@torch.no_grad()
def batch_diagnostics(out: Dict, batch: Dict, cfg: Dict, loss_fn,
                      select: str = "argmax") -> Dict[str, np.ndarray]:
    """Per-point diagnostics of the selected candidate for one batch.

    ``select='argmax'`` is the inference rule of the plan (highest direction
    probability); ``select='minloss'`` is the offline oracle used only as an upper
    bound diagnostic (plan section 14, ablation 8).
    """
    mask = batch["mask"]
    pi = out["pi"]
    terms = loss_fn.local_terms(out, batch)
    l_local = (loss_fn.w_dist * terms["dist"] / loss_fn.s_dist
               + loss_fn.w_safe * terms["safe"] / loss_fn.s_safe
               + loss_fn.w_area * terms["area"][:, :, None] / loss_fn.s_area)
    oracle = l_local.argmin(-1)
    j = pi.argmax(-1) if select == "argmax" else oracle
    q = torch.gather(out["q"], 2, j[..., None, None].expand(-1, -1, 1, 2)).squeeze(2)
    a, b, theta = out["a"], out["b"], out["theta"]
    r_scene = batch["r_scene"]

    s_center = sample_sdf(batch["sdf"], q, r_scene)                  # (B,N)

    template = loss_fn.safe_template
    pts = ellipse_points(q[:, :, None], a, b, theta, template)[:, :, 0]   # (B,N,K,2)
    s_ell = sample_sdf(batch["sdf"], pts, r_scene)                    # (B,N,K)
    margin = loss_fn.d_margin
    safe_proxy = (s_ell >= margin).float().mean(-1)
    outside_frac = (s_ell <= -1.0 + 1e-6).float().mean(-1)

    ent = -(pi.clamp_min(1e-9).log() * pi).sum(-1)
    pmax = pi.max(-1).values
    agree = (oracle == pi.argmax(-1)).float()

    area = torch.pi * a * b
    gt_center = batch["diag_center"]
    gt_valid = batch["diag_valid"] > 0.5
    c_err = torch.linalg.norm(q - gt_center, dim=-1)
    gt_ab = batch["diag_ab"]
    gt_angle = batch["diag_angle"]
    gt_area = batch["diag_area"]

    def flat(x):
        return x[mask].detach().cpu().numpy()

    diag = {
        "n_points": flat(torch.ones_like(c_err)),
        "s_center": flat(s_center),
        "a": flat(a), "b": flat(b), "theta": flat(theta),
        "area": flat(area),
        "safe_proxy": flat(safe_proxy),
        "outside_frac": flat(outside_frac),
        "entropy": flat(ent),
        "p_max": flat(pmax),
        "agree_oracle": flat(agree),
        "cand_loss": (l_local.detach().cpu().numpy()[mask.detach().cpu().numpy()]),
        "center": q.detach().cpu().numpy()[mask.detach().cpu().numpy()],
        "a_raw": a.detach().cpu().numpy()[mask.detach().cpu().numpy()],
        "b_raw": b.detach().cpu().numpy()[mask.detach().cpu().numpy()],
        "theta_raw": theta.detach().cpu().numpy()[mask.detach().cpu().numpy()],
        "gt_valid": flat(gt_valid.float()),
        "center_err": flat(c_err),
        "gt_center": gt_center.detach().cpu().numpy()[mask.detach().cpu().numpy()],
        "gt_ab": gt_ab.detach().cpu().numpy()[mask.detach().cpu().numpy()],
        "gt_angle": gt_angle.detach().cpu().numpy()[mask.detach().cpu().numpy()],
        "gt_area": gt_area.detach().cpu().numpy()[mask.detach().cpu().numpy()],
        "r_scene": batch["r_scene"].detach().cpu().numpy(),
        "occupancy": batch["occupancy"].detach().cpu().numpy(),
        "mask": mask.detach().cpu().numpy(),
    }
    return diag


def summarize(diag: Dict[str, np.ndarray], exact: Optional[Dict[str, object]] = None,
              cfg: Optional[Dict] = None) -> Dict[str, object]:
    """Turn the accumulated diagnostics into the acceptance numbers of plan 14."""
    v = diag["gt_valid"] > 0.5
    out: Dict[str, object] = {
        "n_points": int(diag["n_points"].size),
        "n_trajectories": int(diag["mask"].shape[0]),
        "selected_center_clearance": quantiles(diag["s_center"], "sdf_"),
        "ellipse_area_m2": quantiles(diag["area"]),
        "semi_major_a_m": quantiles(diag["a"]),
        "semi_minor_b_m": quantiles(diag["b"]),
        "sdf_safety_rate": float(np.mean(diag["safe_proxy"])),
        "sdf_outside_sample_frac": float(np.mean(diag["outside_frac"])),
        "direction_entropy": quantiles(diag["entropy"]),
        "direction_p_max": quantiles(diag["p_max"]),
        "argmax_vs_min_loss_agreement": float(np.mean(diag["agree_oracle"])),
        "candidate_loss_distribution": quantiles(diag["cand_loss"].mean(axis=0)),
        "gt_center_error_m": quantiles(diag["center_err"][v]),
        "gt_area_ratio": quantiles((diag["area"][v] /
                                    np.maximum(diag["gt_area"][v], 1e-6))),
        "gt_angle_error_rad": quantiles(_angle_err(diag["theta_raw"][v],
                                                   diag["gt_angle"][v])),
        "gt_n_points": int(v.sum()),
    }
    if exact is not None:
        out.update({
            "exact_collision_rate": float(1.0 - np.mean(exact["safe"])),
            "exact_safe_rate": float(np.mean(exact["safe"])),
            "exact_center_collision_rate": float(np.mean(exact["center_blocked"])),
            "exact_min_quadratic": quantiles(exact["min_quadratic"], "minq_"),
            "continuity_pair_rate": float(np.mean(exact["pair_ok"])) if len(exact["pair_ok"]) else None,
            "boundary_cover_rate": float(np.mean(exact["pair_cover"])) if len(exact["pair_cover"]) else None,
            "border_violation_rate": float(np.mean(exact["border"])),
            "corridor_success_rate": float(np.mean(exact["corridor_ok"])) if "corridor_ok" in exact else None,
        })
    lam = float(cfg["loss"]["w_conn"]) if cfg else 1.0
    out["loss_weights"] = {"w_conn": lam}
    return out


def _angle_err(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    d = np.abs(pred - gt) % np.pi
    return np.minimum(d, np.pi - d)
