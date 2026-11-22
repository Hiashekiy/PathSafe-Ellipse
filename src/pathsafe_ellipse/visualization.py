"""Visualisation helpers (map / sample / path level) for the dataset."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse, Rectangle  # noqa: E402

from .local_features import PatchFeatures, patch_clearance  # noqa: E402
from .path_search import clearance_class  # noqa: E402

PATCH_CENTER = 64.0


def _ellipse_patch(center_xy, semi_a_m, semi_b_m, angle_rad, resolution, **kwargs):
    return Ellipse(
        (center_xy[0], center_xy[1]),
        width=2.0 * semi_a_m / resolution,
        height=2.0 * semi_b_m / resolution,
        angle=np.degrees(angle_rad),
        **kwargs,
    )


def plot_map_overview(
    raw: np.ndarray,
    inflated: np.ndarray,
    paths: Sequence[np.ndarray],
    out_path: str,
    title: str = "",
    clearance: Optional[np.ndarray] = None,
) -> None:
    """Raw vs inflated occupancy, added obstacle cells and the A* paths."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2))
    axes[0].imshow(raw, cmap="gray_r", interpolation="nearest")
    axes[0].set_title("raw occupancy")
    axes[1].imshow(inflated, cmap="gray_r", interpolation="nearest")
    axes[1].set_title("inflated (3x3, 1 iteration)")
    added = (inflated.astype(bool) & ~raw.astype(bool)).astype(np.uint8)
    axes[2].imshow(raw, cmap="gray", interpolation="nearest")
    axes[2].imshow(added, cmap="autumn", alpha=0.6, interpolation="nearest")
    axes[2].set_title("added obstacle cells (red)")
    if clearance is not None:
        im = axes[3].imshow(clearance, cmap="viridis", interpolation="nearest")
        fig.colorbar(im, ax=axes[3], fraction=0.046, label="EDT [cells]")
        axes[3].set_title("distance transform")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    if paths:
        for path in paths:
            arr = np.asarray(path)
            axes[0].plot(arr[:, 0], arr[:, 1], lw=0.8, color="tab:blue")
            axes[1].plot(arr[:, 0], arr[:, 1], lw=0.8, color="tab:blue")
            axes[1].plot(arr[0, 0], arr[0, 1], "o", ms=2.5, color="lime")
            axes[1].plot(arr[-1, 0], arr[-1, 1], "x", ms=4, color="red")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_patch_with_rays(
    patch: np.ndarray,
    ray_dirs: np.ndarray,
    ray_raw: np.ndarray,
    ray_hits: np.ndarray,
    point_xy: Tuple[float, float],
    out_path: str,
    title: str = "",
    raw_patch: Optional[np.ndarray] = None,
) -> None:
    """Local patch, 16 sparse range rays and their endpoints."""
    n = 1 if raw_patch is None else 2
    fig, axes = plt.subplots(1, n, figsize=(6.4 * n, 6.0), squeeze=False)
    panels = [(patch, "inflated patch (network input)")]
    if raw_patch is not None:
        panels.append((raw_patch, "raw patch (for reference only)"))
    for ax, (img, label) in zip(axes[0], panels):
        ax.imshow(img, cmap="gray_r", interpolation="nearest")
        ax.plot(point_xy[0], point_xy[1], "+", ms=10, color="red")
        ax.set_title(label)
        ax.set_xticks([]); ax.set_yticks([])
    ax = axes[0][0]
    for k in range(len(ray_dirs)):
        d = ray_raw[k]
        end = (point_xy[0] + d * ray_dirs[k][0], point_xy[1] + d * ray_dirs[k][1])
        color = "tab:orange" if ray_hits[k] else "tab:cyan"
        ax.plot([point_xy[0], end[0]], [point_xy[1], end[1]], "-", lw=0.7, color=color, alpha=0.9)
        ax.plot(end[0], end[1], ".", ms=3, color=color)
    ax.set_title(f"{panels[0][1]}\n16 rays (orange = hit, cyan = max range)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_sample_ellipse(
    patch: np.ndarray,
    ray_dirs: np.ndarray,
    ray_raw: np.ndarray,
    ray_hits: np.ndarray,
    candidates: Sequence[Tuple[Tuple[float, float], int, float]],
    chosen_center: Tuple[float, float],
    ellipse: Tuple[float, float, float],
    label: Dict[str, float],
    resolution: float,
    out_path: str,
    title: str = "",
    extra_ellipses: Optional[List[Tuple[Tuple[float, float], float, float, float, str]]] = None,
) -> None:
    """One sample: patch, rays, all candidate centres, chosen ellipse."""
    fig, ax = plt.subplots(figsize=(7.4, 7.0))
    ax.imshow(patch, cmap="gray_r", interpolation="nearest")
    point_xy = (PATCH_CENTER, PATCH_CENTER)
    for k in range(len(ray_dirs)):
        d = ray_raw[k]
        ax.plot([point_xy[0], point_xy[0] + d * ray_dirs[k][0]],
                [point_xy[1], point_xy[1] + d * ray_dirs[k][1]],
                lw=0.5, color="tab:orange" if ray_hits[k] else "tab:cyan", alpha=0.7)
    for (cx, cy), direction, ratio in candidates:
        ax.plot(cx, cy, "o", ms=2.5, mfc="none", mec="tab:green", mew=0.6)
    for (c, a_m, b_m, ang, color) in (extra_ellipses or []):
        ax.add_patch(_ellipse_patch(c, a_m, b_m, ang, resolution,
                                    fill=False, ec=color, lw=1.0, ls="--"))
    a_m, b_m, ang = ellipse
    ax.add_patch(_ellipse_patch(chosen_center, a_m, b_m, ang, resolution,
                                fill=False, ec="tab:blue", lw=1.8))
    ax.plot(point_xy[0], point_xy[1], "+", ms=11, color="red", mew=2)
    ax.plot(chosen_center[0], chosen_center[1], "*", ms=11, color="tab:blue")
    ax.set_title(
        f"{title}\n"
        f"a={a_m:.2f} m  b={b_m:.2f} m  dir={int(label['offset_direction'])}  "
        f"lam={label['offset_ratio']:.2f}  rho={label['point_inside_metric']:.3f}"
    )
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_path_ellipses(
    inflated: np.ndarray,
    path: np.ndarray,
    ellipses: Sequence[Tuple[Tuple[float, float], float, float, float]],
    out_path: str,
    title: str = "",
    resolution: float = 0.25,
    zoom: bool = True,
) -> None:
    """All valid ellipses of one path drawn on the full map."""
    fig, ax = plt.subplots(figsize=(8.5, 8.0))
    ax.imshow(inflated, cmap="gray_r", interpolation="nearest")
    ax.plot(path[:, 0], path[:, 1], "-", lw=1.0, color="tab:orange", label="A* path")
    for (c, a_m, b_m, ang) in ellipses:
        ax.add_patch(_ellipse_patch(c, a_m, b_m, ang, resolution,
                                    fill=False, ec="tab:blue", lw=0.9, alpha=0.85))
        ax.plot(c[0], c[1], ".", ms=2, color="tab:red")
    ax.plot([], [], color="tab:blue", label="safe ellipses")
    if zoom:
        margin = 70
        x0 = max(0, int(path[:, 0].min()) - margin); x1 = int(path[:, 0].max()) + margin
        y0 = max(0, int(path[:, 1].min()) - margin); y1 = int(path[:, 1].max()) + margin
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
