"""Fixed-centre maximum-volume inscribed ellipse (MVIE) kernels.

Two interchangeable backends solve the same convex program with the ellipse
centre ``c`` held fixed::

    maximise   log det L
    s.t.       a_j^T c + ||L^T a_j||_2 <= b_j        (half-planes)
               L >= eps * I                           (numerical floor)

* ``barrier``: a direct log-barrier / damped-Newton solver on the three free
  entries of ``L`` (pure NumPy, ~1 ms per solve).  Used for production
  throughput.
* ``cvxpy``: CLARABEL (SCS fallback), the reference solve required by the
  implementation plan.  Used for verification and as the fallback whenever the
  barrier solver fails to converge.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Fast barrier backend
# --------------------------------------------------------------------------- #
def _unpack(v: np.ndarray) -> np.ndarray:
    return np.array([[v[0], v[1]], [v[1], v[2]]], dtype=float)


def _objective(v: np.ndarray, A: np.ndarray, r: np.ndarray, mu: float) -> float:
    p, q, s = v
    D = p * s - q * q
    if D <= 0:
        return np.inf
    u1 = p * A[:, 0] + q * A[:, 1]
    u2 = q * A[:, 0] + s * A[:, 1]
    g = r - np.sqrt(u1 * u1 + u2 * u2)
    if np.any(g <= 0):
        return np.inf
    return -np.log(D) - mu * float(np.sum(np.log(g)))


def _g_values(v: np.ndarray, A: np.ndarray, r: np.ndarray) -> np.ndarray:
    p, q, s = v
    u1 = p * A[:, 0] + q * A[:, 1]
    u2 = q * A[:, 0] + s * A[:, 1]
    return r - np.sqrt(u1 * u1 + u2 * u2)


def _newton_central_path(v0: np.ndarray, A: np.ndarray, r: np.ndarray,
                         mus: Tuple[float, ...], max_newton: int) -> Tuple[np.ndarray | None, str]:
    """Damped-Newton central path for one barrier schedule, starting at ``v0``."""
    v = v0.copy()
    status = "ok"
    for mu in mus:
        for _ in range(max_newton):
            p, q, s = v
            D = p * s - q * q
            if D <= 1e-18:
                return None, "lost_positivity"
            u1 = p * A[:, 0] + q * A[:, 1]
            u2 = q * A[:, 0] + s * A[:, 1]
            norm_u = np.sqrt(u1 * u1 + u2 * u2)
            g = r - norm_u
            if np.any(g <= 0):
                return None, "lost_feasibility"
            grad_D = np.array([s, -2.0 * q, p])
            jac = np.stack([u1 * A[:, 0],
                            u1 * A[:, 1] + u2 * A[:, 0],
                            u2 * A[:, 1]], axis=1) / norm_u[:, None]
            grad = -grad_D / D - mu * np.sum(jac / g[:, None], axis=0)
            hess_D = np.array([[0.0, 0.0, 1.0], [0.0, -2.0, 0.0], [1.0, 0.0, 0.0]])
            W = jac / g[:, None]
            hess = np.outer(grad_D, grad_D) / (D * D) - hess_D / D + mu * (W.T @ W)
            scale = float(np.trace(np.abs(hess))) / 3.0
            hess = hess + max(1e-14, 1e-12 * scale) * np.eye(3)
            try:
                step = -np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                step = -np.linalg.lstsq(hess, grad, rcond=None)[0]
            if not np.all(np.isfinite(step)):
                return None, "nonfinite_step"
            f_cur = _objective(v, A, r, mu)
            slope = float(grad @ step)
            t = 1.0
            accepted = False
            for _ls in range(50):
                v_new = v + t * step
                f_new = _objective(v_new, A, r, mu)
                if np.isfinite(f_new) and f_new <= f_cur + 1e-4 * t * slope:
                    accepted = True
                    break
                t *= 0.5
            if not accepted:
                status = "line_search_failed"
                break
            v = v_new
            if np.linalg.norm(t * step) < 1e-11:
                break
    return v, status


def solve_mvie_barrier(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    min_eigenvalue: float = 1e-3,
    max_newton: int = 40,
    barrier_mus: Tuple[float, ...] = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6),
    init_L: np.ndarray | None = None,
) -> Tuple[np.ndarray | None, str]:
    """Log-barrier interior-point solve of the fixed-centre MVIE.

    Returns ``(L, status)`` with ``L`` symmetric positive definite (shape matrix
    in cells) or ``None`` when the centre is infeasible / the method fails.
    ``init_L`` warm-starts from a previous (shrunk) solution; a cold start is
    retried if the warm-started run breaks down.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    r_raw = b - A @ c
    keep = r_raw > 1e-9
    if not np.any(keep):
        return None, "center_infeasible"
    A = A[keep]
    r_raw = r_raw[keep]

    norms = np.linalg.norm(A, axis=1)
    good = norms > 1e-12
    if not np.any(good):
        return None, "no_constraints"
    A = A[good] / norms[good, None]
    r = r_raw[good] / norms[good]
    if np.any(r <= 0):
        return None, "center_infeasible"

    def is_feasible(v: np.ndarray) -> bool:
        p, q, s = v
        if p <= 0 or s <= 0 or p * s - q * q <= 0:
            return False
        return bool(np.all(_g_values(v, A, r) > 0))

    starts: list[np.ndarray] = []
    if init_L is not None:
        base = np.array([init_L[0, 0], init_L[0, 1], init_L[1, 1]], dtype=float)
        for shrink in (0.98, 0.9, 0.8, 0.6, 0.45, 0.3, 0.2, 0.1, 0.05, 0.01):
            trial = base * shrink
            if is_feasible(trial):
                starts.append(trial)
                break
    start = min(float(np.min(r)) * 0.5, 1.0)
    start = max(start, 1e-9)
    cold = np.array([start, 0.0, start])
    while not is_feasible(cold) and cold[0] > 1e-14:
        cold = cold * 0.5
    if is_feasible(cold):
        starts.append(cold)
    if not starts:
        return None, "no_feasible_start"

    last_status = "failed"
    for v0 in starts:
        v, status = _newton_central_path(v0, A, r, barrier_mus, max_newton)
        last_status = status
        if v is None:
            continue
        L = _unpack(v)
        if not np.all(np.isfinite(L)) or np.linalg.eigvalsh(L).min() <= 0:
            last_status = "not_positive_definite"
            continue
        if not np.all(_g_values(v, A, r) > -1e-9):
            last_status = "final_infeasible"
            continue
        return L, "barrier_optimal" if status == "ok" else f"barrier_{status}"
    return None, last_status


# --------------------------------------------------------------------------- #
# cvxpy (CLARABEL / SCS) reference backend
# --------------------------------------------------------------------------- #
class CvxpyMvieSolver:
    """Reference MVIE solve with CLARABEL (SCS fallback) on a compiled problem."""

    def __init__(self, capacity: int, min_eigenvalue: float = 1e-3,
                 solver_primary: str = "CLARABEL", solver_backup: str = "SCS"):
        import cvxpy as cp

        self.cp = cp
        self.capacity = int(capacity)
        self.solver_primary = solver_primary
        self.solver_backup = solver_backup
        self._A = cp.Parameter((self.capacity, 2))
        self._b = cp.Parameter(self.capacity)
        self._L = cp.Variable((2, 2), symmetric=True)
        # The centre is folded into the right-hand side (b - A c) so the problem
        # stays DPP and cvxpy can reuse one compilation for every solve.
        constraints = [
            cp.norm(self._L.T @ self._A[j]) <= self._b[j]
            for j in range(self.capacity)
        ]
        constraints.append(self._L - float(min_eigenvalue) * np.eye(2) >> 0)
        self._problem = cp.Problem(cp.Maximize(cp.log_det(self._L)), constraints)

    def solve(self, A: np.ndarray, b: np.ndarray, c: np.ndarray,
              min_eigenvalue: float = 1e-3) -> Tuple[np.ndarray | None, str, str]:
        n = A.shape[0]
        if n > self.capacity:
            raise ValueError(f"polytope has {n} rows, capacity is {self.capacity}")
        c = np.asarray(c, dtype=float)
        A_pad = np.zeros((self.capacity, 2))
        b_pad = np.ones(self.capacity)
        A_pad[:n] = A
        b_pad[:n] = b - A @ c          # shift the polytope so the centre is the origin
        self._A.value = A_pad
        self._b.value = b_pad
        for solver in (self.solver_primary, self.solver_backup):
            try:
                self._problem.solve(solver=solver, verbose=False)
            except Exception:
                continue
            if self._problem.status in ("optimal", "optimal_inaccurate") and self._L.value is not None:
                L_val = np.asarray(self._L.value, dtype=float)
                L_val = (L_val + L_val.T) / 2.0
                if np.all(np.isfinite(L_val)) and np.linalg.eigvalsh(L_val).min() > 0:
                    return L_val, self._problem.status, solver
        return None, str(getattr(self._problem, "status", "unknown")), ""
