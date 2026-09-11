"""Minimum-norm phase-unwrapping algorithms."""

from dataclasses import dataclass
import math

import numpy as np
from scipy.fft import dctn, idctn

from .core import (_apply_q_numpy, _divergence, _poisson_scale, _wrap,
                   unwrap)


@dataclass
class LpInfo:
    p: float
    outer_iterations: int
    inner_iterations: int
    objective: float
    relative_change: float


def _solve_edge_weighted(gx, gy, wx, wy, initial, max_iter, tol):
    """PCG solve with independent horizontal and vertical edge weights."""
    rhs = _divergence(wx * gx, wy * gy)
    phi = np.asarray(initial, dtype=np.float64).copy()
    phi -= phi.mean()
    residual = rhs - _apply_q_numpy(phi, wx, wy)
    norm0 = np.linalg.norm(residual)
    if norm0 == 0:
        return phi, 0
    scale = _poisson_scale(phi.shape, phi.dtype)
    direction = None
    previous = None
    for iteration in range(1, max_iter + 1):
        z = idctn(dctn(residual) / scale)
        rz = float(np.vdot(residual, z))
        direction = z.copy() if direction is None else z + (rz / previous) * direction
        q = _apply_q_numpy(direction, wx, wy)
        denominator = float(np.vdot(direction, q))
        if not math.isfinite(denominator) or abs(denominator) < np.finfo(float).tiny:
            break
        phi += (rz / denominator) * direction
        residual -= (rz / denominator) * q
        if np.linalg.norm(residual) <= tol * norm0:
            break
        previous = rz
    phi -= phi.mean()
    return phi, iteration


def unwrap_lp(phase, p=1.2, *, epsilon=1e-3, outer_iter=12,
              inner_iter=100, tol=1e-7, return_info=False):
    """Minimum-``Lp`` unwrapping by iteratively reweighted least squares.

    ``p=2`` reduces to ordinary unweighted least squares.  For ``1 <= p < 2``
    the method increasingly preserves sparse discontinuities. ``epsilon`` is
    expressed in radians and prevents singular weights at zero residual.
    """
    phase = np.asarray(phase, dtype=np.float64)
    if phase.ndim != 2 or min(phase.shape) < 2 or not np.isfinite(phase).all():
        raise ValueError("phase must be a finite 2-D array with dimensions >= 2")
    if not (1 <= p <= 2):
        raise ValueError("p must lie in [1, 2]")
    if epsilon <= 0 or outer_iter < 1 or inner_iter < 1 or tol <= 0:
        raise ValueError("epsilon, iteration counts, and tol must be positive")
    phi = unwrap(phase, backend="numpy")
    gx = _wrap(np.diff(phase, axis=1))
    gy = _wrap(np.diff(phase, axis=0))
    total_inner = 0
    relative_change = 0.0
    if p == 2:
        objective = float(np.sum((np.diff(phi, axis=1) - gx) ** 2)
                          + np.sum((np.diff(phi, axis=0) - gy) ** 2))
        info = LpInfo(p, 0, 1, objective, 0.0)
        return (phi, info) if return_info else phi
    exponent = p / 2.0 - 1.0
    for outer in range(1, outer_iter + 1):
        rx = np.diff(phi, axis=1) - gx
        ry = np.diff(phi, axis=0) - gy
        wx = (rx * rx + epsilon * epsilon) ** exponent
        wy = (ry * ry + epsilon * epsilon) ** exponent
        updated, inner = _solve_edge_weighted(gx, gy, wx, wy, phi, inner_iter, tol)
        total_inner += inner
        denominator = max(np.linalg.norm(phi), np.finfo(float).eps)
        relative_change = float(np.linalg.norm(updated - phi) / denominator)
        phi = updated
        if relative_change <= tol:
            break
    rx = np.diff(phi, axis=1) - gx
    ry = np.diff(phi, axis=0) - gy
    objective = float(np.sum((rx * rx + epsilon * epsilon) ** (p / 2.0))
                      + np.sum((ry * ry + epsilon * epsilon) ** (p / 2.0)))
    info = LpInfo(p, outer, total_inner, objective, relative_change)
    return (phi, info) if return_info else phi
