"""CUDA backend. Imported only when explicitly requested."""

import math
import cupy as cp
from cupyx.scipy.fft import dctn, idctn
from .core import Info


def unwrap_cupy(phase, weight, max_iter, tol):
    p = cp.asarray(phase, dtype=cp.float64)
    w2 = cp.ones_like(p) if weight is None else cp.asarray(weight, dtype=cp.float64) ** 2
    if p.ndim != 2 or w2.shape != p.shape:
        raise ValueError("phase and weight must be matching 2-D arrays")
    wx, wy = cp.minimum(w2[:, :-1], w2[:, 1:]), cp.minimum(w2[:-1], w2[1:])
    wrap = lambda x: (x + cp.pi) % (2 * cp.pi) - cp.pi
    # A^T is a difference after adding zero-valued boundary edges.
    bx = wx * wrap(cp.diff(p, axis=1)); by = wy * wrap(cp.diff(p, axis=0))
    residual = cp.diff(cp.pad(bx, ((0, 0), (1, 1))), axis=1) + cp.diff(cp.pad(by, ((1, 1), (0, 0))), axis=0)
    initial = float(cp.linalg.norm(residual).get()); phi = cp.zeros_like(p)
    rows, cols = p.shape
    iy, ix = cp.arange(rows)[:, None], cp.arange(cols)[None, :]
    scale = 2 * (cp.cos(cp.pi * iy / rows) + cp.cos(cp.pi * ix / cols) - 2); scale[0, 0] = 1
    previous = direction = None; relative = 0.0; converged = initial == 0
    for iteration in range(1, max_iter + 1):
        z = idctn(dctn(residual) / scale)
        rz = float(cp.vdot(residual, z).get())
        direction = z.copy() if previous is None else z + (rz / previous) * direction
        q = cp.diff(cp.pad(wx * cp.diff(direction, axis=1), ((0, 0), (1, 1))), axis=1)
        q += cp.diff(cp.pad(wy * cp.diff(direction, axis=0), ((1, 1), (0, 0))), axis=0)
        denom = float(cp.vdot(direction, q).get())
        if not math.isfinite(denom) or abs(denom) < 1e-300: break
        alpha = rz / denom; phi += alpha * direction; residual -= alpha * q
        relative = float(cp.linalg.norm(residual).get()) / initial
        if relative <= tol: converged = True; break
        previous = rz
    return cp.asnumpy(phi), Info("cupy", iteration, relative, converged)
