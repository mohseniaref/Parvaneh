"""Ghiglia--Romero weighted least-squares phase unwrapping backends."""

from dataclasses import dataclass
import importlib.util
import math
import numpy as np
from scipy.fft import dctn, idctn
from scipy.linalg.blas import ddot


def _wrap(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _edges(phase, weight):
    w2 = np.ones_like(phase) if weight is None else np.asarray(weight) ** 2
    wx = np.minimum(w2[:, :-1], w2[:, 1:])
    wy = np.minimum(w2[:-1, :], w2[1:, :])
    return wx, wy, wx * _wrap(np.diff(phase, axis=1)), wy * _wrap(np.diff(phase, axis=0))


def _divergence(x, y):
    out = np.empty((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
    out[:, 0], out[:, -1] = x[:, 0], -x[:, -1]
    out[:, 1:-1] = x[:, 1:] - x[:, :-1]
    out[0, :] += y[0, :]
    out[-1, :] -= y[-1, :]
    out[1:-1, :] += y[1:, :] - y[:-1, :]
    return out


def _apply_q_numpy(p, wx, wy):
    return _divergence(wx * np.diff(p, axis=1), wy * np.diff(p, axis=0))


def _poisson_scale(shape, dtype):
    rows, cols = shape
    iy = np.arange(rows, dtype=dtype)[:, None]
    ix = np.arange(cols, dtype=dtype)[None, :]
    scale = 2.0 * (np.cos(np.pi * iy / rows) + np.cos(np.pi * ix / cols) - 2.0)
    scale[0, 0] = 1.0
    return scale


def _dot_blas(a, b):
    # Explicit BLAS backend; ravel(order='K') avoids a copy for normal C arrays.
    return float(ddot(np.ravel(a, order="K"), np.ravel(b, order="K")))


def _apply_q_numba(p, wx, wy):
    from .numba_backend import apply_q
    return apply_q(p, wx, wy)


def available_backends():
    result = {"numpy": True, "blas": True, "numba": importlib.util.find_spec("numba") is not None,
              "cython": False, "cupy": False}
    try:
        from ._cython_backend import apply_q as _unused
        result["cython"] = True
    except ImportError:
        pass
    try:
        import cupy as cp
        result["cupy"] = cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        pass
    return result


@dataclass
class Info:
    backend: str
    iterations: int
    relative_residual: float
    converged: bool


def _unwrap_cpu(phase, weight, backend, max_iter, tol, workers):
    phase = np.ascontiguousarray(phase, dtype=np.float64)
    if phase.ndim != 2 or min(phase.shape) < 2:
        raise ValueError("phase must be a 2-D array with both dimensions >= 2")
    if weight is not None:
        weight = np.ascontiguousarray(weight, dtype=np.float64)
        if weight.shape != phase.shape or np.any(weight < 0) or not np.all(np.isfinite(weight)):
            raise ValueError("weight must be finite, nonnegative, and match phase.shape")
    wx, wy, bx, by = _edges(phase, weight)
    residual = _divergence(bx, by)
    initial = np.linalg.norm(residual)
    phi = np.zeros_like(phase)
    if initial == 0:
        return phi, Info(backend, 0, 0.0, True)
    scale = _poisson_scale(phase.shape, phase.dtype)
    apply_q, dot = _apply_q_numpy, lambda a, b: float(np.vdot(a, b))
    if backend == "numba":
        apply_q = _apply_q_numba
    elif backend == "cython":
        from ._cython_backend import apply_q
    elif backend == "blas":
        dot = _dot_blas
    elif backend != "numpy":
        raise ValueError(f"unknown CPU backend: {backend}")
    previous, direction, converged = None, None, False
    # ``relative`` and ``iterations`` are reported even when the loop stops for
    # a reason other than convergence, so they are defined before it starts.
    relative, iterations = float("inf"), 0
    for iteration in range(1, max_iter + 1):
        iterations = iteration
        z = idctn(dctn(residual, workers=workers) / scale, workers=workers)
        rz = dot(residual, z)
        direction = z.copy() if previous is None else z + (rz / previous) * direction
        q = apply_q(direction, wx, wy)
        denominator = dot(direction, q)
        if not math.isfinite(denominator) or abs(denominator) < np.finfo(float).tiny:
            break
        alpha = rz / denominator
        phi += alpha * direction
        residual -= alpha * q
        relative = np.linalg.norm(residual) / initial
        if relative <= tol:
            converged = True
            break
        previous = rz
    return phi, Info(backend, iterations, relative, converged)


def unwrap(phase, weight=None, *, backend="numpy", max_iter=100, tol=1e-8,
           workers=1, return_info=False):
    """Unwrap a 2-D wrapped phase image; the arbitrary output offset is zero mean.

    ``phase`` must be finite.  A no-data or zero-weight pixel has no value to
    unwrap, so replace such pixels with a finite placeholder (for example
    ``np.where(valid, phase, 0.0)``) and pass ``weight`` to exclude them.

    For CPU backends, ``workers=-1`` lets SciPy use all available cores for the
    DCT preconditioner.  The default is one worker to avoid oversubscription in
    applications that already parallelize at a higher level.
    """
    phase = np.asarray(phase, dtype=np.float64)
    if phase.ndim != 2 or min(phase.shape) < 2:
        raise ValueError("phase must be a 2-D array with both dimensions >= 2")
    if not np.isfinite(phase).all():
        raise ValueError(
            "phase must be finite; a NaN or infinite pixel has no value to "
            "unwrap. Replace invalid pixels with a finite placeholder and give "
            "them zero weight, for example "
            "unwrap(np.where(valid, phase, 0.0), valid.astype(float)).")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    if backend == "cupy":
        from .cupy_backend import unwrap_cupy
        result, info = unwrap_cupy(phase, weight, max_iter, tol)
    else:
        result, info = _unwrap_cpu(phase, weight, backend, max_iter, tol, workers)
    result -= result.mean()
    return (result, info) if return_info else result
