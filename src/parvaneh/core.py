"""Ghiglia--Romero weighted least-squares phase unwrapping backends."""

from dataclasses import dataclass
import importlib.util
import math
import numpy as np
from scipy.fft import dctn, idctn
from scipy.linalg.blas import ddot


def _wrap(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _axis_slice(array, axis, item):
    """``array`` with ``item`` applied to ``axis`` and nothing changed elsewhere."""
    return array[(slice(None),) * axis + (item,)]


def _edges(phase, weight):
    """Per-axis edge weights and weighted wrapped differences.

    Two tuples come back, both ordered by axis: element ``k`` is one sample
    shorter than ``phase`` along axis ``k`` and holds the edges that join
    neighbours along that axis.  An edge weight is the smaller of the weights of
    its two endpoint pixels, so a weight of zero removes the edge from the
    problem entirely.  Weights are squared first, which is the convention that
    makes the matrix-free solver below implement the weighted formulation.
    """
    w2 = np.ones_like(phase) if weight is None else np.asarray(weight) ** 2
    weights = tuple(
        np.minimum(_axis_slice(w2, axis, slice(None, -1)),
                   _axis_slice(w2, axis, slice(1, None)))
        for axis in range(phase.ndim))
    differences = tuple(w * _wrap(np.diff(phase, axis=axis))
                        for axis, w in enumerate(weights))
    return weights, differences


def _divergence(fields):
    """Backward-difference divergence of one edge field per axis.

    This is ``-D^T`` for the forward-difference operator ``D``, so that
    ``_divergence(_apply_q_numpy(p, w))`` is the weighted discrete Neumann
    Laplacian of ``p``.  Each axis contributes its first edge to the boundary
    sample at index ``0``, its last edge to the boundary sample at the far end,
    and every interior edge to the sample it leaves behind.  Axes are visited
    from the last to the first, matching the order of the two-dimensional
    original so that the arithmetic is unchanged.
    """
    fields = tuple(fields)
    shape = list(fields[0].shape)
    shape[0] += 1
    out = np.empty(shape, dtype=fields[0].dtype)
    first_axis = True
    for axis in reversed(range(len(fields))):
        field = fields[axis]
        length = shape[axis]
        head = _axis_slice(out, axis, slice(0, 1))
        tail = _axis_slice(out, axis, slice(length - 1, length))
        start = _axis_slice(field, axis, slice(0, 1))
        stop = _axis_slice(field, axis, slice(length - 2, length - 1))
        if first_axis:
            head[...] = start
            tail[...] = -stop
        else:
            head += start
            tail -= stop
        if length > 2:
            interior = (_axis_slice(field, axis, slice(1, length - 1))
                        - _axis_slice(field, axis, slice(0, length - 2)))
            target = _axis_slice(out, axis, slice(1, length - 1))
            if first_axis:
                target[...] = interior
            else:
                target += interior
        first_axis = False
    return out


def _apply_q_numpy(p, weights):
    return _divergence(tuple(w * np.diff(p, axis=axis)
                             for axis, w in enumerate(weights)))


def _poisson_scale(shape, dtype):
    """Divisor of the DCT preconditioner: the Neumann Laplacian eigenvalues.

    For a grid of extent ``(rows, cols)`` this is
    ``2 * (cos(pi * iy / rows) + cos(pi * ix / cols) - 2)``, with the constant
    mode replaced by one so that it stays invertible.
    """
    total = None
    for axis, size in enumerate(shape):
        view = [1] * len(shape)
        view[axis] = size
        term = np.cos(np.pi * np.arange(size, dtype=dtype) / size)
        total = term.reshape(view) if total is None else total + term.reshape(view)
    scale = 2.0 * (total - len(shape))
    scale[(0,) * len(shape)] = 1.0
    return scale


def _check_phase(phase):
    """Return ``phase`` as a contiguous float64 array, or raise ``ValueError``.

    Every axis must be at least two samples long, because one sample along an
    axis carries no edge to compare with and the problem would be empty.
    """
    phase = np.ascontiguousarray(phase, dtype=np.float64)
    if phase.ndim < 1 or min(phase.shape) < 2:
        raise ValueError("phase must be an array with at least one dimension, "
                         "and every axis must hold at least 2 samples")
    return phase


def _resolve_engine(backend, ndim, weights):
    """Return ``(apply_q, dot, name)`` for ``backend`` on an ``ndim`` grid.

    The compiled kernels are written for two-dimensional arrays, so any other
    dimension quietly falls back to the NumPy engine; the returned name is the
    engine that will really run, which is what ``Info.backend`` reports.
    """
    dot = lambda a, b: float(np.vdot(a, b))
    if backend == "numpy":
        return lambda p: _apply_q_numpy(p, weights), dot, "numpy"
    if backend == "blas":
        return lambda p: _apply_q_numpy(p, weights), _dot_blas, "blas"
    if backend == "numba":
        if ndim == 2:
            from .numba_backend import apply_q as kernel
            return lambda p: kernel(p, weights[1], weights[0]), dot, "numba"
        return lambda p: _apply_q_numpy(p, weights), dot, "numpy"
    if backend == "cython":
        if ndim == 2:
            from ._cython_backend import apply_q as kernel
            return lambda p: kernel(p, weights[1], weights[0]), dot, "cython"
        return lambda p: _apply_q_numpy(p, weights), dot, "numpy"
    raise ValueError("unknown CPU backend: %s" % (backend,))


def _dot_blas(a, b):
    # Explicit BLAS backend; ravel(order='K') avoids a copy for normal C arrays.
    return float(ddot(np.ravel(a, order="K"), np.ravel(b, order="K")))


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
    phase = _check_phase(phase)
    if weight is not None:
        weight = np.ascontiguousarray(weight, dtype=np.float64)
        if weight.shape != phase.shape or np.any(weight < 0) or not np.all(np.isfinite(weight)):
            raise ValueError("weight must be finite, nonnegative, and match phase.shape")
    weights, differences = _edges(phase, weight)
    apply_q, dot, backend = _resolve_engine(backend, phase.ndim, weights)
    residual = _divergence(differences)
    initial = np.linalg.norm(residual)
    phi = np.zeros_like(phase)
    if initial == 0:
        return phi, Info(backend, 0, 0.0, True)
    scale = _poisson_scale(phase.shape, phase.dtype)
    previous, direction, converged = None, None, False
    # ``relative`` and ``iterations`` are reported even when the loop stops for
    # a reason other than convergence, so they are defined before it starts.
    relative, iterations = float("inf"), 0
    for iteration in range(1, max_iter + 1):
        iterations = iteration
        z = idctn(dctn(residual, workers=workers) / scale, workers=workers)
        rz = dot(residual, z)
        direction = z.copy() if previous is None else z + (rz / previous) * direction
        q = apply_q(direction)
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
    """Unwrap a wrapped phase array of any dimension; the output offset is zero mean.

    The result is the array whose wrapped gradients are closest to those of
    ``phase`` in the least-squares sense, which is the Poisson formulation of
    Ghiglia and Romero.  Every axis is treated the same way, so a three-
    dimensional stack of images is solved as one volume: the answer at one
    acquisition is constrained by its neighbours in time.  A one-dimensional
    array is accepted as well, which is the classic one-dimensional line
    integral.

    The compiled kernels (``"numba"``, ``"cython"``) and the GPU backend
    (``"cupy"``) are written for two-dimensional arrays.  Any other number of
    dimensions runs on the NumPy engine instead, and ``Info.backend`` reports
    the engine that really ran, so the substitution is never silent.

    ``phase`` must be finite.  A no-data or zero-weight pixel has no value to
    unwrap, so replace such pixels with a finite placeholder (for example
    ``np.where(valid, phase, 0.0)``) and pass ``weight`` to exclude them.

    For CPU backends, ``workers=-1`` lets SciPy use all available cores for the
    DCT preconditioner.  The default is one worker to avoid oversubscription in
    applications that already parallelize at a higher level.
    """
    phase = np.asarray(phase, dtype=np.float64)
    if phase.ndim < 1 or min(phase.shape) < 2:
        raise ValueError("phase must be an array with at least one dimension, "
                         "and every axis must hold at least 2 samples")
    if not np.isfinite(phase).all():
        raise ValueError(
            "phase must be finite; a NaN or infinite pixel has no value to "
            "unwrap. Replace invalid pixels with a finite placeholder and give "
            "them zero weight, for example "
            "unwrap(np.where(valid, phase, 0.0), valid.astype(float)).")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    if backend == "cupy" and phase.ndim == 2:
        from .cupy_backend import unwrap_cupy
        result, info = unwrap_cupy(phase, weight, max_iter, tol)
    else:
        engine = "numpy" if backend == "cupy" else backend
        result, info = _unwrap_cpu(phase, weight, engine, max_iter, tol, workers)
    result -= result.mean()
    return (result, info) if return_info else result
