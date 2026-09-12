"""Quality measures used by two-dimensional path-following algorithms.

The functions operate in radians and return larger values for more reliable
pixels.  They are independent of the historical reference implementation and
can therefore be distributed with this package.
"""

import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter


def _validate_phase(phase):
    phase = np.asarray(phase, dtype=np.float64)
    if phase.ndim != 2 or min(phase.shape) < 2 or not np.isfinite(phase).all():
        raise ValueError("phase must be a finite 2-D array with dimensions >= 2")
    return phase


def wrapped_gradients(phase):
    """Return wrapped forward differences ``(dx, dy)`` in radians."""
    phase = _validate_phase(phase)
    dx = (np.diff(phase, axis=1) + np.pi) % (2 * np.pi) - np.pi
    dy = (np.diff(phase, axis=0) + np.pi) % (2 * np.pi) - np.pi
    return dx, dy


def max_gradient_quality(phase, window=1):
    """Quality inverse to the largest local wrapped phase derivative.

    The result is linearly scaled to ``[0, 1]``.  ``window=1`` uses the two
    forward derivatives incident on each pixel, including mirrored backward
    differences on the last row and column as in the reference formulation.
    """
    phase = _validate_phase(phase)
    if window < 1 or int(window) != window:
        raise ValueError("window must be a positive integer")
    dx, dy = wrapped_gradients(phase)
    gx = np.empty_like(phase)
    gy = np.empty_like(phase)
    gx[:, :-1], gx[:, -1] = dx, -dx[:, -1]
    gy[:-1, :], gy[-1, :] = dy, -dy[-1, :]
    cost = np.abs(gx) + np.abs(gy)
    if window > 1:
        cost = maximum_filter(cost, size=int(window), mode="nearest")
    span = np.ptp(cost)
    return np.ones_like(cost) if span == 0 else (cost.max() - cost) / span


def pseudocorrelation_quality(phase, window=3):
    """Local phase phasor coherence, between zero and one."""
    phase = _validate_phase(phase)
    if window < 1 or int(window) != window:
        raise ValueError("window must be a positive integer")
    real = uniform_filter(np.cos(phase), size=int(window), mode="nearest")
    imag = uniform_filter(np.sin(phase), size=int(window), mode="nearest")
    return np.hypot(real, imag)


def derivative_variance_quality(phase, window=3):
    """Quality inverse to local variance of wrapped x/y derivatives."""
    phase = _validate_phase(phase)
    if window < 1 or int(window) != window:
        raise ValueError("window must be a positive integer")
    dx, dy = wrapped_gradients(phase)
    gx = np.pad(dx, ((0, 0), (0, 1)), mode="edge")
    gy = np.pad(dy, ((0, 1), (0, 0)), mode="edge")
    size = int(window)
    var = (uniform_filter(gx * gx, size=size) - uniform_filter(gx, size=size) ** 2
           + uniform_filter(gy * gy, size=size) - uniform_filter(gy, size=size) ** 2)
    var = np.maximum(var, 0)
    span = np.ptp(var)
    return np.ones_like(var) if span == 0 else (var.max() - var) / span
