"""Generic binary-array input/output helpers.

The public API requires callers to state shape and dtype explicitly. It does
not infer conventions from a third-party filename or depend on a particular
dataset layout.
"""

from pathlib import Path

import numpy as np


def read_raw_raster(path, shape, dtype="<f4", *, scale=1.0, offset=0.0,
                    order="C"):
    """Read a headerless 2-D raster with explicit conventions.

    ``shape`` is ``(rows, columns)``. ``dtype`` should include byte order when
    relevant. Optional physical-value conversion is ``value * scale + offset``.
    """
    if len(shape) != 2 or any(int(size) <= 0 for size in shape):
        raise ValueError("shape must contain two positive dimensions")
    if order not in ("C", "F"):
        raise ValueError("order must be 'C' or 'F'")
    rows, cols = map(int, shape)
    data = np.fromfile(str(Path(path)), dtype=np.dtype(dtype))
    expected = rows * cols
    if data.size != expected:
        raise ValueError(
            "expected {} values for shape {}, found {}".format(
                expected, (rows, cols), data.size))
    array = data.reshape((rows, cols), order=order)
    if scale != 1.0 or offset != 0.0:
        array = array.astype(np.float64) * scale + offset
    return array


def write_raw_raster(path, array, dtype="<f4", *, order="C"):
    """Write a finite 2-D array as a headerless raster and return its path."""
    values = np.asarray(array)
    if values.ndim != 2:
        raise ValueError("array must be two-dimensional")
    if not np.isfinite(values).all():
        raise ValueError("array contains NaN or infinity")
    if order not in ("C", "F"):
        raise ValueError("order must be 'C' or 'F'")
    output = Path(path)
    flattened = np.asarray(values, dtype=np.dtype(dtype)).ravel(order=order)
    flattened.tofile(str(output))
    return output

