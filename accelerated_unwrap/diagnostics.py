"""Correctness and discontinuity diagnostics for unwrapped surfaces."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter


@dataclass
class DifferenceMetrics:
    mean_offset: float
    rmse_aligned: float
    mean_absolute_aligned: float
    evaluated_pixels: int


def surface_difference(first, second, mask=None, *, return_metrics=False):
    """Return ``first-second`` after removing its valid-pixel mean offset."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("first and second must be equally shaped 2-D arrays")
    valid = np.ones(first.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if valid.shape != first.shape or not valid.any():
        raise ValueError("mask must match the surfaces and contain a valid pixel")
    raw = first - second
    offset = float(raw[valid].mean())
    difference = raw - offset
    difference[~valid] = -offset  # reproduces the historical diagnostic file
    selected = difference[valid]
    metrics = DifferenceMetrics(offset, float(np.sqrt(np.mean(selected * selected))),
                                float(np.mean(np.abs(selected))), int(valid.sum()))
    return (difference, metrics) if return_metrics else difference


def discontinuity_map(surface, mask=None, *, threshold=np.pi,
                      return_fraction=False):
    """Mark cells whose right or downward surface difference exceeds threshold."""
    surface = np.asarray(surface, dtype=np.float64)
    if surface.ndim != 2 or min(surface.shape) < 2:
        raise ValueError("surface must be a 2-D array with dimensions >= 2")
    valid = np.ones(surface.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if valid.shape != surface.shape:
        raise ValueError("mask must match surface.shape")
    if mask is not None:
        invalid = maximum_filter((~valid).astype(np.uint8), size=3) != 0
        valid = ~invalid
    cells_valid = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1]
    jumps = ((np.abs(np.diff(surface, axis=1)[:-1, :]) > threshold)
             | (np.abs(np.diff(surface, axis=0)[:, :-1]) > threshold))
    result = np.zeros(surface.shape, dtype=bool)
    result[:-1, :-1] = jumps & cells_valid
    denominator = int(cells_valid.sum())
    fraction = float(result.sum() / denominator) if denominator else float("nan")
    return (result, fraction) if return_fraction else result
