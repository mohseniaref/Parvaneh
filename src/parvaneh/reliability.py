"""Reliability-sorting phase unwrapping in any number of dimensions."""

from dataclasses import dataclass

import numpy as np

from .core import _axis_slice, _wrap


@dataclass
class ReliabilityInfo:
    """Bookkeeping for one reliability-sorting run."""

    backend: str
    pixels: int
    edges: int
    merges: int
    discarded: int
    components: int


def _confidence(phase, mask, weight):
    """Combine ``mask`` and ``weight`` into one per-pixel confidence array."""
    confidence = np.ones(phase.shape, dtype=np.float64)
    if weight is not None:
        weight = np.asarray(weight, dtype=np.float64)
        if weight.shape != phase.shape or not np.isfinite(weight).all():
            raise ValueError("weight must be finite and match phase.shape")
        if np.any(weight < 0):
            raise ValueError("weight must be nonnegative")
        confidence = weight.copy()
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != phase.shape:
            raise ValueError("mask must match phase.shape")
        confidence[~mask] = 0.0
    return confidence


def _check_phase(phase):
    phase = np.ascontiguousarray(phase, dtype=np.float64)
    if phase.ndim < 1 or min(phase.shape) < 2 or not np.isfinite(phase).all():
        raise ValueError(
            "phase must be a finite array with at least one dimension and at "
            "least 2 samples along every axis")
    return phase


def _axis_differences(phase, axis):
    """Return the wrapped step and the turn count between neighbours.

    ``wrap_index`` counts the whole turns removed from the raw phase
    difference, so the wrapped step is the raw difference minus ``2 pi`` times
    that count.  The rule only compares the raw difference against ``+pi`` and
    ``-pi``, so no rounding decision is ever taken.
    """
    raw = np.diff(phase, axis=axis)
    wrap_index = ((raw > np.pi).astype(np.int8)
                  - (raw <= -np.pi).astype(np.int8))
    return _wrap(raw), wrap_index


def _axis_confidence(confidence, axis):
    """Confidence shared by every pair of neighbours along one axis."""
    return np.minimum(_axis_slice(confidence, axis, slice(0, -1)),
                      _axis_slice(confidence, axis, slice(1, None)))


def pixel_reliability(phase, weight=None, mask=None):
    """Rate every pixel by how trustworthy its phase looks.

    The rating is ``1 / (1 + S)`` where ``S`` is the confidence-weighted sum of
    the sizes of the wrapped phase steps between the pixel and its immediate
    neighbours.  A pixel sitting in a smooth area therefore scores close to
    one, while a pixel crossed by a steep fringe or by noise scores lower.
    Pixels excluded by ``mask`` or carrying zero ``weight`` are reported as
    ``NaN``, because they have no rating at all.
    """
    phase = _check_phase(phase)
    confidence = _confidence(phase, mask, weight)
    step_sum = np.zeros(phase.shape, dtype=np.float64)
    for axis in range(phase.ndim):
        step, _ = _axis_differences(phase, axis)
        contribution = np.abs(step) * _axis_confidence(confidence, axis)
        for item in (slice(0, -1), slice(1, None)):
            spread = np.zeros(phase.shape, dtype=np.float64)
            _axis_slice(spread, axis, item)[:] = contribution
            step_sum += spread
    rating = 1.0 / (1.0 + step_sum)
    rating[confidence <= 0.0] = np.nan
    return rating


def _merge_python(node_a, node_b, wrap_index, size, parent, shift):
    """Reference union-find merge; bit-identical to the numba kernel."""
    merges = 0
    discarded = 0
    for index in range(node_a.size):
        a = int(node_a[index])
        b = int(node_b[index])
        turns_a = 0
        root_a = a
        while int(parent[root_a]) != root_a:
            turns_a += int(shift[root_a])
            root_a = int(parent[root_a])
        turns_b = 0
        root_b = b
        while int(parent[root_b]) != root_b:
            turns_b += int(shift[root_b])
            root_b = int(parent[root_b])
        if root_a == root_b:
            discarded += 1
            continue
        relation = turns_a - turns_b - int(wrap_index[index])
        if size[root_a] >= size[root_b]:
            parent[root_b] = root_a
            shift[root_b] = relation
            size[root_a] += size[root_b]
        else:
            parent[root_a] = root_b
            shift[root_a] = -relation
            size[root_b] += size[root_a]
        merges += 1
    return merges, discarded


def _accumulated_turns(parent, shift):
    """Turn count from every pixel up to its region root, as an integer."""
    node = np.arange(parent.size, dtype=np.int64)
    turns = np.zeros(parent.size, dtype=np.int64)
    while True:
        moving = parent[node] != node
        if not moving.any():
            return turns
        turns += shift[node]
        node = parent[node]


def reliability_unwrap(phase, mask=None, weight=None, *, backend="python",
                       return_info=False):
    """Unwrap an array of any dimension by sorting edges by reliability.

    The method is the reliability-sorting idea of Herraez and co-workers: rate
    every pixel (see :func:`pixel_reliability`), list every pair of neighbouring
    pixels, and visit those pairs from the most to the least reliable.  A pair
    is used only when its two pixels are not yet part of the same region, so the
    used pairs form a spanning tree of each region: exactly ``pixels - 1`` pairs
    per region regardless of how many residues the phase contains.  Pairs that
    would close a loop are listed as ``discarded`` and their inconsistency is
    left behind, which is what makes the method immune to residues instead of
    merely smoothing them away.

    Because the pairs are found along the axes of the array, the algorithm does
    not care how many axes there are.  A three-dimensional stack is unwrapped as
    a whole volume, so a pixel that is noisy in one acquisition can still be
    reached through its neighbours in space and in time.

    The output keeps the wrapped values of the input: ``u`` satisfies
    ``wrap(u) == wrap(phase)``, and every used pair satisfies
    ``u[b] - u[a] == wrap(phase[b] - phase[a])``.  The overall offset of a
    region is anchored at the first pixel the region grew from, and separate
    regions get independent offsets.  Pixels that are excluded by ``mask`` or
    carry zero ``weight`` are returned as ``NaN``.

    ``backend="numba"`` runs the merge loop in compiled code and gives exactly
    the same numbers as ``backend="python"``; the Python loop is a reference
    implementation and is slow for large volumes.
    """
    phase = _check_phase(phase)
    confidence = _confidence(phase, mask, weight)
    if backend not in ("python", "numba"):
        raise ValueError("backend must be 'python' or 'numba'")
    base = np.arange(phase.size, dtype=np.int64).reshape(phase.shape)
    node_a, node_b, wrap_index = [], [], []
    for axis in range(phase.ndim):
        _, turns = _axis_differences(phase, axis)
        usable = _axis_confidence(confidence, axis) > 0.0
        node_a.append(_axis_slice(base, axis, slice(0, -1))[usable])
        node_b.append(_axis_slice(base, axis, slice(1, None))[usable])
        wrap_index.append(turns[usable].astype(np.int64))
    if node_a:
        node_a = np.concatenate(node_a)
        node_b = np.concatenate(node_b)
        wrap_index = np.concatenate(wrap_index)
    else:
        node_a = np.zeros(0, dtype=np.int64)
        node_b = np.zeros(0, dtype=np.int64)
        wrap_index = np.zeros(0, dtype=np.int64)
    rating = pixel_reliability(phase, weight, mask)
    stack = rating.ravel()
    priority = np.minimum(stack[node_a], stack[node_b])
    order = np.argsort(-priority, kind="stable")
    node_a, node_b = node_a[order], node_b[order]
    wrap_index = wrap_index[order]
    parent = np.arange(phase.size, dtype=np.int64)
    shift = np.zeros(phase.size, dtype=np.int64)
    size = np.ones(phase.size, dtype=np.int64)
    if backend == "numba":
        from .numba_backend import reliability_merge
        merges, discarded = reliability_merge(node_a, node_b, wrap_index, size,
                                              parent, shift)
    else:
        merges, discarded = _merge_python(node_a, node_b, wrap_index, size,
                                          parent, shift)
    turns = _accumulated_turns(parent, shift)
    result = phase + 2.0 * np.pi * turns.reshape(phase.shape)
    result[confidence <= 0.0] = np.nan
    pixels = int(np.count_nonzero(confidence > 0.0))
    if not return_info:
        return result
    info = ReliabilityInfo(backend, pixels, int(node_a.size), int(merges),
                           int(discarded), pixels - int(merges))
    return result, info
