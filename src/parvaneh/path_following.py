"""Path-following phase-unwrapping algorithms."""

import bisect
import numpy as np

from .quality import max_gradient_quality, wrapped_gradients


def phase_residues(phase, mask=None):
    """Return residue charges on the upper-left corners of 2x2 cells.

    Charges are normally -1, 0, or +1.  Cells touching a false mask pixel are
    assigned zero, matching the usual definition for undefined observations.
    """
    phase = np.asarray(phase, dtype=np.float64)
    if phase.ndim != 2 or min(phase.shape) < 2:
        raise ValueError("phase must be a 2-D array with dimensions >= 2")
    dx, dy = wrapped_gradients(phase)
    circulation = dx[:-1, :] + dy[:, 1:] - dx[1:, :] - dy[:, :-1]
    charge = np.rint(circulation / (2 * np.pi)).astype(np.int8)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != phase.shape:
            raise ValueError("mask must match phase.shape")
        valid = mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, 1:] & mask[1:, :-1]
        charge = np.where(valid, charge, 0).astype(np.int8)
    return charge


def quality_guided_unwrap(phase, quality=None, mask=None, *, return_order=False,
                          max_frontier=None, backend="python"):
    """Unwrap connected regions by a best-first, quality-guided traversal.

    Each queued candidate stores the already-unwrapped parent that proposed it.
    The highest-quality candidate is accepted first.  Disconnected valid
    regions are restarted independently and therefore each has its own offset.
    """
    phase = np.asarray(phase, dtype=np.float64)
    if phase.ndim != 2 or min(phase.shape) < 2 or not np.isfinite(phase).all():
        raise ValueError("phase must be a finite 2-D array with dimensions >= 2")
    edge_priority = isinstance(quality, str) and quality == "min_gradient"
    if isinstance(quality, str) and not edge_priority:
        raise ValueError("the only string quality mode is 'min_gradient'")
    if not edge_priority:
        quality = max_gradient_quality(phase) if quality is None else np.asarray(quality, dtype=np.float64)
        if quality.shape != phase.shape or not np.isfinite(quality).all():
            raise ValueError("quality must be finite and match phase.shape")
    valid = np.ones(phase.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if valid.shape != phase.shape:
        raise ValueError("mask must match phase.shape")

    # The edge-priority formulation was originally specified in phase cycles
    # and single precision.  Keeping that arithmetic is important because the
    # traversal can legitimately change when quantized gradients tie.
    work_phase = (((phase + np.pi) / (2 * np.pi)) % 1.0).astype(np.float32) if edge_priority else phase
    half_period = 0.5 if edge_priority else np.pi
    period = 1.0 if edge_priority else 2 * np.pi

    rows, cols = phase.shape
    result = np.full(phase.shape, np.nan, dtype=work_phase.dtype)
    # 0 unseen, 1 queued, 2 accepted, 3 temporarily postponed.
    state = np.zeros(phase.shape, dtype=np.uint8)
    order = np.full(phase.shape, -1, dtype=np.int64)
    saved_priority = np.zeros(phase.shape, dtype=np.float64)
    sequence = 0
    max_frontier = rows + cols if max_frontier is None else int(max_frontier)
    if max_frontier < 2:
        raise ValueError("max_frontier must be at least two")
    if backend == "numba":
        if not edge_priority:
            raise ValueError("the numba path backend currently requires quality='min_gradient'")
        from .numba_backend import quality_guided_min_gradient
        result, order = quality_guided_min_gradient(
            np.ascontiguousarray(work_phase), np.ascontiguousarray(valid), max_frontier)
        result = result.astype(np.float64) * (2 * np.pi)
        return (result, order) if return_order else result
    if backend != "python":
        raise ValueError("backend must be 'python' or 'numba'")

    def insert(row, col, candidate, priority, frontier, minimum_priority):
        nonlocal sequence
        result[row, col] = candidate
        saved_priority[row, col] = priority
        if priority < minimum_priority:
            state[row, col] = 3
            return minimum_priority
        # New ties go before old ties; popping the rightmost entry is FIFO.
        position = bisect.bisect_left([item[0] for item in frontier], priority)
        frontier.insert(position, (priority, sequence, row, col))
        state[row, col] = 1
        sequence += 1
        if len(frontier) >= max_frontier:
            discard = len(frontier) // 2
            for _, _, postponed_row, postponed_col in frontier[:discard]:
                state[postponed_row, postponed_col] = 3
            del frontier[:discard]
            minimum_priority = frontier[0][0]
        return minimum_priority

    def propose(row, col, parent_row, parent_col, frontier, minimum_priority):
        if not valid[row, col] or state[row, col] in (1, 2):
            return minimum_priority
        # Preserve +pi versus -pi at the branch boundary.  The historical
        # algorithm uses strict comparisons rather than modulo arithmetic;
        # byte rasters can contain an exact half-cycle difference.
        delta = work_phase[row, col] - work_phase[parent_row, parent_col]
        if delta > half_period:
            delta -= period
        elif delta < -half_period:
            delta += period
        candidate = result[parent_row, parent_col] + delta
        priority = -abs(delta) if edge_priority else quality[row, col]
        return insert(row, col, candidate, priority, frontier, minimum_priority)

    for flat_start in np.flatnonzero(valid.ravel()):
        sr, sc = divmod(int(flat_start), cols)
        if state[sr, sc]:
            continue
        result[sr, sc] = work_phase[sr, sc]
        state[sr, sc] = 2
        order[sr, sc] = sequence
        sequence += 1
        frontier = []
        minimum_priority = -1e10
        for nr, nc in ((sr, sc - 1), (sr, sc + 1), (sr - 1, sc), (sr + 1, sc)):
            if 0 <= nr < rows and 0 <= nc < cols:
                minimum_priority = propose(nr, nc, sr, sc, frontier, minimum_priority)
        while frontier:
            _, _, row, col = frontier.pop()
            state[row, col] = 2
            order[row, col] = sequence
            sequence += 1
            for nr, nc in ((row, col - 1), (row, col + 1), (row - 1, col), (row + 1, col)):
                if 0 <= nr < rows and 0 <= nc < cols:
                    minimum_priority = propose(nr, nc, row, col, frontier, minimum_priority)
            if not frontier:
                minimum_priority = -1e10
                for postponed_flat in np.flatnonzero(state.ravel() == 3):
                    pr, pc = divmod(int(postponed_flat), cols)
                    minimum_priority = insert(pr, pc, result[pr, pc],
                                              saved_priority[pr, pc], frontier,
                                              minimum_priority)
    if edge_priority:
        result = result.astype(np.float64) * (2 * np.pi)
    return (result, order) if return_order else result
