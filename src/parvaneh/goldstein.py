"""Goldstein branch-cut phase unwrapping."""

import bisect
import numpy as np
from scipy.ndimage import maximum_filter


def _gradient_cycles(first, second):
    value = np.float32(first - second)
    if value > 0.5:
        value = np.float32(value - 1.0)
    elif value < -0.5:
        value = np.float32(value + 1.0)
    return value


def _residue_charges_cycles(phase, border):
    rows, cols = phase.shape
    charges = np.zeros((rows, cols), dtype=np.int8)
    for row in range(rows - 1):
        for col in range(cols - 1):
            if (border[row, col] or border[row, col + 1]
                    or border[row + 1, col + 1] or border[row + 1, col]):
                continue
            circulation = (
                _gradient_cycles(phase[row, col + 1], phase[row, col])
                + _gradient_cycles(phase[row + 1, col + 1], phase[row, col + 1])
                + _gradient_cycles(phase[row + 1, col], phase[row + 1, col + 1])
                + _gradient_cycles(phase[row, col], phase[row + 1, col]))
            if circulation > 0.01:
                charges[row, col] = 1
            elif circulation < -0.01:
                charges[row, col] = -1
    return charges


def _nearest_border(border, col, row):
    rows, cols = border.shape
    for box in range(rows + cols):
        found = None
        best_distance = 1_000_000
        for candidate_row in range(row - box, row + box + 1):
            for candidate_col in range(col - box, col + box + 1):
                is_border = (candidate_col <= 0 or candidate_col >= cols - 1
                             or candidate_row <= 0 or candidate_row >= rows - 1)
                if not is_border and border[candidate_row, candidate_col]:
                    is_border = True
                if is_border:
                    distance = ((candidate_row - row) ** 2
                                + (candidate_col - col) ** 2)
                    if distance < best_distance:
                        best_distance = distance
                        found = (candidate_col, candidate_row)
        if found is not None:
            return best_distance, found
    raise RuntimeError("no border found")


def _place_cut(cuts, start_col, start_row, end_col, end_row):
    rows, cols = cuts.shape
    a, b, c, d = start_col, start_row, end_col, end_row
    if c > a and a > 0:
        a += 1
    elif c < a and c > 0:
        c += 1
    if d > b and b > 0:
        b += 1
    elif d < b and d > 0:
        d += 1
    if a == c and b == d:
        if 0 <= b < rows and 0 <= a < cols:
            cuts[b, a] = True
        return
    horizontal = abs(c - a)
    vertical = abs(d - b)
    if horizontal > vertical:
        step = 1 if a < c else -1
        slope = (d - b) / (c - a)
        for col in range(a, c + step, step):
            row = int(b + (col - a) * slope + 0.5)
            if 0 <= row < rows and 0 <= col < cols:
                cuts[row, col] = True
    else:
        step = 1 if b < d else -1
        slope = (c - a) / (d - b)
        for row in range(b, d + step, step):
            col = int(a + (row - b) * slope + 0.5)
            if 0 <= row < rows and 0 <= col < cols:
                cuts[row, col] = True


def _branch_cuts(charges, border, max_cut_length):
    rows, cols = charges.shape
    cuts = np.zeros(charges.shape, dtype=bool)
    visited = np.zeros(charges.shape, dtype=bool)
    active = np.zeros(charges.shape, dtype=bool)
    max_cut_length = max(2, int(max_cut_length))
    for row in range(rows):
        for col in range(cols):
            if not charges[row, col] or visited[row, col]:
                continue
            visited[row, col] = True
            active[row, col] = True
            charge = int(charges[row, col])
            active_list = [(col, row)]
            balanced = False
            for box_size in range(3, 2 * max_cut_length + 1, 2):
                half = box_size // 2
                for center_col, center_row in list(active_list):
                    for candidate_row in range(center_row - half, center_row + half + 1):
                        for candidate_col in range(center_col - half, center_col + half + 1):
                            if not (0 <= candidate_col < cols and 0 <= candidate_row < rows):
                                continue
                            if (candidate_col == 0 or candidate_col == cols - 1
                                    or candidate_row == 0 or candidate_row == rows - 1
                                    or border[candidate_row, candidate_col]):
                                charge = 0
                                _, (rim_col, rim_row) = _nearest_border(
                                    border, center_col, center_row)
                                _place_cut(cuts, rim_col, rim_row,
                                           center_col, center_row)
                            elif (charges[candidate_row, candidate_col]
                                  and not active[candidate_row, candidate_col]):
                                if not visited[candidate_row, candidate_col]:
                                    charge += int(charges[candidate_row, candidate_col])
                                    visited[candidate_row, candidate_col] = True
                                active_list.append((candidate_col, candidate_row))
                                active[candidate_row, candidate_col] = True
                                _place_cut(cuts, candidate_col, candidate_row,
                                           center_col, center_row)
                            if charge == 0:
                                balanced = True
                                break
                        if balanced:
                            break
                    if balanced:
                        break
                if balanced:
                    break
            if charge != 0:
                best = None
                for active_col, active_row in active_list:
                    distance, (rim_col, rim_row) = _nearest_border(
                        border, active_col, active_row)
                    if best is None or distance < best[0]:
                        best = (distance, active_col, active_row, rim_col, rim_row)
                _, near_col, near_row, rim_col, rim_row = best
                _place_cut(cuts, near_col, near_row, rim_col, rim_row)
            for active_col, active_row in active_list:
                active[active_row, active_col] = False
    return cuts


def _unwrap_around_cuts(phase, cuts, blocked):
    """Integrate the wrapped field away from the cuts.

    ``blocked`` marks the pixels that carry no usable phase, so they are never
    entered and keep their initial value of zero.  Cut pixels are likewise left
    at zero.  Every other pixel, including a valid pixel that merely touches a
    masked one, is integrated normally.
    """
    rows, cols = phase.shape
    output = np.zeros(phase.shape, dtype=np.float32)
    accepted = np.zeros(phase.shape, dtype=bool)
    avoid = cuts | blocked
    pieces = 0
    for start in range(rows * cols):
        row, col = divmod(start, cols)
        if avoid[row, col] or accepted[row, col]:
            continue
        pieces += 1
        output[row, col] = phase[row, col]
        accepted[row, col] = True
        stack = []

        def propose(next_row, next_col, parent_row, parent_col):
            if (next_row < 0 or next_row >= rows or next_col < 0 or next_col >= cols
                    or avoid[next_row, next_col] or accepted[next_row, next_col]):
                return
            output[next_row, next_col] = np.float32(
                output[parent_row, parent_col]
                + _gradient_cycles(phase[next_row, next_col], phase[parent_row, parent_col]))
            accepted[next_row, next_col] = True
            stack.append((next_row, next_col))

        propose(row, col - 1, row, col)
        propose(row, col + 1, row, col)
        propose(row - 1, col, row, col)
        propose(row + 1, col, row, col)
        while stack:
            current_row, current_col = stack.pop()
            propose(current_row, current_col - 1, current_row, current_col)
            propose(current_row, current_col + 1, current_row, current_col)
            propose(current_row - 1, current_col, current_row, current_col)
            propose(current_row + 1, current_col, current_row, current_col)
    for row in range(1, rows):
        for col in range(1, cols):
            if cuts[row, col]:
                if accepted[row, col - 1]:
                    output[row, col] = np.float32(
                        output[row, col - 1]
                        + _gradient_cycles(phase[row, col], phase[row, col - 1]))
                    accepted[row, col] = True
                elif accepted[row - 1, col]:
                    output[row, col] = np.float32(
                        output[row - 1, col]
                        + _gradient_cycles(phase[row, col], phase[row - 1, col]))
                    accepted[row, col] = True
    return output, pieces


def goldstein_unwrap(phase, mask=None, *, max_cut_length=None,
                     return_cuts=False):
    """Unwrap phase with Goldstein's expanding-box branch-cut method.

    The output uses radians.  ``mask`` is true for valid pixels.  The optional
    cut map is a boolean pixel mask suitable for plotting and diagnostics.
    Pixels outside ``mask`` and pixels on a cut keep the value zero, because
    no phase can be assigned there; every other valid pixel is unwrapped,
    including the ones that only touch the mask.
    """
    phase = np.asarray(phase, dtype=np.float64)
    if phase.ndim != 2 or min(phase.shape) < 2 or not np.isfinite(phase).all():
        raise ValueError("phase must be a finite 2-D array with dimensions >= 2")
    valid = np.ones(phase.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if valid.shape != phase.shape:
        raise ValueError("mask must match phase.shape")
    invalid = ~valid
    # The guard is the invalid region grown by one pixel.  Residue counting and
    # branch-cut growth need it, because a quad that straddles the edge of the
    # data has no phase continuity and would report a spurious residue.  The
    # guard must not reach the integration step: its pixels are valid data and
    # have to be unwrapped like any other.
    border = invalid
    if border.any():
        border = maximum_filter(border.astype(np.uint8), size=3) != 0
    cycles = (((phase + np.pi) / (2 * np.pi)) % 1.0).astype(np.float32)
    charges = _residue_charges_cycles(cycles, border)
    if max_cut_length is None:
        max_cut_length = sum(phase.shape) // 2
    cuts = _branch_cuts(charges, border, max_cut_length)
    output, _ = _unwrap_around_cuts(cycles, cuts, invalid)
    output = output.astype(np.float64) * (2 * np.pi)
    return (output, cuts) if return_cuts else output


def _quality_guided_mask(phase, charges, border):
    rows, cols = phase.shape
    cuts = np.zeros(phase.shape, dtype=bool)
    for start_row in range(rows):
        for start_col in range(cols):
            if not charges[start_row, start_col] or cuts[start_row, start_col] or border[start_row, start_col]:
                continue
            charge = int(charges[start_row, start_col])
            cuts[start_row, start_col] = True
            visited = np.zeros(phase.shape, dtype=bool)
            frontier = []
            sequence = 0

            def propose(row, col, parent_row, parent_col):
                nonlocal sequence
                if not (0 <= row < rows and 0 <= col < cols) or visited[row, col]:
                    return
                delta = _gradient_cycles(phase[row, col], phase[parent_row, parent_col])
                priority = -abs(float(delta))
                position = bisect.bisect_left([item[0] for item in frontier], priority)
                frontier.insert(position, (priority, sequence, row, col))
                sequence += 1
                visited[row, col] = True

            propose(start_row, start_col - 1, start_row, start_col)
            propose(start_row, start_col + 1, start_row, start_col)
            propose(start_row - 1, start_col, start_row, start_col)
            propose(start_row + 1, start_col, start_row, start_col)
            while charge != 0 and frontier:
                _, _, row, col = frontier.pop()
                if charges[row, col] and not cuts[row, col]:
                    charge += int(charges[row, col])
                if (border[row, col] or col == 0 or col == cols - 1
                        or row == 0 or row == rows - 1):
                    charge = 0
                cuts[row, col] = True
                propose(row, col - 1, row, col)
                propose(row, col + 1, row, col)
                propose(row - 1, col, row, col)
                propose(row + 1, col, row, col)
    return cuts


def _thin_mask(cuts, charges, border):
    rows, cols = cuts.shape
    avoid = (charges != 0) | border
    removed = True
    while removed:
        removed = False
        for row in range(1, rows - 1):
            for col in range(1, cols - 1):
                if not cuts[row, col] or avoid[row - 1:row + 2, col - 1:col + 2].any():
                    continue
                ring = [cuts[row - 1, col], cuts[row - 1, col + 1],
                        cuts[row, col + 1], cuts[row + 1, col + 1],
                        cuts[row + 1, col], cuts[row + 1, col - 1],
                        cuts[row, col - 1], cuts[row - 1, col - 1]]
                transitions = sum(bool(ring[index]) != bool(ring[(index + 1) % 8])
                                  for index in range(8))
                # The reference counts entries into and out of connected arcs;
                # fewer than three flips means one local component.
                if transitions < 3 and (not cuts[row - 1, col]
                                         or not cuts[row, col + 1]
                                         or not cuts[row + 1, col]
                                         or not cuts[row, col - 1]):
                    cuts[row, col] = False
                    removed = True
    return cuts


def mask_cut_unwrap(phase, mask=None, *, return_cuts=False):
    """Quality-guided mask-cut unwrapping using minimum-gradient paths.

    ``mask`` is true for valid pixels; pixels outside it and pixels on a cut
    keep the value zero.
    """
    phase = np.asarray(phase, dtype=np.float64)
    if phase.ndim != 2 or min(phase.shape) < 2 or not np.isfinite(phase).all():
        raise ValueError("phase must be a finite 2-D array with dimensions >= 2")
    valid = np.ones(phase.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if valid.shape != phase.shape:
        raise ValueError("mask must match phase.shape")
    invalid = ~valid
    # Same split as in goldstein_unwrap: the guard fed to the cut stages is the
    # invalid region grown by one pixel, while integration only skips pixels
    # that really carry no phase.
    border = invalid
    if border.any():
        border = maximum_filter(border.astype(np.uint8), size=3) != 0
    cycles = (((phase + np.pi) / (2 * np.pi)) % 1.0).astype(np.float32)
    charges = _residue_charges_cycles(cycles, border)
    cuts = _thin_mask(_quality_guided_mask(cycles, charges, border), charges, border)
    output, _ = _unwrap_around_cuts(cycles, cuts, invalid)
    output = output.astype(np.float64) * (2 * np.pi)
    return (output, cuts) if return_cuts else output
