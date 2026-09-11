"""Flynn minimum-discontinuity phase unwrapping."""

import numpy as np


LEFT, RIGHT, UP, DOWN = 0x08, 0x04, 0x02, 0x01
THIS_TIME, NEXT_TIME = 0x40, 0x80
CLEAR_LEFT, CLEAR_RIGHT = np.uint8(0xFF ^ LEFT), np.uint8(0xFF ^ RIGHT)
CLEAR_UP, CLEAR_DOWN = np.uint8(0xFF ^ UP), np.uint8(0xFF ^ DOWN)
CLEAR_THIS_TIME = np.uint8(0xFF ^ THIS_TIME)
CLEAR_NEXT_TIME = np.uint8(0xFF ^ NEXT_TIME)
BIG = 25500


def _nint(value):
    return int(value + 0.5) if value > 0 else int(value - 0.5)


def _isign(magnitude, sign):
    return magnitude if sign >= 0 else -magnitude


def _children(i, j, flags, cols, rows):
    bits = flags[j, i]
    if bits & LEFT:
        yield i, j - 1
    if bits & RIGHT:
        yield i, j + 1
    if bits & UP:
        yield i - 1, j
    if bits & DOWN:
        yield i + 1, j


def _change_extension(i, j, last_i, last_j, increment, values, flags, cols, rows):
    stack = [(i, j)]
    nodes = []
    loop_found = False
    while stack:
        node_i, node_j = stack.pop()
        loop_found = loop_found or (node_i == last_i and node_j == last_j)
        nodes.append((node_i, node_j))
        stack.extend(_children(node_i, node_j, flags, cols, rows))
    for node_i, node_j in reversed(nodes):
        values[node_j, node_i] += increment
        flags[node_j, node_i] |= NEXT_TIME | THIS_TIME
    return loop_found


def _change_orphan(i, j, shift, values, flags, cols, rows):
    stack = [(i, j, shift)]
    while stack:
        node_i, node_j, node_shift = stack.pop()
        if values[node_j, node_i] + node_shift < 0:
            node_shift = -values[node_j, node_i]
            if node_j > 0:
                flags[node_j - 1, node_i] &= CLEAR_RIGHT
            if node_j < rows:
                flags[node_j + 1, node_i] &= CLEAR_LEFT
            if node_i > 0:
                flags[node_j, node_i - 1] &= CLEAR_DOWN
            if node_i < cols:
                flags[node_j, node_i + 1] &= CLEAR_UP
        children = list(_children(node_i, node_j, flags, cols, rows))
        for child_i, child_j in reversed(children):
            stack.append((child_i, child_j, node_shift))
        flags[node_j, node_i] |= NEXT_TIME | THIS_TIME
        values[node_j, node_i] += node_shift


def _remove_loop(base_i, base_j, last_i, last_j, values, flags,
                 vertical_jump, horizontal_jump, cols, rows):
    if base_j > last_j:
        vertical_jump[base_j, base_i] -= 1
    elif base_j < last_j:
        vertical_jump[last_j, last_i] += 1
    elif base_i > last_i:
        horizontal_jump[base_j, base_i] += 1
    elif base_i < last_i:
        horizontal_jump[last_j, last_i] -= 1
    tip_i, tip_j = last_i, last_j
    while True:
        _change_orphan(tip_i, tip_j, -values[tip_j, tip_i],
                       values, flags, cols, rows)
        if tip_j > 0 and flags[tip_j - 1, tip_i] & RIGHT:
            vertical_jump[tip_j, tip_i] -= 1
            flags[tip_j - 1, tip_i] &= CLEAR_RIGHT
            tip_j -= 1
        elif tip_j < rows and flags[tip_j + 1, tip_i] & LEFT:
            vertical_jump[tip_j + 1, tip_i] += 1
            flags[tip_j + 1, tip_i] &= CLEAR_LEFT
            tip_j += 1
        elif tip_i > 0 and flags[tip_j, tip_i - 1] & DOWN:
            horizontal_jump[tip_j, tip_i] += 1
            flags[tip_j, tip_i - 1] &= CLEAR_DOWN
            tip_i -= 1
        elif tip_i < cols and flags[tip_j, tip_i + 1] & UP:
            horizontal_jump[tip_j, tip_i + 1] -= 1
            flags[tip_j, tip_i + 1] &= CLEAR_UP
            tip_i += 1
        else:
            raise RuntimeError("broken Flynn loop at ({}, {})".format(tip_i, tip_j))
        if tip_i == base_i and tip_j == base_j:
            break


def flynn_unwrap(phase, quality=None, mask=None, *, return_iterations=False):
    """Unwrap by Flynn's minimum-discontinuity network algorithm.

    ``quality`` is scaled to ``[0, 1]`` and weights discontinuity changes.
    False mask pixels receive zero quality. The implementation intentionally
    retains integer network costs and single-precision phase-cycle arithmetic.
    """
    phase = np.asarray(phase, dtype=np.float64)
    if phase.ndim != 2 or min(phase.shape) < 2 or not np.isfinite(phase).all():
        raise ValueError("phase must be a finite 2-D array with dimensions >= 2")
    rows, cols = phase.shape
    cycles = (((phase + np.pi) / (2 * np.pi)) % 1.0).astype(np.float32)
    if quality is None:
        quality_map = np.ones(phase.shape, dtype=np.float32)
    else:
        quality_map = np.asarray(quality, dtype=np.float32)
        if quality_map.shape != phase.shape or not np.isfinite(quality_map).all():
            raise ValueError("quality must be finite and match phase.shape")
        low, high = float(quality_map.min()), float(quality_map.max())
        quality_map = (quality_map - low) / (high - low) if high != low else np.ones_like(quality_map)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != phase.shape:
            raise ValueError("mask must match phase.shape")
        quality_map = quality_map.copy()
        quality_map[~mask] = 0

    values = np.zeros((rows + 1, cols + 1), dtype=np.int64)
    flags = np.full((rows + 1, cols + 1), THIS_TIME, dtype=np.uint8)
    vertical_jump = np.zeros((rows + 1, cols + 1), dtype=np.int16)
    horizontal_jump = np.zeros((rows + 1, cols + 1), dtype=np.int16)
    epsilon = 1e-6
    for j in range(1, rows):
        for i in range(1, cols + 1):
            derivative = float(np.float32(cycles[j, i - 1] - cycles[j - 1, i - 1])) + epsilon
            horizontal_jump[j, i] = _nint(derivative)
    for j in range(1, rows + 1):
        for i in range(1, cols):
            derivative = float(np.float32(cycles[j - 1, i] - cycles[j - 1, i - 1])) + epsilon
            vertical_jump[j, i] = _nint(derivative)

    iterations = 0
    while True:
        iterations += 1
        new_edges = 0
        # Left-to-right network edges.
        for j in range(0, rows):
            for i in range(0, cols + 1):
                next_j, next_i = j + 1, i
                if not (flags[j, i] & THIS_TIME or flags[next_j, next_i] & THIS_TIME):
                    continue
                if next_j <= 1 or next_j >= rows or next_i <= 0 or next_i >= cols:
                    increment = -_isign(1, -vertical_jump[next_j, next_i])
                else:
                    cost = int(1.0 + BIG * quality_map[next_j, next_i])
                    increment = -_isign(cost, -vertical_jump[next_j, next_i])
                change = values[j, i] + increment - values[next_j, next_i]
                if change > 0:
                    new_edges += 1
                    if _change_extension(next_i, next_j, i, j, change,
                                         values, flags, cols, rows):
                        _remove_loop(next_i, next_j, i, j, values, flags,
                                     vertical_jump, horizontal_jump, cols, rows)
                        _change_orphan(next_i, next_j, -change, values, flags, cols, rows)
                    else:
                        flags[j, i] |= RIGHT
                        if next_j < rows:
                            flags[next_j + 1, next_i] &= CLEAR_LEFT
                        if next_i < cols:
                            flags[next_j, next_i + 1] &= CLEAR_UP
                        if next_i > 0:
                            flags[next_j, next_i - 1] &= CLEAR_DOWN
        # Top-to-bottom network edges.
        for j in range(0, rows + 1):
            for i in range(0, cols):
                next_j, next_i = j, i + 1
                if not (flags[j, i] & THIS_TIME or flags[next_j, next_i] & THIS_TIME):
                    continue
                if next_j <= 0 or next_j >= rows or next_i <= 1 or next_i >= cols:
                    increment = -_isign(1, horizontal_jump[next_j, next_i])
                else:
                    cost = int(1.0 + BIG * quality_map[next_j, next_i])
                    increment = -_isign(cost, horizontal_jump[next_j, next_i])
                change = values[j, i] + increment - values[next_j, next_i]
                if change > 0:
                    new_edges += 1
                    if _change_extension(next_i, next_j, i, j, change,
                                         values, flags, cols, rows):
                        _remove_loop(next_i, next_j, i, j, values, flags,
                                     vertical_jump, horizontal_jump, cols, rows)
                        _change_orphan(next_i, next_j, -change, values, flags, cols, rows)
                    else:
                        flags[j, i] |= DOWN
                        if next_j < rows:
                            flags[next_j + 1, next_i] &= CLEAR_LEFT
                        if next_j > 0:
                            flags[next_j - 1, next_i] &= CLEAR_RIGHT
                        if next_i < cols:
                            flags[next_j, next_i + 1] &= CLEAR_UP
        # Right-to-left network edges.
        for j in range(rows, 0, -1):
            for i in range(cols, -1, -1):
                next_j, next_i = j - 1, i
                if not (flags[j, i] & THIS_TIME or flags[next_j, next_i] & THIS_TIME):
                    continue
                if j <= 1 or j >= rows or i <= 0 or i >= cols:
                    increment = -_isign(1, vertical_jump[j, i])
                else:
                    cost = int(1.0 + BIG * quality_map[j, i])
                    increment = -_isign(cost, vertical_jump[j, i])
                change = values[j, i] + increment - values[next_j, next_i]
                if change > 0:
                    new_edges += 1
                    if _change_extension(next_i, next_j, i, j, change,
                                         values, flags, cols, rows):
                        _remove_loop(next_i, next_j, i, j, values, flags,
                                     vertical_jump, horizontal_jump, cols, rows)
                        _change_orphan(next_i, next_j, -change, values, flags, cols, rows)
                    else:
                        flags[j, i] |= LEFT
                        if next_j > 0:
                            flags[next_j - 1, next_i] &= CLEAR_RIGHT
                        if next_i < cols:
                            flags[next_j, next_i + 1] &= CLEAR_UP
                        if next_i > 0:
                            flags[next_j, next_i - 1] &= CLEAR_DOWN
        # Bottom-to-top network edges.
        for j in range(rows, -1, -1):
            for i in range(cols, 0, -1):
                next_j, next_i = j, i - 1
                if not (flags[j, i] & THIS_TIME or flags[next_j, next_i] & THIS_TIME):
                    continue
                if j <= 0 or j >= rows or i <= 1 or i >= cols:
                    increment = -_isign(1, -horizontal_jump[j, i])
                else:
                    cost = int(1.0 + BIG * quality_map[j, i])
                    increment = -_isign(cost, -horizontal_jump[j, i])
                change = values[j, i] + increment - values[next_j, next_i]
                if change > 0:
                    new_edges += 1
                    if _change_extension(next_i, next_j, i, j, change,
                                         values, flags, cols, rows):
                        _remove_loop(next_i, next_j, i, j, values, flags,
                                     vertical_jump, horizontal_jump, cols, rows)
                        _change_orphan(next_i, next_j, -change, values, flags, cols, rows)
                    else:
                        flags[j, i] |= UP
                        if next_j < rows:
                            flags[next_j + 1, next_i] &= CLEAR_LEFT
                        if next_j > 0:
                            flags[next_j - 1, next_i] &= CLEAR_RIGHT
                        if next_i > 0:
                            flags[next_j, next_i - 1] &= CLEAR_DOWN
        for j in range(rows + 1):
            for i in range(cols + 1):
                if flags[j, i] & NEXT_TIME:
                    flags[j, i] |= THIS_TIME
                else:
                    flags[j, i] &= CLEAR_NEXT_TIME
                    flags[j, i] &= CLEAR_THIS_TIME
        if new_edges == 0:
            break

    for i in range(cols - 1):
        derivative = float(np.float32(cycles[0, i + 1] - cycles[0, i])) + epsilon
        old_jump = _nint(derivative)
        cycles[0, i + 1] = np.float32(
            cycles[0, i + 1] + vertical_jump[1, i + 1] - old_jump)
    for j in range(rows - 1):
        for i in range(cols):
            derivative = float(np.float32(cycles[j + 1, i] - cycles[j, i])) + epsilon
            old_jump = _nint(derivative)
            cycles[j + 1, i] = np.float32(
                cycles[j + 1, i] + horizontal_jump[j + 1, i + 1] - old_jump)
    output = cycles.astype(np.float64) * (2 * np.pi)
    return (output, iterations) if return_iterations else output
