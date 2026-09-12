"""Flynn minimum-discontinuity phase unwrapping.

Flynn's method builds the unwrapped phase by repeatedly adding network edges and
flipping the direction of an edge whenever that reduces the total number of
discontinuities.  The inner loops below touch every node of the network several
times per outer iteration, so the network is kept in flat Python lists indexed
``j * width + i`` rather than in NumPy arrays: reading a single element out of a
NumPy array allocates and boxes a NumPy scalar every time, which costs far more
than the integer arithmetic it feeds.  The lists are converted back to NumPy
arrays only at the very end.  See ``docs/performance.md`` for the measurements
that motivated this.
"""

import numpy as np


LEFT, RIGHT, UP, DOWN = 0x08, 0x04, 0x02, 0x01
THIS_TIME, NEXT_TIME = 0x40, 0x80
CLEAR_LEFT = 0xFF ^ LEFT
CLEAR_RIGHT = 0xFF ^ RIGHT
CLEAR_UP = 0xFF ^ UP
CLEAR_DOWN = 0xFF ^ DOWN
CLEAR_NEXT_THIS = 0xFF ^ (NEXT_TIME | THIS_TIME)
NEXT_THIS = NEXT_TIME | THIS_TIME
BIG = 25500


def _nint(value):
    return int(value + 0.5) if value > 0 else int(value - 0.5)


def _change_extension(start, target, increment, values, flags, width):
    """Add ``increment`` to every node reachable from ``start``.

    Returns ``True`` when ``target`` is reachable from ``start``, which means the
    candidate edge would close a loop.  Reachable nodes are collected first and
    updated afterwards, so the traversal sees the network as it was before the
    update -- the same ordering as the reference formulation.  A node may be
    collected more than once and is then incremented once per collection.
    """
    stack = [start]
    nodes = []
    loop_found = False
    while stack:
        node = stack.pop()
        if node == target:
            loop_found = True
        nodes.append(node)
        bits = flags[node]
        if bits & LEFT:
            stack.append(node - width)
        if bits & RIGHT:
            stack.append(node + width)
        if bits & UP:
            stack.append(node - 1)
        if bits & DOWN:
            stack.append(node + 1)
    for node in reversed(nodes):
        values[node] += increment
        flags[node] |= NEXT_THIS
    return loop_found


def _change_orphan(node, shift, values, flags, width, cols, rows):
    """Shift one subtree by ``shift``, detaching children that would go negative.

    Children are pushed so that they are popped in ``LEFT, RIGHT, UP, DOWN``
    order, because the traversal mutates the network as it goes and the result
    depends on that order.
    """
    stack = [(node, shift)]
    while stack:
        node, node_shift = stack.pop()
        if values[node] + node_shift < 0:
            node_shift = -values[node]
            node_i = node % width
            node_j = node // width
            if node_j > 0:
                flags[node - width] &= CLEAR_RIGHT
            if node_j < rows:
                flags[node + width] &= CLEAR_LEFT
            if node_i > 0:
                flags[node - 1] &= CLEAR_DOWN
            if node_i < cols:
                flags[node + 1] &= CLEAR_UP
        bits = flags[node]
        if bits & DOWN:
            stack.append((node + 1, node_shift))
        if bits & UP:
            stack.append((node - 1, node_shift))
        if bits & RIGHT:
            stack.append((node + width, node_shift))
        if bits & LEFT:
            stack.append((node - width, node_shift))
        flags[node] |= NEXT_THIS
        values[node] += node_shift


def _remove_loop(base, last, values, flags, vertical_jump, horizontal_jump,
                 width, cols, rows):
    """Flip the loop that the new edge closed and update the jump counts."""
    base_i, base_j = base % width, base // width
    last_i, last_j = last % width, last // width
    if base_j > last_j:
        vertical_jump[base] -= 1
    elif base_j < last_j:
        vertical_jump[last] += 1
    elif base_i > last_i:
        horizontal_jump[base] += 1
    elif base_i < last_i:
        horizontal_jump[last] -= 1
    tip = last
    while True:
        _change_orphan(tip, -values[tip], values, flags, width, cols, rows)
        tip_i, tip_j = tip % width, tip // width
        if tip_j > 0 and flags[tip - width] & RIGHT:
            vertical_jump[tip] -= 1
            flags[tip - width] &= CLEAR_RIGHT
            tip -= width
        elif tip_j < rows and flags[tip + width] & LEFT:
            vertical_jump[tip + width] += 1
            flags[tip + width] &= CLEAR_LEFT
            tip += width
        elif tip_i > 0 and flags[tip - 1] & DOWN:
            horizontal_jump[tip] += 1
            flags[tip - 1] &= CLEAR_DOWN
            tip -= 1
        elif tip_i < cols and flags[tip + 1] & UP:
            horizontal_jump[tip + 1] -= 1
            flags[tip + 1] &= CLEAR_UP
            tip += 1
        else:
            raise RuntimeError("broken Flynn loop at ({}, {})".format(tip_i, tip_j))
        if tip == base:
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

    width = cols + 1
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

    # An edge cost only depends on the quality map, which never changes, so it
    # is evaluated once.  It is laid out on the same padded (rows + 1) x
    # (cols + 1) grid as the jump counts so that all four sweeps index it the
    # same way; the padding is never read.  Evaluating
    # ``int(1.0 + BIG * np.float32(q))`` per element reproduces the original
    # scalar expression exactly, whereas the vectorised form would round
    # differently because it keeps the product in single precision.
    padded_quality = np.zeros((rows + 1, cols + 1), dtype=np.float32)
    padded_quality[:rows, :cols] = quality_map
    cost = [int(1.0 + BIG * np.float32(q)) for q in padded_quality.ravel().tolist()]
    vertical_jump = vertical_jump.ravel().tolist()
    horizontal_jump = horizontal_jump.ravel().tolist()
    values = [0] * ((rows + 1) * width)
    flags = [THIS_TIME] * ((rows + 1) * width)

    iterations = 0
    while True:
        iterations += 1
        new_edges = 0
        # Left-to-right network edges.
        for j in range(rows):
            base = j * width
            nbase = base + width
            edge_row = j == 0 or j + 1 >= rows
            for i in range(cols + 1):
                b = base + i
                n = nbase + i
                if not ((flags[b] | flags[n]) & THIS_TIME):
                    continue
                jump = vertical_jump[n]
                if edge_row or i == 0 or i == cols:
                    increment = 1 if jump > 0 else -1
                else:
                    c = cost[n]
                    increment = c if jump > 0 else -c
                change = values[b] + increment - values[n]
                if change > 0:
                    new_edges += 1
                    if _change_extension(n, b, change, values, flags, width):
                        _remove_loop(n, b, values, flags,
                                     vertical_jump, horizontal_jump, width, cols, rows)
                        _change_orphan(n, -change, values, flags, width, cols, rows)
                    else:
                        flags[b] |= RIGHT
                        if j + 1 < rows:
                            flags[n + width] &= CLEAR_LEFT
                        if i < cols:
                            flags[n + 1] &= CLEAR_UP
                        if i > 0:
                            flags[n - 1] &= CLEAR_DOWN
        # Top-to-bottom network edges.
        for j in range(rows + 1):
            base = j * width
            edge_row = j == 0 or j >= rows
            for i in range(cols):
                b = base + i
                n = b + 1
                if not ((flags[b] | flags[n]) & THIS_TIME):
                    continue
                jump = horizontal_jump[n]
                if edge_row or i == 0 or i >= cols - 1:
                    increment = 1 if jump < 0 else -1
                else:
                    c = cost[n]
                    increment = c if jump < 0 else -c
                change = values[b] + increment - values[n]
                if change > 0:
                    new_edges += 1
                    if _change_extension(n, b, change, values, flags, width):
                        _remove_loop(n, b, values, flags,
                                     vertical_jump, horizontal_jump, width, cols, rows)
                        _change_orphan(n, -change, values, flags, width, cols, rows)
                    else:
                        flags[b] |= DOWN
                        if j < rows:
                            flags[n + width] &= CLEAR_LEFT
                        if j > 0:
                            flags[n - width] &= CLEAR_RIGHT
                        if i < cols - 1:
                            flags[n + 1] &= CLEAR_UP
        # Right-to-left network edges.
        for j in range(rows, 0, -1):
            base = j * width
            nbase = base - width
            edge_row = j <= 1 or j >= rows
            for i in range(cols, -1, -1):
                b = base + i
                n = nbase + i
                if not ((flags[b] | flags[n]) & THIS_TIME):
                    continue
                jump = vertical_jump[b]
                if edge_row or i == 0 or i == cols:
                    increment = 1 if jump < 0 else -1
                else:
                    c = cost[b]
                    increment = c if jump < 0 else -c
                change = values[b] + increment - values[n]
                if change > 0:
                    new_edges += 1
                    if _change_extension(n, b, change, values, flags, width):
                        _remove_loop(n, b, values, flags,
                                     vertical_jump, horizontal_jump, width, cols, rows)
                        _change_orphan(n, -change, values, flags, width, cols, rows)
                    else:
                        flags[b] |= LEFT
                        if j > 1:
                            flags[b - width - width] &= CLEAR_RIGHT
                        if i < cols:
                            flags[n + 1] &= CLEAR_UP
                        if i > 0:
                            flags[n - 1] &= CLEAR_DOWN
        # Bottom-to-top network edges.
        for j in range(rows, -1, -1):
            base = j * width
            edge_row = j == 0 or j >= rows
            for i in range(cols, 0, -1):
                b = base + i
                n = b - 1
                if not ((flags[b] | flags[n]) & THIS_TIME):
                    continue
                jump = horizontal_jump[b]
                if edge_row or i <= 1 or i == cols:
                    increment = 1 if jump > 0 else -1
                else:
                    c = cost[b]
                    increment = c if jump > 0 else -c
                change = values[b] + increment - values[n]
                if change > 0:
                    new_edges += 1
                    if _change_extension(n, b, change, values, flags, width):
                        _remove_loop(n, b, values, flags,
                                     vertical_jump, horizontal_jump, width, cols, rows)
                        _change_orphan(n, -change, values, flags, width, cols, rows)
                    else:
                        flags[b] |= UP
                        if j < rows:
                            flags[n + width] &= CLEAR_LEFT
                        if j > 0:
                            flags[n - width] &= CLEAR_RIGHT
                        if i > 1:
                            flags[n - 1] &= CLEAR_DOWN
        flags = [(f | THIS_TIME) if f & NEXT_TIME else (f & CLEAR_NEXT_THIS)
                 for f in flags]
        if new_edges == 0:
            break

    # Reconstruction is not performance-critical, so the jump counts are turned
    # back into int16 arrays here.  That is not cosmetic: adding an int16 to the
    # float32 cycle grid rounds the sum to single precision, while adding a
    # Python int would widen it to double precision and flip the last bit of the
    # result.
    vertical_jump = np.array(vertical_jump, dtype=np.int16).reshape(rows + 1, cols + 1)
    horizontal_jump = np.array(horizontal_jump, dtype=np.int16).reshape(rows + 1, cols + 1)
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
