import numpy as np
from numba import njit


@njit(cache=True)
def apply_q(p, wx, wy):
    rows, cols = p.shape
    out = np.zeros_like(p)
    for i in range(rows):
        for j in range(cols):
            value = 0.0
            if j < cols - 1:
                value += wx[i, j] * (p[i, j + 1] - p[i, j])
            if j > 0:
                value -= wx[i, j - 1] * (p[i, j] - p[i, j - 1])
            if i < rows - 1:
                value += wy[i, j] * (p[i + 1, j] - p[i, j])
            if i > 0:
                value -= wy[i - 1, j] * (p[i, j] - p[i - 1, j])
            out[i, j] = value
    return out


@njit(cache=True)
def reliability_merge(node_a, node_b, wrap_index, size, parent, shift):
    """Merge the edges of a reliability-sorted list into spanning trees.

    ``node_a``, ``node_b`` and ``wrap_index`` are the edges in visiting order.
    ``parent``, ``shift`` and ``size`` are the mutable union-find state: for
    every node ``x``, ``shift[x]`` counts how many whole turns must be added
    when stepping from ``x`` to ``parent[x]``.  Integer turns keep the merge
    exact, so this kernel and the Python fallback agree bit for bit.
    """
    merges = 0
    discarded = 0
    for index in range(node_a.size):
        a = node_a[index]
        b = node_b[index]
        turns_a = 0
        root_a = a
        while parent[root_a] != root_a:
            turns_a += shift[root_a]
            root_a = parent[root_a]
        turns_b = 0
        root_b = b
        while parent[root_b] != root_b:
            turns_b += shift[root_b]
            root_b = parent[root_b]
        if root_a == root_b:
            discarded += 1
            continue
        relation = turns_a - turns_b - wrap_index[index]
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


@njit(cache=True)
def _frontier_insert(result, saved_priority, state, priorities, indices,
                     count, max_frontier, index, candidate, priority,
                     minimum_priority):
    result[index] = candidate
    saved_priority[index] = priority
    if priority < minimum_priority:
        state[index] = 3
        return count, minimum_priority
    position = 0
    while position < count and priorities[position] < priority:
        position += 1
    for cursor in range(count, position, -1):
        priorities[cursor] = priorities[cursor - 1]
        indices[cursor] = indices[cursor - 1]
    priorities[position] = priority
    indices[position] = index
    count += 1
    state[index] = 1
    if count >= max_frontier:
        discard = count // 2
        for cursor in range(discard):
            state[indices[cursor]] = 3
        for cursor in range(count - discard):
            priorities[cursor] = priorities[cursor + discard]
            indices[cursor] = indices[cursor + discard]
        count -= discard
        minimum_priority = priorities[0]
    return count, minimum_priority


@njit(cache=True)
def _propose_min_gradient(phase, valid, result, saved_priority, state,
                          priorities, indices, count, max_frontier,
                          index, parent, minimum_priority):
    if index < 0 or index >= phase.size or not valid[index]:
        return count, minimum_priority
    if state[index] == 1 or state[index] == 2:
        return count, minimum_priority
    delta = np.float32(phase[index] - phase[parent])
    if delta > 0.5:
        delta = np.float32(delta - 1.0)
    elif delta < -0.5:
        delta = np.float32(delta + 1.0)
    candidate = np.float32(result[parent] + delta)
    priority = np.float32(-abs(delta))
    return _frontier_insert(result, saved_priority, state, priorities,
                            indices, count, max_frontier, index, candidate,
                            priority, minimum_priority)


@njit(cache=True)
def quality_guided_min_gradient(phase, valid, max_frontier):
    """Compiled cycle-domain equivalent of the bounded reference traversal."""
    rows, cols = phase.shape
    flat_phase = phase.ravel()
    flat_valid = valid.ravel()
    result = np.empty(flat_phase.size, dtype=np.float32)
    for index in range(result.size):
        result[index] = np.nan
    saved_priority = np.zeros(flat_phase.size, dtype=np.float32)
    state = np.zeros(flat_phase.size, dtype=np.uint8)
    order = np.full(flat_phase.size, -1, dtype=np.int64)
    priorities = np.empty(max_frontier + 1, dtype=np.float32)
    indices = np.empty(max_frontier + 1, dtype=np.int64)
    sequence = 0
    for start in range(flat_phase.size):
        if not flat_valid[start] or state[start] != 0:
            continue
        result[start] = flat_phase[start]
        state[start] = 2
        order[start] = sequence
        sequence += 1
        count = 0
        minimum_priority = -1e10
        row, col = start // cols, start % cols
        if col > 0:
            count, minimum_priority = _propose_min_gradient(
                flat_phase, flat_valid, result, saved_priority, state,
                priorities, indices, count, max_frontier, start - 1, start,
                minimum_priority)
        if col + 1 < cols:
            count, minimum_priority = _propose_min_gradient(
                flat_phase, flat_valid, result, saved_priority, state,
                priorities, indices, count, max_frontier, start + 1, start,
                minimum_priority)
        if row > 0:
            count, minimum_priority = _propose_min_gradient(
                flat_phase, flat_valid, result, saved_priority, state,
                priorities, indices, count, max_frontier, start - cols, start,
                minimum_priority)
        if row + 1 < rows:
            count, minimum_priority = _propose_min_gradient(
                flat_phase, flat_valid, result, saved_priority, state,
                priorities, indices, count, max_frontier, start + cols, start,
                minimum_priority)
        while count > 0:
            current = indices[count - 1]
            count -= 1
            state[current] = 2
            order[current] = sequence
            sequence += 1
            row, col = current // cols, current % cols
            if col > 0:
                count, minimum_priority = _propose_min_gradient(
                    flat_phase, flat_valid, result, saved_priority, state,
                    priorities, indices, count, max_frontier, current - 1,
                    current, minimum_priority)
            if col + 1 < cols:
                count, minimum_priority = _propose_min_gradient(
                    flat_phase, flat_valid, result, saved_priority, state,
                    priorities, indices, count, max_frontier, current + 1,
                    current, minimum_priority)
            if row > 0:
                count, minimum_priority = _propose_min_gradient(
                    flat_phase, flat_valid, result, saved_priority, state,
                    priorities, indices, count, max_frontier, current - cols,
                    current, minimum_priority)
            if row + 1 < rows:
                count, minimum_priority = _propose_min_gradient(
                    flat_phase, flat_valid, result, saved_priority, state,
                    priorities, indices, count, max_frontier, current + cols,
                    current, minimum_priority)
            if count == 0:
                minimum_priority = -1e10
                for postponed in range(flat_phase.size):
                    if state[postponed] == 3:
                        count, minimum_priority = _frontier_insert(
                            result, saved_priority, state, priorities, indices,
                            count, max_frontier, postponed, result[postponed],
                            saved_priority[postponed], minimum_priority)
    return result.reshape(rows, cols), order.reshape(rows, cols)
