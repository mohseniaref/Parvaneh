"""Minimum-cost-flow phase unwrapping on the dual network of Costantini.

An interferogram measures a phase difference modulo ``2 pi``, so the integer
number of whole turns that has to be added to every neighbouring difference is
unknown.  This module treats those unknowns as the flow on the edges of a
network and asks for the cheapest flow that makes the corrected field curl
free.

The network is the dual grid Costantini introduced for radar interferometry:
one node per 2x2 cell of the phase grid, one arc per pixel edge, and one extra
"ground" node that every border arc reaches.  Conservation of flow at a cell
node says that the four corrected steps around that cell add up to zero, so a
feasible flow is exactly a curl-free field, and the flow on an arc *is* the
integer ambiguity of that pixel edge.  The solver then minimises

.. math::

    \\sum_e w_e |k_e| ,

the total weighted discontinuity of the integer jumps.  This is the same
objective that the branch-cut and Flynn methods aim at, but here it is solved
to optimality instead of being reduced by a heuristic, because the objective
is a linear program and the constraint matrix of a network is totally
unimodular, so an integer optimum comes out of the continuous relaxation.

A few consequences are worth stating because they are visible in the output.
The ground node is what makes the network feasible: a pixel edge on the border
of the data has only one neighbouring cell, and the missing cell is the
ground, so boundary steps never force an inconsistency.  Cells that touch a
masked pixel are dropped, which means curl freeness is deliberately *not*
enforced there; a loop that walks around a masked hole can therefore still
contain a discontinuity, and that is a property of the data, not a defect.
Finally, the solution is unique in its objective value but not always in the
field itself, so the result is anchored at the first valid pixel of each
connected region and different regions keep independent offsets.

References
----------
Costantini, M. (1998). A novel phase unwrapping method based on network
programming. *IEEE Transactions on Geoscience and Remote Sensing*, 36(3),
813-821.  The dual network below is the one described there.

Chen, C. W., and Zebker, H. A. (2002). Phase unwrapping for large SAR
interferograms: statistical segmentation and generalized network models.
*IEEE Transactions on Geoscience and Remote Sensing*, 40(8), 1709-1719.
This is the formulation used by SNAPHU, which uses a different network (one
node per pixel) and is a useful cross-check but not the construction here.

Ahuja, R. K., Magnanti, T. L., and Orlin, J. B. (1993). *Network Flows:
Theory, Algorithms, and Applications*.  Prentice Hall.  The successive
shortest augmenting path method used below is Algorithm 9.5 (section 9.3)
there, which is stated for linear costs; the quadratic cost uses the same
augmentation with marginal costs, as in the convex-cost chapter.
"""

from collections import deque
from dataclasses import dataclass
from heapq import heapify, heappop, heappush

import numpy as np

from .path_following import phase_residues
from .quality import wrapped_gradients
from .reliability import _confidence

COST_SCALE = 1000
"""Weight of one reliable edge in integer arithmetic."""

COST_MODES = ("linear", "quadratic")
"""Names accepted by the ``cost`` argument."""

_INFINITY = 1 << 60


@dataclass
class NetworkFlowInfo:
    """Bookkeeping for one minimum-cost-flow run."""

    backend: str
    cost: str
    pixels: int
    nodes: int
    edges: int
    residues: int  # total absolute charge; one dipole counts as two
    augmentations: int
    components: int
    max_jump: int  # largest number of whole turns placed on one pixel edge
    ground_imbalance: int
    total_cost: int  # weighted |k| total, reported for both cost modes


def _check_phase(phase):
    phase = np.ascontiguousarray(phase, dtype=np.float64)
    if phase.ndim != 2 or min(phase.shape) < 2 or not np.isfinite(phase).all():
        raise ValueError(
            "phase must be a finite 2-D array with dimensions >= 2")
    return phase


def _edge_costs(confidence):
    """Integer unit cost of every pixel edge, from the ends' confidence.

    The cost of a pixel edge is the smaller confidence of its two endpoint
    pixels, which is the rule the least-squares backends already use, rescaled
    to integers so that the flow stays exact.  An edge with an unusable end has
    cost zero, which is what lets the network drop it for free.
    """
    scaled = np.rint(COST_SCALE * np.clip(confidence, 0.0, 1.0))
    scaled = scaled.astype(np.int64)
    scaled = np.where(confidence > 0.0, np.maximum(scaled, 1), 0)
    down = np.minimum(scaled[:-1, :], scaled[1:, :])
    right = np.minimum(scaled[:, :-1], scaled[:, 1:])
    return down, right


def _build(phase, confidence):
    """Build the dual network, its supplies, and the edge lookup tables."""
    rows, cols = phase.shape
    valid = confidence > 0.0
    cells = (valid[:-1, :-1] & valid[:-1, 1:]
             & valid[1:, 1:] & valid[1:, :-1])
    charge = phase_residues(phase, valid).astype(np.int64)
    down_cost, right_cost = _edge_costs(confidence)

    ncells = int(cells.sum())
    ground = ncells
    lookup = np.full((rows - 1, cols - 1), -1, dtype=np.int64)
    lookup[cells] = np.arange(ncells)
    # A cell that touches the mask is dropped, so it acts as the ground just
    # like a cell outside the data does.
    padded = np.full((rows + 1, cols + 1), ground, dtype=np.int64)
    padded[1:rows, 1:cols] = np.where(lookup < 0, ground, lookup)

    supply = np.zeros(ncells + 1, dtype=np.int64)
    supply[:ncells] = -charge[cells]
    supply[ground] = -supply[:ncells].sum()

    # An arc is oriented by rotating its pixel step a quarter turn clockwise on
    # the grid.  A down step dy[i, j] therefore runs left to right across the
    # edge, and a right step dx[i, j] runs from the lower cell to the upper one.
    # Using one rotation for both families is what makes the flow leaving a
    # cell equal the circulation of the corrected field around it, so that the
    # supplies below encode the true discrete curl.  Cells outside the data
    # are the ground.
    left = padded[1:rows, 0:cols]
    right = padded[1:rows, 1:cols + 1]
    upper = padded[0:rows, 1:cols]
    lower = padded[1:rows + 1, 1:cols]

    keep_down = ((left != ground) | (right != ground)).ravel()
    keep_right = ((upper != ground) | (lower != ground)).ravel()
    down_index = np.flatnonzero(keep_down)
    right_index = np.flatnonzero(keep_right)

    m_arc = np.full((rows - 1, cols), -1, dtype=np.int64)
    n_arc = np.full((rows, cols - 1), -1, dtype=np.int64)
    m_arc.ravel()[down_index] = np.arange(down_index.size)
    n_arc.ravel()[right_index] = down_index.size + np.arange(right_index.size)

    tail = np.concatenate([left.ravel()[down_index],
                           lower.ravel()[right_index]])
    head = np.concatenate([right.ravel()[down_index],
                           upper.ravel()[right_index]])
    unit_cost = np.concatenate([down_cost.ravel()[down_index],
                                right_cost.ravel()[right_index]])
    return (supply, tail, head, unit_cost, m_arc, n_arc, charge, cells,
            ground)


def _residual_lists(nodes, tail, head):
    """Adjacency of the residual network, in node order.

    The two residual arcs of an arc ``e`` are stored as one array of length
    ``2 * nedges``: slot ``e`` leaves the tail of the arc and raises its flow,
    and slot ``e + nedges`` leaves the head and lowers it.
    """
    nedges = tail.size
    source = np.concatenate([tail, head])
    order = np.argsort(source, kind="stable")
    counts = np.bincount(source, minlength=nodes)
    start = np.zeros(nodes + 1, dtype=np.int64)
    np.cumsum(counts, out=start[1:])
    return (start, order % nedges, order < nedges,
            np.concatenate([head, tail])[order])


def _solve(nodes, tail, head, unit_cost, supply, cost, max_iter):
    """Find the cheapest feasible flow by successive shortest augmentations.

    Every arc is stored as one signed flow, so the residual network of an arc
    offers two moves: raising the flow by one, which costs the marginal cost of
    the arc at its current value, and lowering it by one, which costs the
    marginal cost with the opposite sign.  Both marginal costs are monotone in
    the flow, which is why Dijkstra with potentials is enough even though some
    residual arcs carry negative cost.  All original costs are non-negative, so
    the potentials can start at zero.
    """
    nedges = tail.size
    flow = np.zeros(nedges, dtype=np.int64)
    excess = supply.copy()
    potential = np.zeros(nodes, dtype=np.int64)
    start, adjacency, forward, neighbour = _residual_lists(nodes, tail, head)

    distance = np.empty(nodes, dtype=np.int64)
    previous_node = np.empty(nodes, dtype=np.int64)
    previous_edge = np.empty(nodes, dtype=np.int64)
    previous_step = np.empty(nodes, dtype=np.int64)

    remaining = int(excess[excess > 0].sum())
    augmentations = 0
    while remaining > 0:
        if max_iter is not None and augmentations >= max_iter:
            raise RuntimeError(
                "minimum-cost flow stopped after max_iter=%d augmentations "
                "with %d units of imbalance left" % (max_iter, remaining))
        distance.fill(_INFINITY)
        previous_node.fill(-1)
        heap = [(0, int(node)) for node in np.flatnonzero(excess > 0)]
        heapify(heap)
        for _, node in heap:
            distance[node] = 0
        sink = -1
        while heap:
            dist, node = heappop(heap)
            if dist > distance[node]:
                continue
            if excess[node] < 0:
                sink = node
                break
            for slot in range(start[node], start[node + 1]):
                arc = adjacency[slot]
                other = neighbour[slot]
                if forward[slot]:
                    step = 1
                    if cost == "quadratic":
                        marginal = unit_cost[arc] * (2 * flow[arc] + 1)
                    else:
                        marginal = (unit_cost[arc] if flow[arc] >= 0
                                    else -unit_cost[arc])
                else:
                    step = -1
                    if cost == "quadratic":
                        marginal = unit_cost[arc] * (1 - 2 * flow[arc])
                    else:
                        marginal = (unit_cost[arc] if flow[arc] <= 0
                                    else -unit_cost[arc])
                reached = (dist + marginal + potential[node]
                           - potential[other])
                if reached < distance[other]:
                    distance[other] = reached
                    previous_node[other] = node
                    previous_edge[other] = arc
                    previous_step[other] = step
                    heappush(heap, (reached, int(other)))
        if sink < 0:
            raise RuntimeError(
                "minimum-cost flow is infeasible: a region of the dual "
                "network cannot balance its integer ambiguities")

        push = int(-excess[sink])
        node = sink
        while previous_node[node] >= 0:
            parent = int(previous_node[node])
            if 0 < excess[parent] < push:
                push = int(excess[parent])
            node = parent
        node = sink
        while previous_node[node] >= 0:
            flow[previous_edge[node]] += previous_step[node] * push
            excess[node] += push
            node = int(previous_node[node])
            excess[node] -= push
        remaining -= push
        augmentations += 1

        # Dijkstra is stopped at the first node that needs flow, so only the
        # nodes popped before it have a final distance.  Giving every other
        # node the distance of that sink keeps all reduced costs non-negative,
        # which is what lets the next Dijkstra start from zero again.
        threshold = distance[sink]
        settled = distance < threshold
        potential[settled] += distance[settled]
        potential[~settled] += threshold
    return flow, augmentations


def _integrate(phase, valid, turn_down, turn_right):
    """Add whole turns to every pixel along a spanning forest of ``valid``.

    The corrected steps are curl free inside the valid area, so any path
    between two pixels gives the same answer there.  A plain breadth-first walk
    therefore integrates the field exactly, and one walk per connected region
    fixes the arbitrary offset of that region.
    """
    rows, cols = phase.shape
    dx, dy = wrapped_gradients(phase)
    down = dy + 2.0 * np.pi * turn_down
    right = dx + 2.0 * np.pi * turn_right
    result = np.full(phase.shape, np.nan, dtype=np.float64)
    seen = np.zeros(phase.shape, dtype=bool)
    components = 0
    for row, col in map(tuple, np.argwhere(valid)):
        if seen[row, col]:
            continue
        components += 1
        seen[row, col] = True
        result[row, col] = phase[row, col]
        queue = deque([(row, col)])
        while queue:
            r, c = queue.popleft()
            if c + 1 < cols and valid[r, c + 1] and not seen[r, c + 1]:
                seen[r, c + 1] = True
                result[r, c + 1] = result[r, c] + right[r, c]
                queue.append((r, c + 1))
            if c > 0 and valid[r, c - 1] and not seen[r, c - 1]:
                seen[r, c - 1] = True
                result[r, c - 1] = result[r, c] - right[r, c - 1]
                queue.append((r, c - 1))
            if r + 1 < rows and valid[r + 1, c] and not seen[r + 1, c]:
                seen[r + 1, c] = True
                result[r + 1, c] = result[r, c] + down[r, c]
                queue.append((r + 1, c))
            if r > 0 and valid[r - 1, c] and not seen[r - 1, c]:
                seen[r - 1, c] = True
                result[r - 1, c] = result[r, c] - down[r - 1, c]
                queue.append((r - 1, c))
    return result, components


def network_flow_unwrap(phase, weight=None, mask=None, *, cost="linear",
                        backend="python", max_iter=None, return_info=False):
    """Unwrap a 2-D phase field through a minimum-cost-flow network.

    The phase is turned into a network whose nodes are the 2x2 cells of the
    grid, as described in the module docstring, and the integer ambiguity of
    every pixel edge is chosen so that the total weighted discontinuity
    ``sum(w * |k|)`` is as small as possible.  The minimum is exact: the
    problem is a linear program on a network, so an integer optimum exists and
    the successive shortest path solver finds one.

    Parameters
    ----------
    phase : array_like
        Two-dimensional array of wrapped phase in radians, all values finite.
    weight : array_like, optional
        Per-pixel confidence in ``[0, 1]`` with ``phase.shape``.  The cost of a
        pixel edge is the smaller of its two endpoints, so a confident edge is
        expensive to jump across.  ``None`` means every valid pixel is equally
        reliable.
    mask : array_like of bool, optional
        ``True`` where the phase is usable.  Cells touching a masked pixel are
        removed from the network and the matching pixels come back as ``NaN``.
        A zero ``weight`` also removes a pixel, as in
        :func:`parvaneh.reliability_unwrap`.
    cost : {"linear", "quadratic"}, optional
        ``"linear"`` minimises the total discontinuity, which is the classical
        formulation.  ``"quadratic"`` minimises the sum of squared jumps, which
        spreads a jump over several edges instead of concentrating it on one.
    backend : {"python"}, optional
        Only the reference implementation is provided.
    max_iter : int, optional
        Hard limit on the number of flow augmentations.  ``None`` lets the
        solver run to the end, which always terminates for integer input.
    return_info : bool, optional
        Return the result together with a :class:`NetworkFlowInfo` record.

    Returns
    -------
    result : ndarray
        Unwrapped phase, ``NaN`` where the phase is not usable.  It keeps the
        wrapped values of the input, so ``wrap(result) == wrap(phase)``, and
        every step between neighbouring valid pixels is the wrapped step plus
        a whole number of turns.
    info : NetworkFlowInfo
        Only returned when ``return_info`` is true.

    Raises
    ------
    ValueError
        If ``phase`` is not a finite 2-D array of at least 2 by 2, if ``cost``
        or ``backend`` is unknown, or if ``weight`` or ``mask`` is malformed.
    RuntimeError
        If ``max_iter`` is reached before the flow is complete.

    Examples
    --------
    A plane that wraps once is recovered exactly, up to the constant offset
    that no unwrapping method can determine:

    >>> import numpy as np
    >>> ramp = np.arange(12).reshape(3, 4) * 0.9
    >>> from parvaneh.synthetic import wrap_phase
    >>> result = network_flow_unwrap(wrap_phase(ramp))
    >>> bool(np.abs(wrap_phase(result - ramp)).max() < 1e-5)
    True

    Notes
    -----
    The number of augmentations is reported in the info record and grows with
    the number of residues, not with the number of pixels, so a clean scene
    solves without any augmentation at all.
    """
    phase = _check_phase(phase)
    if cost not in COST_MODES:
        raise ValueError("cost must be one of %s" % (", ".join(COST_MODES),))
    if backend != "python":
        raise ValueError("backend must be 'python'")
    if max_iter is not None and max_iter < 0:
        raise ValueError("max_iter must be nonnegative or None")
    confidence = _confidence(phase, mask, weight)

    (supply, tail, head, unit_cost, m_arc, n_arc, charge, cells,
     ground) = _build(phase, confidence)
    flow, augmentations = _solve(supply.size, tail, head, unit_cost, supply,
                                 cost, max_iter)

    turn_down = np.zeros((phase.shape[0] - 1, phase.shape[1]), dtype=np.int64)
    turn_right = np.zeros((phase.shape[0], phase.shape[1] - 1),
                          dtype=np.int64)
    located = m_arc >= 0
    turn_down[located] = flow[m_arc[located]]
    located = n_arc >= 0
    turn_right[located] = flow[n_arc[located]]

    result, components = _integrate(phase, confidence > 0.0, turn_down,
                                    turn_right)
    if not return_info:
        return result
    info = NetworkFlowInfo(
        backend=backend,
        cost=cost,
        pixels=int(np.count_nonzero(confidence > 0.0)),
        nodes=supply.size,
        edges=int(tail.size),
        residues=int(np.abs(charge).sum()),
        augmentations=augmentations,
        components=components,
        max_jump=int(np.abs(flow).max()) if flow.size else 0,
        ground_imbalance=int(supply[ground]),
        total_cost=int((unit_cost * np.abs(flow)).sum()),
    )
    return result, info
