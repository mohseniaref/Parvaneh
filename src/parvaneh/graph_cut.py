"""Phase unwrapping by graph cuts.

Labels instead of jumps
-----------------------
The network methods of :mod:`parvaneh.network_flow` and
:mod:`parvaneh.flow_nd` choose a whole number of turns ``k`` for every edge of
the grid.  A graph cut chooses instead a whole number of turns for every
*sample*: the answer is

    phi = psi + 2 pi K,

with one integer ``K`` per sample, and the corrected step of an edge follows
from the labels of its two samples,

    g(e) = (psi[j] - psi[i]) + 2 pi (K[j] - K[i]).

That is a change of variable and not a different model.  Adding the same whole
number of turns to every label changes no step, so the labels carry one
arbitrary constant per connected group of usable samples, exactly like the
integrated fields of the other methods.  What the labels give for free is the
closure constraint: the steps of any labelling add up to zero around every loop
of the grid, because they are the steps of the single valued field ``phi``.
The two formulations therefore have the same feasible corrections once the
gauge is fixed, but they assign different costs to those corrections.

The objective
-------------
The cost of a labelling is the weighted ``p``-th power of the corrected steps,

    E(K) = sum over edges  w(e) * abs(g(e)) ** p,

where ``e`` runs over the edges whose two samples are both usable and ``w`` is
the smaller confidence of the two samples, which is the weight the other
methods give an edge.  ``p`` may be any number from one upwards; the published
variant of the method uses ``1 <= p <= 2`` and the default here is ``p = 1``,
the weighted total variation.  The two ends of that range are worth naming:

``p = 1``
    the weighted total variation of the corrected steps.  This is not a
    constant multiple of the minimum-cost-flow objective: graph cuts minimise
    ``sum(w * abs(wrapped_step + 2 pi k))``, whereas the flow methods minimise
    ``sum(w * abs(k))``.  They search the same feasible corrections but rank
    them by different costs.  The ``total_cost`` field of
    :class:`parvaneh.network_flow.NetworkFlowInfo` is an integer count of unit
    costs; :data:`parvaneh.network_flow.COST_SCALE` converts that count to the
    weighted whole-turn objective.
``p = 2``
    the weighted sum of squared corrected steps, which is the cost the Poisson
    formulation of :func:`parvaneh.core.unwrap` minimises over real valued
    fields.  Here the labels are whole numbers, so this is the integer form of
    the same cost and not the same answer.

Expansions of the labelling
---------------------------
The cost is not convex in the labels, and a method that raises one label at a
time gets stuck at once: raising a single sample by one turn costs its edges
more than the turns it saves.  The graph cut decides about all the samples
together instead.  A step considers the labellings

    K + s * x,   with x(p) in {0, 1},

that is, it decides for every sample whether to raise it by ``s`` turns.  The
labels can climb many turns, because a step is repeated as long as it finds an
improvement; ``max_jump`` limits the size of a step and not the answer.  Step
``s`` takes the cost of an edge from ``V(g)`` to one of four values, with
``span = 2 pi s`` and ``V(t) = w * abs(t) ** p``:

    ==============  =============  ===========================
    x(i), x(j)      cost           in words
    ==============  =============  ===========================
    0, 0            V(g)           unchanged
    1, 0            V(g - span)    raised at the tail only
    0, 1            V(g + span)    raised at the head only
    1, 1            V(g)           both raised, unchanged
    ==============  =============  ===========================

A function of two binary variables whose two diagonal entries are equal is the
sum of a constant, two unary terms and one pairwise term (Kolmogorov and Zabih
2004), and this table is of that kind.  With ``current = V(g)``,
``plus = V(g + span)`` and ``minus = V(g - span)`` the decomposition is

    half(e)   = (plus - minus) / 2,
    lambda(e) = (plus + minus) / 2 - current,

where ``-half`` is paid when the tail sample is raised, ``+half`` when the head
sample is raised, and ``lambda`` when exactly one of the two is raised.  The
constant term is dropped, because it is the same for every step of the
expansion.  ``lambda`` is never negative exactly when ``V`` is convex, which is
what ``p >= 1`` guarantees; a smaller ``p`` would make the cut wrong rather
than merely slow, so it is refused.  A value that rounds to a small negative
number is clamped to zero.

One step as a minimum cut
-------------------------
With that decomposition, the best step is a minimum cut of a graph with

* one node per usable sample, a source and a sink;
* an arc from the source of capacity ``unary(i)`` where

      unary(i) = sum(half over the edges whose head is i)
               - sum(half over the edges whose tail is i)

  is positive, and an arc to the sink of capacity ``-unary(i)`` where it is
  negative;
* an arc of capacity ``lambda(e)`` in each direction between the two samples of
  every edge.

A cut that leaves a sample on the source side means ``x = 0`` there and one
that leaves it on the sink side means ``x = 1``.  The value of the cut is
``C + E(x) - E(0)``, with ``C`` the sum of ``-unary(i)`` over the negative
ones, so the cheapest cut is the best step.  A maximum flow gives the cheapest
cut and the side of the source at the same time: the samples that stay
reachable from the source in the residual network once the flow is complete are
the ones with ``x = 0`` (maximum flow equals minimum cut; the layered algorithm
used here is Dinitz 1970).

The cut is exact for the cost of the expansion, and the phase data are used
once more to recompute the true cost of the proposed labelling, because the
decomposition is arithmetic on floating point differences while the labels
themselves are exact integers.  A step is taken only when the recomputed cost
is really lower, so an arithmetic surprise can lose an improvement and never
invent one.  The all zero labelling is always available and its cut costs
``C``, so no step can raise the cost, ``E(x) <= E(0)``: every accepted step of
the descent is downhill, and the cost never increases.

Steps and stopping
------------------
The schedule is the powers of two from the largest one not above ``max_jump``
down to one.  A coarse step moves a whole region first, the fine steps that
follow trim its boundary, and each step is repeated while it keeps helping.
A sweep over the schedule ends the search when it leaves the cost unchanged, so
the answer is a local minimum of the cost: the method promises nothing beyond
that as it stands, and ``info.proven`` is false by construction.  ``max_iter``
bounds the number of sweeps and defaults to twenty per step of the schedule.

Two dimensions, three dimensions and more
-----------------------------------------
A graph cut labels samples, so no axis of the grid is special and no structure
of the grid has to be a network.  The dual network that makes the two
dimensional flow methods exact (Costantini 1998) has no three dimensional
counterpart, which is why the volumetric flow methods need an outer search or a
relaxation (Chen and Zebker 2002); the graph cut needs none of it and handles
an image and a stack of images with the same code and the same cost.  A third
axis is not free: the cut is again a heuristic, it is only a heuristic whose
step is available in any number of dimensions, and whose cost is written once
for the whole array.

Gauge
-----
The labels are defined only up to one whole number of turns per connected group
of usable samples, and ``align`` fixes that constant the way
:func:`parvaneh.flow_nd.integrate_edges` fixes it: ``"zero"`` keeps the wrapped
value of the first sample in raster order, and ``"mean"`` and ``"median"``
remove the whole number of turns closest to the average or the median label of
the group.  Every one of them is a pure shift of the field, so no corrected
step depends on the choice.  None can recover a genuine offset of many turns,
so this is a convention and not a measurement.

Scale
-----
A two dimensional grid of ``R`` by ``C`` samples has ``R * C`` nodes and
``R * (C - 1) + (R - 1) * C`` edges, and a step's graph has the same order of
size, with at most two extra arcs per sample.  Every step of the schedule
builds a maximum flow on that graph from scratch, and the layered algorithm
costs ``O(V ** 2 * E)`` on capacitated graphs in the worst case, so nothing
here reaches large images in reasonable time.  The arcs are built in a Python
loop and the flow runs on Python lists.  That is deliberate: this is a
reference implementation whose answers the faster paths can be checked
against.

References
----------
J. M. Bioucas-Dias and G. Valadao, "Phase unwrapping via graph cuts," *IEEE
Transactions on Image Processing* 16(3) (2007) 698-709.

Y. Boykov, O. Veksler and R. Zabih, "Fast approximate energy minimization via
graph cuts," *IEEE Transactions on Pattern Analysis and Machine Intelligence*
23(11) (2001) 1222-1239.

V. Kolmogorov and R. Zabih, "What energy functions can be minimized via graph
cuts?," *IEEE Transactions on Pattern Analysis and Machine Intelligence* 26(2)
(2004) 147-159.

Y. Boykov and V. Kolmogorov, "An experimental comparison of min-cut/max-flow
algorithms for energy minimization in vision," *IEEE Transactions on Pattern
Analysis and Machine Intelligence* 26(9) (2004) 1124-1137.

E. A. Dinitz, "Algorithm for solution of a problem of maximum flow in networks
with power estimation," *Soviet Mathematics Doklady* 11 (1970) 1277-1280.

M. Costantini, "A novel phase unwrapping method based on network programming,"
*IEEE Transactions on Geoscience and Remote Sensing* 36(3) (1998) 813-821.

C. W. Chen and H. A. Zebker, "Phase unwrapping for large SAR interferograms:
statistical segmentation and generalized network models," *IEEE Transactions on
Geoscience and Remote Sensing* 40(8) (2002) 1709-1719.

D. C. Ghiglia and M. D. Pritt, *Two-Dimensional Phase Unwrapping: Theory,
Algorithms and Software*, Wiley, 1998.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .flow_nd import grid_edges
from .reliability import _axis_confidence, _check_phase, _confidence

_TWO_PI = 2.0 * np.pi

#: Residual capacity below which an arc counts as saturated.
_EPSILON = 1e-12

#: Gauge conventions accepted by :func:`puma_unwrap`.
ALIGN_MODES = ("zero", "mean", "median")


@dataclass
class GraphCutInfo:
    """Bookkeeping for one graph cut run.

    ``pixels`` counts the usable samples, which are the nodes the cut decides
    about, while ``nodes`` and ``edges`` describe the whole grid of the input
    array.  ``cuts`` is the number of maximum flow problems that were solved,
    ``accepted`` how many of them lowered the cost, and ``passes`` how many
    sweeps over the step schedule were made.  ``max_step`` is the largest step
    the schedule used and ``max_jump`` the largest whole turn correction any
    single edge received.

    ``objective`` is ``sum(weight * abs(corrected step) ** p)`` and
    ``initial_objective`` its value at the start.  ``total_variation`` is the
    same sum with ``p = 1``.  Both are measured in the units of the phase, so
    they are ``2 pi`` times the quantities of the same names in
    :class:`parvaneh.flow_nd.FlowNDInfo`, which counts whole turns.
    ``proven`` is always false: an expansion stops at a local minimum of a cost
    that is not convex, and nothing stronger can be claimed.
    """

    backend: str
    p: float
    shape: tuple
    pixels: int
    nodes: int
    edges: int
    components: int
    cuts: int
    accepted: int
    passes: int
    max_step: int
    max_jump: int
    initial_objective: float
    objective: float
    total_variation: float
    proven: bool


class _Dinic:
    """Maximum flow of a capacitated graph by the layered algorithm.

    The graph is held in four parallel Python lists: ``tail``, ``tip`` and
    ``capacity`` are indexed by arc, and ``head`` is the adjacency list of
    every node.  An arc and its residual twin are the consecutive entries
    ``2 * a`` and ``2 * a + 1``, so the twin of an arc is found by flipping its
    last bit and a push never has to search for it.

    This is a reference implementation: a compiled maximum flow library would
    be much faster on a real image, and none is a dependency of this package.
    """

    __slots__ = ("tail", "tip", "capacity", "head")

    def __init__(self):
        self.tail = []
        self.tip = []
        self.capacity = []
        self.head = []

    def _node(self, node):
        """Make sure the adjacency list of ``node`` exists."""
        while len(self.head) <= node:
            self.head.append([])

    def add(self, tail, tip, capacity):
        """Add one directed arc, with its residual twin right behind it.

        An arc of capacity zero or less is dropped, which is how a term that
        does not apply to an edge disappears from the graph.
        """
        if capacity <= 0.0:
            return
        tail = int(tail)
        tip = int(tip)
        self._node(tail)
        self._node(tip)
        arc = len(self.tail)
        self.tail.extend((tail, tip))
        self.tip.extend((tip, tail))
        self.capacity.extend((float(capacity), 0.0))
        self.head[tail].append(arc)
        self.head[tip].append(arc + 1)

    def add_undirected(self, first, second, capacity):
        """Add one arc of equal capacity in each direction.

        The pair acts as a single undirected edge of the cut: a cut that
        separates the two samples pays ``capacity`` whichever side each of them
        ended up on, which is the pairwise term of a step.
        """
        if capacity <= 0.0:
            return
        first = int(first)
        second = int(second)
        self._node(first)
        self._node(second)
        arc = len(self.tail)
        self.tail.extend((first, second))
        self.tip.extend((second, first))
        self.capacity.extend((float(capacity), float(capacity)))
        self.head[first].append(arc)
        self.head[second].append(arc + 1)

    def max_flow(self, source, sink):
        """Push as much flow as possible from ``source`` to ``sink``.

        The layers come from a full breadth first search of the residual
        network, the blocking flow from an iterative depth first walk with one
        position pointer per node.  The walk is iterative on purpose: a path
        can be as long as the graph has nodes, which is deeper than the
        interpreter's own recursion limit allows.

        A walk that reaches the sink is saturated at the *first* saturated arc
        of the path: the path is cut back to that arc's tail so the saturated
        arc is skipped by the scan from then on.  Cutting back only the last
        arc of the path would leave such an arc in the middle of every later
        path, and the bottleneck of the path would be zero forever.
        """
        if source == sink:
            return 0.0
        self._node(source)
        self._node(sink)
        nodes = len(self.head)
        total = 0.0
        while True:
            level = [-1] * nodes
            level[source] = 0
            queue = deque([source])
            while queue:
                node = queue.popleft()
                for arc in self.head[node]:
                    tip = self.tip[arc]
                    if self.capacity[arc] > _EPSILON and level[tip] < 0:
                        level[tip] = level[node] + 1
                        queue.append(tip)
            if level[sink] < 0:
                return total
            position = [0] * nodes
            path = []
            node = source
            while True:
                if node == sink:
                    bottleneck = min(self.capacity[arc] for arc in path)
                    for arc in path:
                        self.capacity[arc] -= bottleneck
                        self.capacity[arc ^ 1] += bottleneck
                    total += bottleneck
                    stop = 0
                    while self.capacity[path[stop]] > _EPSILON:
                        stop += 1
                    node = self.tail[path[stop]]
                    del path[stop:]
                    continue
                found = False
                while position[node] < len(self.head[node]):
                    arc = self.head[node][position[node]]
                    if (self.capacity[arc] > _EPSILON
                            and level[self.tip[arc]] == level[node] + 1):
                        path.append(arc)
                        node = self.tip[arc]
                        found = True
                        break
                    position[node] += 1
                if found:
                    continue
                level[node] = -1
                if node == source:
                    break
                node = self.tail[path.pop()]

    def source_side(self, source):
        """Nodes still reachable from ``source`` in the residual network.

        After a maximum flow those nodes are one side of a cheapest cut, and
        the side of the source is the one that decides a step of the
        expansion.
        """
        self._node(source)
        reach = np.zeros(len(self.head), dtype=bool)
        reach[source] = True
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for arc in self.head[node]:
                tip = self.tip[arc]
                if self.capacity[arc] > _EPSILON and not reach[tip]:
                    reach[tip] = True
                    queue.append(tip)
        return reach


def _expansion(weights, gradients, p, step, tail, head, pixels):
    """Solve one binary expansion of the labelling, as a minimum cut.

    Parameters
    ----------
    weights, gradients : numpy.ndarray
        Weight and current corrected step of every usable edge.
    p : float
        Exponent of the cost.
    step : int
        Size of the step, in whole turns.
    tail, head : numpy.ndarray
        Endpoints of every usable edge, numbered locally to the cut.
    pixels : int
        Number of usable samples.

    Returns
    -------
    numpy.ndarray
        One where a sample is raised by ``step`` turns, zero where it is not.
    float
        Change of the cost caused by the step, never positive because the
        empty step is always available.

    Notes
    -----
    The decomposition and the wiring of the graph are the ones derived in the
    module docstring.  The change of the cost is recomputed from the labels
    rather than read off the value of the cut, because the cut value is the
    value of the decomposed cost up to the constant that was dropped, and the
    two only differ by arithmetic of the order of the rounding error.
    """
    span = _TWO_PI * step
    current = weights * np.abs(gradients) ** p
    plus = weights * np.abs(gradients + span) ** p
    minus = weights * np.abs(gradients - span) ** p
    half = 0.5 * (plus - minus)
    penalty = np.maximum(0.5 * (plus + minus) - current, 0.0)
    unary = (np.bincount(head, weights=half, minlength=pixels)
             - np.bincount(tail, weights=half, minlength=pixels))
    source, sink = pixels, pixels + 1
    graph = _Dinic()
    for node in np.nonzero(unary > 0.0)[0]:
        graph.add(source, node, unary[node])
    for node in np.nonzero(unary < 0.0)[0]:
        graph.add(node, sink, -unary[node])
    for edge in np.nonzero(penalty > 0.0)[0]:
        graph.add_undirected(tail[edge], head[edge], penalty[edge])
    graph.max_flow(source, sink)
    move = (~graph.source_side(source)[:pixels]).astype(np.int64)
    change = float(np.dot(unary, move)
                   + np.sum(penalty * (move[tail] != move[head])))
    return move, change


def _components(tail, head, pixels):
    """Group the usable samples whose labels are tied to each other.

    Two samples belong to the same group when an edge between them is usable,
    which is the grouping :func:`parvaneh.flow_nd.integrate_edges` integrates
    over.  The constant of a group is the one thing neither the data nor the
    cut can fix.

    Parameters
    ----------
    tail, head : numpy.ndarray
        Endpoints of every usable edge, numbered locally.
    pixels : int
        Number of usable samples.

    Returns
    -------
    numpy.ndarray
        Group number of every usable sample.
    int
        Number of groups.
    """
    if pixels == 0:
        return np.zeros(0, dtype=np.int64), 0
    if tail.size == 0:
        return np.arange(pixels, dtype=np.int64), pixels
    rows = np.concatenate([tail, head])
    columns = np.concatenate([head, tail])
    graph = csr_matrix((np.ones(rows.size, dtype=np.int8), (rows, columns)),
                       shape=(pixels, pixels))
    components, labels = connected_components(graph, directed=False)
    return np.asarray(labels, dtype=np.int64), int(components)


def _gauge(turns, labels, components, align):
    """Whole turn offset to remove from every group of samples.

    ``"zero"`` keeps the label of the first sample of the group in raster order
    at zero, which is what :mod:`parvaneh.network_flow` does.  ``"mean"`` and
    ``"median"`` remove the whole number of turns closest to the average or the
    median label of the group, which is what
    :func:`parvaneh.flow_nd.integrate_edges` does.  All three are shifts of the
    field and change no corrected step.

    Parameters
    ----------
    turns : numpy.ndarray
        Label of every usable sample.
    labels : numpy.ndarray
        Group number of every usable sample.
    components : int
        Number of groups.
    align : {"zero", "mean", "median"}
        Convention to apply.

    Returns
    -------
    numpy.ndarray
        Offset of every group.
    """
    offsets = np.zeros(components, dtype=np.int64)
    if components == 0:
        return offsets
    if align == "zero":
        first = np.full(components, turns.size, dtype=np.int64)
        np.minimum.at(first, labels, np.arange(turns.size))
        return turns[first]
    if align == "mean":
        total = np.bincount(labels, weights=turns.astype(np.float64),
                            minlength=components)
        count = np.bincount(labels, minlength=components)
        return np.rint(total / np.maximum(count, 1)).astype(np.int64)
    order = np.argsort(labels, kind="stable")
    starts = np.searchsorted(labels[order], np.arange(components + 1))
    for group in range(components):
        block = turns[order[starts[group]:starts[group + 1]]]
        offsets[group] = int(np.rint(np.median(block)))
    return offsets


def puma_unwrap(phase, weight=None, mask=None, *, p=1.0, max_jump=1,
                align="zero", max_iter=None, return_info=False):
    """Unwrap a phase field by cutting the labels into steps.

    Parameters
    ----------
    phase : array_like of float
        Wrapped phase in radians, with at least two axes and at least two
        samples along every axis.
    weight : array_like of float, optional
        Nonnegative per-sample confidence; zero removes a sample.
    mask : array_like of bool, optional
        Samples to use, combined with ``weight`` the way
        :func:`parvaneh.reliability.pixel_reliability` combines them.
    p : float, optional
        Exponent of the cost, at least one, see the module docstring.
    max_jump : int, optional
        Largest whole turn step of the schedule, which is made of every power
        of two up to it.  A step is repeated while it helps, so the labels can
        climb many turns: this bounds the size of a step and not the answer.
    align : {"zero", "mean", "median"}, optional
        Gauge of every connected group of usable samples, see :func:`_gauge`.
    max_iter : int, optional
        Bound on the number of sweeps over the schedule, twenty per step of the
        schedule by default.
    return_info : bool, optional
        Also return a :class:`GraphCutInfo`.

    Returns
    -------
    numpy.ndarray
        Unwrapped field, ``NaN`` where the confidence is zero.
    GraphCutInfo
        Only when ``return_info`` is true.

    Raises
    ------
    ValueError
        If the phase is not finite, if an axis is shorter than two samples, if
        the array has fewer than two axes, if ``p`` is below one, or if another
        option is unknown or out of range.

    Examples
    --------
    A stack is unwrapped with its third axis in the problem.  Nothing has to be
    asked for: the labels are those of the samples, so the same call solves an
    image and a volume.

    >>> volume = ...                                  # doctest: +SKIP
    >>> field = puma_unwrap(volume, p=1, max_jump=2)  # doctest: +SKIP
    """
    array = _check_phase(phase)
    if array.ndim < 2:
        raise ValueError("puma_unwrap needs at least two dimensions")
    if not np.isfinite(p) or p < 1.0:
        raise ValueError("p must be a finite number of at least one")
    if max_jump < 1 or max_jump != int(max_jump):
        raise ValueError("max_jump must be a positive whole number")
    if max_iter is not None and (max_iter < 1 or max_iter != int(max_iter)):
        raise ValueError("max_iter must be a positive whole number")
    if align not in ALIGN_MODES:
        raise ValueError("align must be one of %s" % (ALIGN_MODES,))
    p = float(p)
    steps = [1]
    while 2 * steps[-1] <= int(max_jump):
        steps.append(2 * steps[-1])
    steps.reverse()
    confidence = _confidence(array, mask, weight)
    nodes, tail_all, head_all = grid_edges(array.shape)
    edge_confidence = np.concatenate(
        [_axis_confidence(confidence, axis).ravel()
         for axis in range(array.ndim)])
    # The step of an edge is the plain difference of the wrapped phase, the
    # one the labels add whole turns to.  Its principal value would drop a
    # whole turn that no label can then pay back, and a step of magnitude
    # under pi is already the smallest it can be made, so the zero labelling
    # would be optimal and no step would ever be accepted.
    edge_phase = np.concatenate(
        [np.diff(array, axis=axis).ravel()
         for axis in range(array.ndim)])
    inside = edge_confidence > 0.0
    weights = edge_confidence[inside]
    edge_phase = edge_phase[inside]
    valid = confidence.ravel() > 0.0
    pixels = int(np.count_nonzero(valid))
    local = np.full(nodes.size, -1, dtype=np.int64)
    local[valid] = np.arange(pixels, dtype=np.int64)
    tail = local[tail_all[inside]]
    head = local[head_all[inside]]
    gradients = edge_phase.copy()
    turns = np.zeros(pixels, dtype=np.int64)
    objective = float(np.sum(weights * np.abs(gradients) ** p))
    initial = objective
    budget = 20 * len(steps) if max_iter is None else int(max_iter)
    cuts, accepted, passes = 0, 0, 0
    while passes < budget:
        passes += 1
        improved = False
        for step in steps:
            while True:
                move, change = _expansion(weights, gradients, p, step,
                                          tail, head, pixels)
                cuts += 1
                if change >= -_EPSILON * max(1.0, objective):
                    break
                turns += step * move
                gradients = edge_phase + _TWO_PI * (turns[head] - turns[tail])
                objective = float(np.sum(weights * np.abs(gradients) ** p))
                accepted += 1
                improved = True
        if not improved:
            break
    labels, components = _components(tail, head, pixels)
    offsets = _gauge(turns, labels, components, align)
    flat_turns = np.zeros(nodes.size, dtype=np.int64)
    flat_turns[valid] = turns - offsets[labels]
    flat_result = np.full(nodes.size, np.nan)
    flat_result[valid] = array.ravel()[valid] + _TWO_PI * flat_turns[valid]
    result = flat_result.reshape(array.shape)
    if not return_info:
        return result
    jump = np.abs(turns[head] - turns[tail]) if tail.size else np.zeros(0)
    info = GraphCutInfo(
        backend="graph-cut", p=p, shape=array.shape, pixels=pixels,
        nodes=int(nodes.size), edges=int(tail_all.size), components=components,
        cuts=cuts, accepted=accepted, passes=passes, max_step=steps[0],
        max_jump=int(jump.max()) if jump.size else 0,
        initial_objective=initial, objective=objective,
        total_variation=float(np.sum(weights * np.abs(gradients))),
        proven=False)
    return result, info
