"""Cycle-constrained integer flow solvers on general graphs.

Phase unwrapping on a graph needs no phase image.  Put on every edge the
wrapped step

.. math::

    \\Delta\\psi_e = \\psi_v - \\psi_u - 2\\pi\\,n_e ,
    \\qquad n_e = \\operatorname{rint}\\!\\left(
    \\frac{\\psi_v - \\psi_u}{2\\pi}\\right),

measured from the tail ``u`` to the head ``v`` of the edge, so that
``\\Delta\\psi_e`` lies in ``(-\\pi, \\pi]``.  The unwrapped field differs from
the observed one by a whole number of turns per vertex, so the corrected step
of an edge is

.. math::

    \\Delta\\psi_e + 2\\pi k_e , \\qquad k_e = K_v - K_u \\in \\mathbb{Z} .

Nothing else ties the unknowns together except consistency: walking around a
closed loop of the graph, the corrected steps must sum to zero.  Since ``k``
is a difference of vertex potentials, its curl vanishes automatically, and the
problem reduces to choosing one integer per edge so that every loop closes,

.. math::

    \\sum_{e \\in f} \\sigma_{fe} k_e = -r_f ,
    \\qquad r_f = \\operatorname{rint}\\!\\left(
    \\frac{1}{2\\pi} \\sum_{e \\in f} \\sigma_{fe} \\Delta\\psi_e \\right).

Here ``r_f`` is the residue of the loop and ``\\sigma_{fe} = \\pm 1`` says
whether the loop walks edge ``e`` along the edge direction.  A basis of the
cycle space is enough, because satisfying a generating set of loops forces all
the others to close as well, and a basis always has a solution: the jumps along
the edges that carry the basis are free, so any residue can be met.  A
*redundant* loop set need not have a solution.  Loops that cancel edge by edge
impose the same combination on their residues, and ``rint`` is not additive, so
near half-integer circulations can leave the residues of a redundant set
inconsistent; :func:`integer_flow` reports that case instead of looping.

Which solver is used depends on how the edges sit in that basis:

* when every edge appears in at most two basis cycles, with opposite signs,
  the constraints are the conservation laws of a flow network whose nodes are
  the cycles and whose arcs are the edges.  This *dual network* is a minimum
  cost flow problem, solved exactly by successive shortest augmenting paths;
* otherwise the constraints are a general integer program, still solved
  exactly but by branch and bound: our own search
  (:func:`integer_flow`) over the linear-programming relaxation, because the
  SciPy release this package requires has no mixed-integer solver.

The distinction is not academic.  On a two-dimensional grid an edge is shared
by at most two cells, so the first case always applies: that dual network is
the one Costantini (1998) introduced and the one
:mod:`parvaneh.network_flow` builds.  In three or more dimensions an interior
edge belongs to four or more plaquettes, the constraint matrix is no longer a
network matrix, and the problem becomes NP-hard in general (Chen and Zebker,
2002, section IV).  The surface of a cube is still a network, which is the
formal reason why a stack of *independently* unwrapped interferograms stays
easy while unwrapping the volume jointly does not.

Costs
-----
Every edge carries a convex cost of its integer jump.  :class:`LinearCost` is
the weighted total variation that Costantini's formulation minimises;
:class:`QuadraticCost` is the Gaussian term inside SNAPHU's statistical costs;
:class:`ShelfCost` adds the flat cap of SNAPHU's deformation model and is *not*
convex, so :func:`convex_flow` rejects it and
:mod:`parvaneh.stat_costs` solves it with a warm-started scheme instead.

References
----------
Costantini, M. (1998). A novel phase unwrapping method based on network
programming. *IEEE Transactions on Geoscience and Remote Sensing*, 36(3),
813-821.

Chen, C. W., and Zebker, H. A. (2002). Phase unwrapping for large SAR
interferograms: statistical segmentation and generalized network models.
*IEEE Transactions on Geoscience and Remote Sensing*, 40(8), 1709-1719.

Ahuja, R. K., Magnanti, T. L., and Orlin, J. B. (1993). *Network Flows:
Theory, Algorithms, and Applications*.  Prentice Hall.  Section 14.5 is the
convex-cost successive shortest path method implemented in
:func:`convex_flow`.

Huangfu, Q., and Hall, J. A. J. (2018). Parallelizing the dual revised simplex
method. *Mathematical Programming Computation*, 10(1), 119-142.  This is the
solver behind ``scipy.optimize.linprog(method="highs")``, used for the general
integer program.
"""

from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix

_TWO_PI = 2.0 * np.pi
_INFINITY = np.inf


@dataclass
class CycleBasis:
    """Cycle basis and spanning forest of one oriented graph.

    Attributes
    ----------
    cycles : list of tuple of numpy.ndarray
        For each basis cycle, the indices of the edges it visits and the sign
        ``+1`` when the cycle walks an edge along the edge direction and
        ``-1`` when it walks it backwards.
    roots : numpy.ndarray
        One vertex per connected component; these are the anchors used by
        :func:`integrate_gradients`.
    parent, parent_edge, parent_dir : numpy.ndarray
        The spanning forest: the parent vertex of each vertex, the edge that
        reaches it, and ``+1`` when that edge points from the parent to the
        vertex.  Roots have ``parent == -1``.
    order : numpy.ndarray
        Breadth-first order of the forest, parents before children.
    in_tree : numpy.ndarray
        Mask of the edges that belong to the forest.
    """

    cycles: list
    roots: np.ndarray
    parent: np.ndarray
    parent_edge: np.ndarray
    parent_dir: np.ndarray
    order: np.ndarray
    in_tree: np.ndarray


@dataclass
class DualNetwork:
    """Flow network whose nodes are the cycles of a graph.

    Attributes
    ----------
    supply : numpy.ndarray
        Net outflow of every node, with the last node being the ground.
    tail, head : numpy.ndarray
        Endpoints of every arc.
    arc_edge : numpy.ndarray
        Graph edge that each arc belongs to.
    ground : int
        Index of the ground node.
    """

    supply: np.ndarray
    tail: np.ndarray
    head: np.ndarray
    arc_edge: np.ndarray
    ground: int


@dataclass
class FlowInfo:
    """Bookkeeping for one flow solve.

    ``residues`` is the sum of the absolute supplies, which is the total
    absolute charge whenever the supplies are the residues of a cycle basis;
    the residues of a cycle basis always sum to zero, so the ground vertex is
    then supplied with zero and the two counts agree.

    ``relaxations`` counts the linear programs solved and ``proven`` records
    whether the exact solver finished its search; both are zero and true for
    the convex solver, which either reaches optimality or raises.

    ``seeded`` records whether the exact solver accepted the flow it was
    offered as a first incumbent; the convex solver cannot be offered one.
    """

    method: str
    nodes: int
    edges: int
    cycles: int
    residues: int
    augmentations: int
    objective: float
    relaxations: int = 0
    proven: bool = True
    seeded: bool = False


def graph_basis(nodes, tail, head):
    """Return a fundamental cycle basis of an oriented graph.

    A breadth-first forest is grown first; every remaining edge closes exactly
    one loop, the one that runs from one of its ends up to the common ancestor
    and back down to the other end.  Those loops are independent and generate
    the whole cycle space, so they are a basis.

    Parameters
    ----------
    nodes : int
        Number of vertices, numbered ``0 .. nodes - 1``.
    tail, head : array_like of int
        Endpoints of every edge, oriented from ``tail`` to ``head``.

    Returns
    -------
    CycleBasis
        The basis, the forest and the connectivity information.

    Notes
    -----
    The signs follow the walk of each loop: an edge traversed in its own
    direction contributes ``+1``, an edge traversed backwards ``-1``.
    """
    tail = np.asarray(tail, dtype=np.int64)
    head = np.asarray(head, dtype=np.int64)
    if tail.shape != head.shape:
        raise ValueError("tail and head must have the same shape")
    if tail.size and (tail.min() < 0 or head.min() < 0
                      or tail.max() >= nodes or head.max() >= nodes):
        raise ValueError("edge endpoint out of range")
    nedges = int(tail.size)
    adjacency = [[] for _ in range(nodes)]
    for edge in range(nedges):
        adjacency[tail[edge]].append((edge, 1))
        adjacency[head[edge]].append((edge, -1))

    parent = np.full(nodes, -1, dtype=np.int64)
    parent_edge = np.full(nodes, -1, dtype=np.int64)
    parent_dir = np.zeros(nodes, dtype=np.int64)
    in_tree = np.zeros(nedges, dtype=bool)
    seen = np.zeros(nodes, dtype=bool)
    order = []
    roots = []
    for start in range(nodes):
        if seen[start]:
            continue
        seen[start] = True
        roots.append(start)
        queue = deque([start])
        while queue:
            vertex = queue.popleft()
            order.append(vertex)
            for edge, direction in adjacency[vertex]:
                other = head[edge] if direction > 0 else tail[edge]
                if seen[other]:
                    continue
                seen[other] = True
                parent[other] = vertex
                parent_edge[other] = edge
                parent_dir[other] = direction
                in_tree[edge] = True
                queue.append(other)

    depth = np.zeros(nodes, dtype=np.int64)
    for vertex in order:
        if parent[vertex] >= 0:
            depth[vertex] = depth[parent[vertex]] + 1

    cycles = []
    for edge in np.flatnonzero(~in_tree):
        coefficients = {}
        left = int(tail[edge])
        right = int(head[edge])
        while depth[left] > depth[right]:
            coefficients[parent_edge[left]] = -parent_dir[left]
            left = parent[left]
        while depth[right] > depth[left]:
            coefficients[parent_edge[right]] = parent_dir[right]
            right = parent[right]
        while left != right:
            coefficients[parent_edge[left]] = -parent_dir[left]
            left = parent[left]
            coefficients[parent_edge[right]] = parent_dir[right]
            right = parent[right]
        # The loop closes along the non-tree edge against its own direction.
        coefficients[edge] = -1
        index = np.fromiter(coefficients.keys(), dtype=np.int64,
                            count=len(coefficients))
        sign = np.fromiter(coefficients.values(), dtype=np.int64,
                           count=len(coefficients))
        cycles.append((index, sign))
    return CycleBasis(cycles=cycles, roots=np.asarray(roots, dtype=np.int64),
                      parent=parent, parent_edge=parent_edge,
                      parent_dir=parent_dir,
                      order=np.asarray(order, dtype=np.int64),
                      in_tree=in_tree)


def residues_of_cycles(gradients, cycles):
    """Whole-turn circulation of a list of cycles.

    Parameters
    ----------
    gradients : array_like of float
        Wrapped step of every edge, in radians, in the orientation of the
        edge.
    cycles : sequence of tuple of numpy.ndarray
        Cycles as ``(edge indices, signs)``.

    Returns
    -------
    numpy.ndarray
        One residue per cycle, ``rint(sum(sign * gradient) / 2 pi)``.

    Notes
    -----
    The residue is the same whole number for any choice of loop through the
    same cells, because the circulation of the wrapped steps changes by a
    multiple of ``2 pi`` when a loop is deformed across a vertex.
    """
    gradients = np.asarray(gradients, dtype=np.float64)
    residues = np.empty(len(cycles), dtype=np.int64)
    for index, (edges, sign) in enumerate(cycles):
        circulation = float(np.dot(sign, gradients[edges]))
        residues[index] = int(np.rint(circulation / _TWO_PI))
    return residues


def cycle_residues(gradients, basis):
    """Whole-turn circulation of every cycle of a basis.

    Parameters
    ----------
    gradients : array_like of float
        Wrapped step of every edge, in radians, in the orientation of the
        edge.
    basis : CycleBasis
        Basis returned by :func:`graph_basis`.

    Returns
    -------
    numpy.ndarray
        One residue per cycle.
    """
    return residues_of_cycles(gradients, basis.cycles)


def build_dual_network(cycles, residues, nedges):
    """Build the dual flow network of a cycle basis.

    Parameters
    ----------
    cycles : sequence of tuple of numpy.ndarray
        Cycles as ``(edge indices, signs)``.
    residues : array_like of int
        Residue of every cycle.
    nedges : int
        Number of graph edges.

    Returns
    -------
    DualNetwork or None
        ``None`` when an edge appears in more than two cycles or in two cycles
        with the same sign, in which case the constraints are *not* those of a
        flow network and the integer program has to be used instead.

    Notes
    -----
    A cycle becomes a node and an edge becomes an arc joining the two cycles
    that share it, oriented from the cycle that walks the edge forwards to the
    one that walks it backwards.  Conservation of flow at a cycle node then
    reads ``sum(sign * k) = -residue``, which is exactly the loop closure
    condition.  An edge on only one cycle reaches the ground node, and a tree
    edge on no cycle at all carries no flow.
    """
    residues = np.asarray(residues, dtype=np.int64)
    incidence = [[] for _ in range(int(nedges))]
    for cycle, (edges, sign) in enumerate(cycles):
        for edge, value in zip(edges, sign):
            incidence[int(edge)].append((cycle, int(value)))
    ncycles = len(cycles)
    ground = ncycles
    supply = np.zeros(ncycles + 1, dtype=np.int64)
    supply[:ncycles] = -residues
    supply[ground] = -supply[:ncycles].sum()
    arc_edge = []
    arc_tail = []
    arc_head = []
    for edge, entries in enumerate(incidence):
        if not entries:
            continue
        if len(entries) == 1:
            cycle, value = entries[0]
            if value > 0:
                arc_tail.append(cycle)
                arc_head.append(ground)
            else:
                arc_tail.append(ground)
                arc_head.append(cycle)
        elif len(entries) == 2:
            (cycle_a, value_a), (cycle_b, value_b) = entries
            if value_a + value_b != 0:
                return None
            # The arc leaves the cycle that walks the edge forwards, exactly
            # as it does on an edge that borders a single cycle.
            if value_a > 0:
                arc_tail.append(cycle_a)
                arc_head.append(cycle_b)
            else:
                arc_tail.append(cycle_b)
                arc_head.append(cycle_a)
        else:
            return None
        arc_edge.append(edge)
    return DualNetwork(supply=supply,
                       tail=np.asarray(arc_tail, dtype=np.int64),
                       head=np.asarray(arc_head, dtype=np.int64),
                       arc_edge=np.asarray(arc_edge, dtype=np.int64),
                       ground=ground)


class ArcCost:
    """Convex cost of one integer arc flow.

    Subclasses describe the cost of a whole arc by three arrays of length
    ``nedges``.  :meth:`evaluate` is the cost itself, and the two marginals
    are the change of the cost when the flow of an arc is raised or lowered by
    one turn.  A convex integer cost is exactly one whose upward marginals are
    non-decreasing in the flow.  :meth:`cost_at` prices a few named arcs at
    several flows each, which is what a local search over the squares of a
    grid needs.
    """

    #: Whether the cost is convex, and therefore usable by :func:`convex_flow`.
    convex = True

    def evaluate(self, flow):
        """Return the cost of the given flow, one value per arc."""
        raise NotImplementedError

    def marginal_up(self, flow):
        """Return ``cost(flow + 1) - cost(flow)`` per arc."""
        raise NotImplementedError

    def marginal_down(self, flow):
        """Return ``cost(flow - 1) - cost(flow)`` per arc."""
        raise NotImplementedError

    def start(self):
        """Return the per-arc flow that minimises the arc cost alone."""
        raise NotImplementedError

    def cost_at(self, arc, values):
        """Return the cost of the arcs ``arc`` when their flow is ``values``.

        ``arc`` and ``values`` broadcast against each other, so a block of
        arcs can be priced in one call.  This is the form used by local search
        over the plaquettes of a grid, where the flow of a few arcs is tried
        at many values at once.
        """
        raise NotImplementedError


class LinearCost(ArcCost):
    """Weighted total variation, ``unit * abs(flow)``.

    This is the objective of Costantini's network formulation, as in
    :mod:`parvaneh.network_flow`: the flow on an edge *is* the number of whole
    turns added to that pixel step, so summing ``abs(flow)`` over the edges
    minimises the total weight of the jumps.

    Parameters
    ----------
    unit : array_like of float
        Nonnegative cost of one turn on every arc.
    """

    def __init__(self, unit):
        self.unit = np.asarray(unit, dtype=np.float64)

    def evaluate(self, flow):
        return self.unit * np.abs(flow)

    def marginal_up(self, flow):
        return np.where(flow >= 0, self.unit, -self.unit)

    def marginal_down(self, flow):
        return np.where(flow > 0, -self.unit, self.unit)

    def cost_at(self, arc, values):
        return self.unit[arc] * np.abs(values)

    def start(self):
        return np.zeros(self.unit.shape, dtype=np.int64)


class QuadraticCost(ArcCost):
    """Gaussian model, ``curvature * (flow - centre) ** 2``.

    This is the shape of the term inside SNAPHU's statistical costs: the
    corrected difference of an edge wants to equal a predicted difference, and
    the penalty grows quadratically with the distance from it.  The centre is
    a real number, so it is not an integer jump count by itself, which is
    precisely why the optimum cannot be found by rounding.

    Parameters
    ----------
    curvature : array_like of float
        Nonnegative curvature of every arc.
    centre : array_like of float
        Flow that would make the arc cost vanish.
    """

    def __init__(self, curvature, centre):
        self.curvature = np.asarray(curvature, dtype=np.float64)
        self.centre = np.asarray(centre, dtype=np.float64)
        if self.curvature.shape != self.centre.shape:
            raise ValueError("curvature and centre must have the same shape")

    def evaluate(self, flow):
        delta = flow - self.centre
        return self.curvature * delta * delta

    def marginal_up(self, flow):
        return self.curvature * (2.0 * (flow - self.centre) + 1.0)

    def marginal_down(self, flow):
        return -self.curvature * (2.0 * (flow - self.centre) - 1.0)

    def cost_at(self, arc, values):
        delta = values - self.centre[arc]
        return self.curvature[arc] * delta * delta

    def start(self):
        return np.rint(self.centre).astype(np.int64)


class ShelfCost(ArcCost):
    """Gaussian model with a flat cap, ``min(curvature * delta**2, cap)``.

    SNAPHU adds such a cap to its deformation model: beyond a largest
    plausible deformation the cost stops growing, because a large
    discontinuity is then explained by an atmospheric or deformation feature
    rather than by an unwrapping error.  Capping a quadratic destroys
    convexity, so this cost is reported as non-convex and
    :func:`convex_flow` refuses it.  It is the flat-cap special case of
    :class:`parvaneh.stat_costs.DeformationCost`, which also lets the cost
    grow again beyond the cap.

    Parameters
    ----------
    curvature, centre : array_like of float
        As in :class:`QuadraticCost`.
    cap : float
        Value at which the quadratic is clipped.
    """

    convex = False

    def __init__(self, curvature, centre, cap):
        self.curvature = np.asarray(curvature, dtype=np.float64)
        self.centre = np.asarray(centre, dtype=np.float64)
        self.cap = float(cap)
        if self.curvature.shape != self.centre.shape:
            raise ValueError("curvature and centre must have the same shape")

    def evaluate(self, flow):
        delta = flow - self.centre
        return np.minimum(self.curvature * delta * delta, self.cap)

    def marginal_up(self, flow):
        return (self.evaluate(flow + 1) - self.evaluate(flow))

    def marginal_down(self, flow):
        return (self.evaluate(flow - 1) - self.evaluate(flow))

    def cost_at(self, arc, values):
        delta = values - self.centre[arc]
        return np.minimum(self.curvature[arc] * delta * delta, self.cap)

    def start(self):
        return np.rint(self.centre).astype(np.int64)


def _residual_lists(nodes, tail, head):
    """Adjacency of the residual network, in vertex order.

    The two residual arcs of an edge are stored in one array of length
    ``2 * nedges``: slot ``e`` leaves the tail of the edge and raises its
    flow, slot ``e + nedges`` leaves the head and lowers it.
    """
    nedges = int(tail.size)
    source = np.concatenate([tail, head])
    order = np.argsort(source, kind="stable")
    counts = np.bincount(source, minlength=nodes)
    start = np.zeros(nodes + 1, dtype=np.int64)
    np.cumsum(counts, out=start[1:])
    return (start, order % nedges, order < nedges,
            np.concatenate([head, tail])[order])


def _dijkstra(nodes, source, adjacency, up, down, tail, head, potential):
    """Shortest residual reduced-cost distances from one vertex."""
    start, slot_edge, slot_forward, slot_other = adjacency
    dist = np.full(nodes, _INFINITY)
    pred = np.full(nodes, -1, dtype=np.int64)
    dist[source] = 0.0
    heap = [(0.0, source)]
    while heap:
        current, vertex = heappop(heap)
        if current > dist[vertex]:
            continue
        for slot in range(start[vertex], start[vertex + 1]):
            edge = slot_edge[slot]
            if slot_forward[slot]:
                reduced = (up[edge] + potential[tail[edge]]
                           - potential[head[edge]])
            else:
                reduced = (down[edge] + potential[head[edge]]
                           - potential[tail[edge]])
            if reduced < 0.0:
                # Only floating point noise can make a reduced cost negative.
                reduced = 0.0
            other = slot_other[slot]
            candidate = current + reduced
            if candidate < dist[other]:
                dist[other] = candidate
                pred[other] = slot
                heappush(heap, (candidate, other))
    return dist, pred


def convex_flow(nodes, tail, head, supply, cost, *, max_iter=None,
                return_info=False, method="flow"):
    """Minimise a separable convex cost over integer arc flows.

    This is the successive shortest augmenting path method of Ahuja,
    Magnanti and Orlin (1993, section 14.5) for convex cost flows.  Arcs are
    unbounded in both directions, so the algorithm never has to worry about
    capacities; it only has to route the excesses of the unbalanced vertices.
    Convexity enters through the marginal costs: decreasing an arc is the
    negative of the last upward marginal, so the residual network is the
    ordinary one with ``cost(flow + 1) - cost(flow)`` in one direction and
    ``cost(flow - 1) - cost(flow)`` in the other.

    Parameters
    ----------
    nodes : int
        Number of vertices.
    tail, head : array_like of int
        Endpoints of every arc, oriented from ``tail`` to ``head``.
    supply : array_like of int
        Net outflow of every vertex.  It must sum to zero, otherwise no flow
        can balance it.
    cost : ArcCost
        Convex cost of every arc, with one value per arc.
    max_iter : int, optional
        Safety limit on the number of augmentations.  The default is
        generous: the method needs at most one augmentation per unbalanced
        vertex.
    return_info : bool, optional
        Also return a :class:`FlowInfo`.

    Returns
    -------
    numpy.ndarray
        Signed flow of every arc.
    FlowInfo
        Only when ``return_info`` is true.

    Raises
    ------
    ValueError
        If the supplies do not sum to zero or the cost is not convex.
    RuntimeError
        If the augmentation limit is reached, which means the solver did not
        balance the supplies.
    """
    tail = np.asarray(tail, dtype=np.int64)
    head = np.asarray(head, dtype=np.int64)
    supply = np.asarray(supply, dtype=np.int64)
    if supply.shape != (nodes,):
        raise ValueError("supply must have one entry per vertex")
    if supply.sum() != 0:
        raise ValueError("supply must sum to zero")
    if not cost.convex:
        raise ValueError(
            "convex_flow needs a convex cost; the shelf model has to be "
            "solved by a warm-started scheme such as parvaneh.stat_costs")
    if tail.shape != head.shape:
        raise ValueError("tail and head must have the same shape")

    flow = np.asarray(cost.start(), dtype=np.int64)
    up = np.asarray(cost.marginal_up(flow), dtype=np.float64)
    down = np.asarray(cost.marginal_down(flow), dtype=np.float64)
    adjacency = _residual_lists(nodes, tail, head)
    # The initial flow minimises each arc on its own, so no residual arc has a
    # negative reduced cost and zero potentials are already feasible ones.
    potential = np.zeros(nodes, dtype=np.float64)

    # The cheapest flow of an arc taken on its own need not balance the
    # vertices, so what the augmentations have to move is the supply less the
    # imbalance the starting flow already carries.  Without this correction the
    # loop stops as soon as ``remain`` is drained and returns a flow whose
    # divergence is ``supply`` plus the imbalance of the start, which violates
    # the constraints the caller handed in.
    balance = np.bincount(tail, weights=flow, minlength=nodes)
    balance -= np.bincount(head, weights=flow, minlength=nodes)
    remain = supply - np.rint(balance).astype(np.int64)
    augmentations = 0
    if max_iter is None:
        max_iter = int(10 * nodes + 10 * np.abs(supply).sum() + 100)
    while True:
        sources = np.flatnonzero(remain > 0)
        if sources.size == 0:
            break
        if augmentations >= max_iter:
            raise RuntimeError(
                "the flow solver did not balance the supplies within "
                "%d augmentations" % max_iter)
        source = int(sources[0])
        dist, pred = _dijkstra(nodes, source, adjacency, up, down, tail, head,
                               potential)
        candidates = np.flatnonzero((remain < 0) & np.isfinite(dist))
        if candidates.size == 0:
            raise RuntimeError(
                "no vertex can absorb the excess of vertex %d; the supplies "
                "are not reachable from each other" % source)
        sink = int(candidates[np.argmin(dist[candidates])])
        delta = int(min(remain[source], -remain[sink]))
        vertex = sink
        while vertex != source:
            slot = int(pred[vertex])
            edge = int(adjacency[1][slot])
            if adjacency[2][slot]:
                flow[edge] += delta
                vertex = int(tail[edge])
            else:
                flow[edge] -= delta
                vertex = int(head[edge])
        # Every marginal is recomputed, not only those of the augmenting path:
        # a cost can be defined over the whole flow at once, and asking such a
        # cost about a subset of arcs silently gives the wrong answer.
        up = np.asarray(cost.marginal_up(flow), dtype=np.float64)
        down = np.asarray(cost.marginal_down(flow), dtype=np.float64)
        remain[source] -= delta
        remain[sink] += delta
        finite = np.isfinite(dist)
        if finite.all():
            potential += dist
        else:
            shift = float(dist[finite].max()) if finite.any() else 0.0
            potential += np.where(finite, dist, shift)
        augmentations += 1

    objective = float(np.sum(cost.evaluate(flow)))
    if not return_info:
        return flow
    info = FlowInfo(method=method, nodes=int(nodes), edges=int(tail.size),
                    cycles=0, residues=int(np.abs(supply).sum()),
                    augmentations=augmentations, objective=objective)
    return flow, info


def _cycle_matrix(cycles, nedges):
    """Sparse ``[A, -A]`` matrix of the cycle constraints.

    The flow of edge ``e`` is split into the difference of two nonnegative
    variables, ``k_e = p_e - n_e``, which turns the absolute value of the
    objective into a linear expression.  Column ``e`` holds the cycle
    coefficients of ``p_e`` and column ``nedges + e`` their negatives, so that
    ``matrix @ [p, n] = A @ k``.
    """
    rows = []
    cols = []
    values = []
    for cycle, (edges, sign) in enumerate(cycles):
        for edge, value in zip(edges, sign):
            rows.append(cycle)
            cols.append(int(edge))
            values.append(float(value))
            rows.append(cycle)
            cols.append(nedges + int(edge))
            values.append(-float(value))
    return csc_matrix((values, (rows, cols)), shape=(len(cycles), 2 * nedges))


def _bound_rows(lower, upper, nedges):
    """Inequality rows that impose ``lower <= p - n <= upper``."""
    rows = []
    cols = []
    values = []
    rhs = []
    index = 0
    for edge in range(nedges):
        if lower[edge] > -_INFINITY:
            rows.extend([index, index])
            cols.extend([edge, nedges + edge])
            values.extend([-1.0, 1.0])
            rhs.append(-lower[edge])
            index += 1
        if upper[edge] < _INFINITY:
            rows.extend([index, index])
            cols.extend([edge, nedges + edge])
            values.extend([1.0, -1.0])
            rhs.append(upper[edge])
            index += 1
    if index == 0:
        return None, None
    matrix = csc_matrix((values, (rows, cols)), shape=(index, 2 * nedges))
    return matrix, np.asarray(rhs, dtype=np.float64)


def _closes_cycles(matrix, residues, flow):
    """Whether an integer flow satisfies every cycle constraint exactly."""
    split = np.concatenate([np.maximum(flow, 0), np.maximum(-flow, 0)])
    residual = np.rint(np.asarray(matrix.dot(split)).ravel()) + residues
    return bool(np.all(residual == 0))


def _lp_relaxation(matrix, objective, residues, lower, upper, nedges):
    """Solve one linear relaxation, or return ``None`` if it is infeasible.

    HiGHS is the simplex implementation of ``scipy.optimize.linprog``; the
    branching constraints enter as two extra inequalities per bounded edge.
    """
    a_ub, b_ub = _bound_rows(lower, upper, nedges)
    result = linprog(objective, A_ub=a_ub, b_ub=b_ub, A_eq=matrix,
                     b_eq=-residues, bounds=(0.0, None), method="highs")
    if result.status == 2:
        return None
    if not result.success:
        raise RuntimeError("the cycle relaxation failed: %s" % result.message)
    flow = result.x[:nedges] - result.x[nedges:]
    return float(result.fun), flow


def integer_flow(cycles, residues, nedges, unit, *, max_nodes=None,
                 initial=None, return_info=False):
    """Solve the cycle constraints exactly by branch and bound.

    The objective is the weighted total variation ``sum(unit * abs(k))``.
    Every linear relaxation is solved by HiGHS, and integrality is imposed by
    a depth-first branch and bound over the fractional jumps.  The branching
    is done here rather than inside ``linprog`` because the ``integrality``
    argument of ``linprog`` only exists from SciPy 1.9, and Parvaneh still
    supports 1.8.

    The search needs an integer solution to start pruning with.  ``initial`` is
    verified before it is accepted; a caller that solves a fundamental basis
    can always supply one, because in that basis each cycle is determined by a
    chord whose own jump fixes the cycle by itself.

    Parameters
    ----------
    cycles : sequence of tuple of numpy.ndarray
        Cycles as ``(edge indices, signs)``.
    residues : array_like of int
        Residue of every cycle.
    nedges : int
        Number of graph edges.
    unit : array_like of float
        Nonnegative cost of one turn on every edge.
    max_nodes : int, optional
        Limit on the number of linear relaxations.  When the search stops early
        the best integer flow found so far is returned and
        ``FlowInfo.proven`` is false.  The default grows slowly with the number
        of edges so that a hard volume cannot hang the caller.
    initial : array_like of int, optional
        A feasible integer flow used as the first incumbent.
    return_info : bool, optional
        Also return a :class:`FlowInfo`.

    Returns
    -------
    numpy.ndarray
        Whole-turn jump of every edge.
    FlowInfo
        Only when ``return_info`` is true.

    Raises
    ------
    RuntimeError
        If no integer flow was found within the node limit, which means even
        the first branch needs more work than the caller allowed.
    """
    unit = np.asarray(unit, dtype=np.float64)
    residues = np.asarray(residues, dtype=np.int64)
    ncycles = len(cycles)
    if max_nodes is None:
        max_nodes = int(min(20000, 500 + 20 * nedges))
    constrained = np.zeros(nedges, dtype=bool)
    for edges, _ in cycles:
        constrained[edges] = True
    if ncycles == 0:
        flow = np.zeros(nedges, dtype=np.int64)
        best_cost = 0.0
        explored = 0
        proven = True
        seeded = False
    else:
        matrix = _cycle_matrix(cycles, nedges)
        objective = np.concatenate([unit, unit])
        best_cost = _INFINITY
        best = None
        seeded = False
        if initial is not None and len(np.asarray(initial).ravel()) == nedges:
            candidate = np.asarray(np.rint(initial), dtype=np.int64)
            if _closes_cycles(matrix, residues, candidate):
                best = candidate
                best_cost = float(np.sum(unit * np.abs(candidate)))
                seeded = True
        lower = np.full(nedges, -_INFINITY)
        upper = np.full(nedges, _INFINITY)
        stack = [(lower, upper)]
        explored = 0
        proven = True
        while stack:
            if explored >= max_nodes:
                proven = False
                break
            lower, upper = stack.pop()
            explored += 1
            relaxed = _lp_relaxation(matrix, objective, residues, lower, upper,
                                     nedges)
            if relaxed is None:
                continue
            value, flow = relaxed
            if value >= best_cost - 1e-6:
                continue
            scale = np.abs(flow - np.rint(flow))
            scale[~constrained] = 0.0
            worst = int(np.argmax(scale))
            if scale[worst] <= 1e-7:
                candidate = np.rint(flow).astype(np.int64)
                if _closes_cycles(matrix, residues, candidate):
                    best = candidate
                    best_cost = value
                continue
            floor = int(np.floor(flow[worst]))
            left_lower = lower.copy()
            left_upper = upper.copy()
            left_upper[worst] = float(floor)
            right_lower = lower.copy()
            right_upper = upper.copy()
            right_lower[worst] = float(floor + 1)
            stack.append((left_lower, left_upper))
            stack.append((right_lower, right_upper))
        if best is None:
            raise RuntimeError(
                "no integer flow satisfying the cycle constraints was found "
                "within %d linear relaxations" % max_nodes)
        flow = best
    # An edge that closes no loop never pays for a jump, so it stays put.
    flow = np.where(constrained, flow, 0)
    if not return_info:
        return flow
    info = FlowInfo(method="ilp", nodes=ncycles, edges=int(nedges),
                    cycles=ncycles, residues=int(np.abs(residues).sum()),
                    augmentations=0,
                    objective=float(np.sum(unit * np.abs(flow))),
                    relaxations=explored, proven=proven, seeded=seeded)
    return flow, info


def _basis_incumbent(basis, residues, nedges):
    """Trivial integer flow of a fundamental basis, used as a first incumbent.

    Every tree edge is left alone and each chord carries the residue of the
    loop it closes.  The chord enters its own loop with coefficient ``-1``, so
    that single jump satisfies the loop exactly, and the loops are then
    satisfied one by one: the guess is always feasible for a basis.  The exact
    solver still verifies it before trusting it.

    Returns ``None`` when the cycle set did not come from a basis.
    """
    if basis is None or not basis.cycles:
        return None
    chord = np.flatnonzero(~basis.in_tree)
    if chord.size != len(basis.cycles):
        return None
    flow = np.zeros(nedges, dtype=np.int64)
    flow[chord] = residues
    return flow


def curl_flow(nodes, tail, head, gradients, *, cost=None, cycles=None,
              method="auto", max_iter=None, max_nodes=None, initial=None,
              return_info=False):
    """Choose integer edge jumps that make every loop of a graph close.

    Parameters
    ----------
    nodes : int
        Number of vertices.
    tail, head : array_like of int
        Endpoints of every edge, oriented from ``tail`` to ``head``.
    gradients : array_like of float
        Wrapped step of every edge, in radians.
    cost : ArcCost or array_like of float, optional
        Cost of one turn on each edge.  An array is interpreted as the
        nonnegative weight of a :class:`LinearCost`, which is the default and
        is the objective used by all the wrapped-cost methods.
    cycles : sequence, optional
        Cycles to use instead of a fundamental basis.  Grid solvers pass their
        plaquettes, whose incidence structure is a network matrix.
    method : {"auto", "flow", "ilp"}, optional
        ``"flow"`` requires the dual network, ``"ilp"`` forces branch and
        bound, and ``"auto"`` uses the dual network when it exists.  Convex
        costs other than the linear one are accepted only with ``"flow"``.
    max_iter : int, optional
        Augmentation limit of the flow solver.
    max_nodes : int, optional
        Relaxation limit of the exact solver.
    initial : array_like of int, optional
        Feasible flow offered to the exact solver as its first incumbent.  It
        is used only when the cycles are not a flow network; the dual-network
        solver ignores it and starts from zero.  A flow that does not close
        every loop is discarded, so passing an invalid guess costs time and
        never correctness.  The default is the flow supported on a
        fundamental cycle basis, whose jumps are read off the residues.
    return_info : bool, optional
        Also return a :class:`FlowInfo`.

    Returns
    -------
    numpy.ndarray
        Whole-turn jump of every edge.
    FlowInfo
        Only when ``return_info`` is true.

    Raises
    ------
    ValueError
        If an unknown ``method`` is given, or if a non-linear cost is used
        with a structure that is not a flow network.
    """
    if method not in ("auto", "flow", "ilp"):
        raise ValueError("method must be 'auto', 'flow' or 'ilp'")
    tail = np.asarray(tail, dtype=np.int64)
    head = np.asarray(head, dtype=np.int64)
    gradients = np.asarray(gradients, dtype=np.float64)
    if gradients.shape != tail.shape:
        raise ValueError("gradients must have one value per edge")
    basis = None
    if cycles is None:
        basis = graph_basis(nodes, tail, head)
        cycles = basis.cycles
    residues = residues_of_cycles(gradients, cycles)

    network = None
    if method in ("auto", "flow"):
        network = build_dual_network(cycles, residues, tail.size)
    if cost is None:
        cost = LinearCost(np.ones(tail.size, dtype=np.float64))
    elif not isinstance(cost, ArcCost):
        cost = LinearCost(cost)
    if network is None:
        if not isinstance(cost, LinearCost):
            raise ValueError(
                "a general cost network is not a flow network, so only the "
                "linear cost can be solved by branch and bound")
        if method == "flow":
            raise ValueError("these cycles do not form a flow network")
        if initial is None:
            initial = _basis_incumbent(basis, residues, tail.size)
        flow, info = integer_flow(cycles, residues, tail.size, cost.unit,
                                  max_nodes=max_nodes, initial=initial,
                                  return_info=True)
        info.nodes = int(nodes)
        info.cycles = len(cycles)
        info.residues = int(np.abs(residues).sum())
    else:
        arc_flow, info = convex_flow(network.ground + 1, network.tail,
                                     network.head, network.supply,
                                     _SubsetCost(cost, network.arc_edge,
                                                 tail.size),
                                     max_iter=max_iter, return_info=True)
        flow = np.zeros(tail.size, dtype=np.int64)
        flow[network.arc_edge] = arc_flow
        info.nodes = int(nodes)
        info.edges = int(tail.size)
        info.cycles = len(cycles)
        info.residues = int(np.abs(residues).sum())
    if not return_info:
        return flow
    return flow, info


class _SubsetCost(ArcCost):
    """An :class:`ArcCost` restricted to the arcs of a dual network.

    A dual network holds one arc per graph edge that lies on a cycle, so its
    arcs are numbered by network position while the parent cost is numbered by
    graph edge.  Every array is therefore scattered into the numbering of the
    parent before the parent is asked, and gathered back into the numbering of
    the network afterwards.  An edge that lies on no cycle at all is absent
    from the network, and the zero it is scattered with never reaches a
    marginal, because the parent is only ever asked about the arcs it holds.

    Parameters
    ----------
    parent : ArcCost
        Cost of one turn on every edge of the graph.
    arc_edge : array_like of int
        Graph edge of every arc of the network.
    nedges : int
        Number of edges of the graph.
    """

    def __init__(self, parent, arc_edge, nedges):
        self.parent = parent
        self.arc_edge = np.asarray(arc_edge, dtype=np.int64)
        self.nedges = int(nedges)
        self.convex = parent.convex

    def _scatter(self, values):
        """Lay network values out over the edges of the graph."""
        values = np.asarray(values)
        full = np.zeros(self.nedges, dtype=values.dtype)
        full[self.arc_edge] = values
        return full

    def evaluate(self, flow):
        return self.parent.evaluate(self._scatter(flow))[self.arc_edge]

    def marginal_up(self, flow):
        return self.parent.marginal_up(self._scatter(flow))[self.arc_edge]

    def marginal_down(self, flow):
        return self.parent.marginal_down(self._scatter(flow))[self.arc_edge]

    def cost_at(self, arc, values):
        return self.parent.cost_at(self.arc_edge[arc], values)

    def start(self):
        return self.parent.start()[self.arc_edge]


def integrate_gradients(nodes, tail, head, gradients, flow, basis=None,
                        anchor=None):
    """Turn edge jumps into vertex potentials.

    The corrected step of an edge is ``gradient + 2 pi * jump``, and these
    steps are consistent by construction, so a spanning forest is enough to
    reconstruct the field: the potential of a child is the potential of its
    parent plus the corrected step of the edge between them.  Each connected
    component gets its own anchor, because nothing fixes the offset of a
    component relative to the others.

    Parameters
    ----------
    nodes : int
        Number of vertices.
    tail, head : array_like of int
        Endpoints of every edge, oriented from ``tail`` to ``head``.
    gradients : array_like of float
        Wrapped step of every edge, in radians.
    flow : array_like of int
        Whole-turn jump of every edge, from :func:`curl_flow`.
    basis : CycleBasis, optional
        Result of :func:`graph_basis`.  It is recomputed when omitted.
    anchor : int, optional
        Vertex whose potential is zero.  Without it every component is
        anchored at the root found by the breadth-first search.

    Returns
    -------
    numpy.ndarray
        Vertex potentials in radians, zero at the root of every component, so
        that within one component the difference of two potentials is the
        unwrapped step between them.  Vertices that no edge touches are
        components of their own and come back as zero.
    """
    tail = np.asarray(tail, dtype=np.int64)
    head = np.asarray(head, dtype=np.int64)
    gradients = np.asarray(gradients, dtype=np.float64)
    flow = np.asarray(flow, dtype=np.int64)
    if basis is None:
        basis = graph_basis(nodes, tail, head)
    corrected = gradients + _TWO_PI * flow
    potential = np.full(nodes, np.nan)
    for vertex in basis.roots:
        potential[vertex] = 0.0
    for vertex in basis.order:
        edge = basis.parent_edge[vertex]
        if edge < 0:
            continue
        step = corrected[edge]
        if basis.parent_dir[vertex] < 0:
            step = -step
        potential[vertex] = potential[basis.parent[vertex]] + step
    if anchor is not None:
        anchor = int(anchor)
        if potential[anchor] == potential[anchor]:
            potential -= potential[anchor]
    return potential
