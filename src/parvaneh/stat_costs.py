"""Statistical cost functions for two dimensional phase unwrapping.

The wrapped cost methods of this package so far price a jump in proportion to
its size and give every edge the same price.  SNAPHU prices an edge instead
by how well a jump explains the measurement there, using a model of the phase
noise of the interferogram.  Two ingredients go into that model.  First, the
wrapped step of an edge is compared with the average wrapped step of a small
neighbourhood around it, and the difference is the residual that a jump is
meant to cancel.  Second, the price per turn of that residual is set by the
coherence of the two samples: a noisy edge is cheap to jump, a clean edge is
expensive.  In a deformed area the price curve is then flattened and allowed
to rise again far from the residual, because a large residual there is more
plausibly a real deformation or an atmospheric delay than an unwrapping
mistake.

The cost of one arc
-------------------

Let ``k`` be the whole number of turns added to the step of an arc, ``d`` the
wrapped step of that arc measured in turns and ``a`` the average of ``d``
over a small box centred on the arc.  Write ``z = abs(k + d - h * a)``, the
size of the residual that the jump leaves behind, where ``h`` is the share of
the average that the model keeps.  The model of Chen and Zebker (2001) gives
the arc the quadratic cost

    c * z ** 2

where the curvature ``c`` is ``w / s ** 2``, with ``w`` the weight of the arc
and ``s ** 2`` the variance of the residual of an edge that carries noise
alone.  That variance is the variance of the wrapped difference of two
samples with coherence ``rho``, plus a floor ``s_c ** 2`` that stands for the
error of the coherence estimate itself,

    s ** 2 = (1 / 6) * (1 - rho) ** e + s_c ** 2.

The exponent ``e`` grows with the number of independent looks of the
interferogram; at one look the first term is the exact wrapped-difference
variance ``(1 - rho) / 6`` of a pair of complex samples.  Edges whose
coherence falls below a threshold ``t`` are taken to be pure noise, which is
done by setting ``rho = 0`` there: this both raises ``s ** 2`` and removes
the cap described next, and is the "statistical segmentation" of Chen and
Zebker (2002).

The smooth mode keeps the whole average, ``h = 1``, in both directions.  The
deformation mode keeps only half of it along the range direction, ``h = 1/2``,
and a quarter for a range edge that is noise alone, ``h = 1/4``; the azimuth
direction is left as in the smooth mode.  The reason is that a range step may
carry a real deformation, which the average of its neighbourhood does not
predict, so the cost of a range edge is centred nearer the raw measurement
there.  SNAPHU builds its two directions separately and differs between them
in exactly this way.

In the deformation mode the quadratic is capped at a flat level ``l`` and
allowed to rise again beyond a residual ``m``,

    min(c * z ** 2, l)                       z <= m
    c * (z - m) ** 2 / f + l                 z >  m

with ``l = w * log(1 / q)`` for a fraction ``q`` close to one and ``f`` the
falloff constant.  The reading is that a residual larger than ``m`` turns is
too large to be explained by noise, so it is priced at the level of a real
deformation, and beyond that it is priced only through the falloff.  The cap
is used for an arc only when it is reachable, that is when ``m`` is not
smaller than ``sqrt(l / c)``; otherwise the arc keeps the plain quadratic
cost.

Convexity
---------

The plain quadratic is convex, so the flow that minimises it is found exactly
by the dual-network solver of :mod:`parvaneh.graph_flow`, and the answer is
reported as proven.  The cap destroys convexity: the slope of the cost drops
to zero at ``z = m``, so the cost has a kink there and the flow problem stops
being a convex program.  This module therefore solves the capped model by a
convex warm start followed by exact moves on the vertices of the grid, and
reports the result as a local optimum rather than as a proven global one.
Asking for ``costmode="smooth"``, or passing ``shelf=False``, keeps the
quadratic model, which is solved exactly.

Relation to SNAPHU
------------------

Every cost here is SNAPHU's cost divided by its scale factor, and SNAPHU's
own reported cost is that fixed multiple of the numbers reported here.  The
scale and the cycle quantisation of SNAPHU cancel in the model above, so the
two formulations rank the candidate flows in the same order and solve the
same problem in exact arithmetic.  See the module notes further down for the
list of deliberate deviations, which are all arithmetic, or belong to the
model of the deformation mode, rather than structural.

Two dimensions
--------------

The model is defined on a two dimensional grid, as in SNAPHU.  In two
dimensions every edge lies in one or two squares, so the constraint matrix is
a network matrix and the dual-network solvers apply.  On a three dimensional
grid an edge lies in four squares, the dual network does not exist, and a
non-linear cost cannot be solved by this module; use
:func:`parvaneh.flow_nd.flow_nd_unwrap` there.

Example
-------
>>> import numpy as np
>>> from parvaneh.stat_costs import stat_cost_unwrap   # doctest: +SKIP
>>> phase = np.zeros((8, 8), dtype=np.float64)         # doctest: +SKIP
>>> coherence = np.full((8, 8), 0.9)                   # doctest: +SKIP
>>> field = stat_cost_unwrap(phase, coherence)         # doctest: +SKIP

References
----------
C. W. Chen and H. A. Zebker, "Two-dimensional phase unwrapping with use of
statistical models for cost functions in nonlinear optimization," Journal of
the Optical Society of America A, vol. 18, no. 2, pp. 338-351, 2001.

C. W. Chen and H. A. Zebker, "Network approaches to two-dimensional phase
unwrapping: intractability and two new algorithms," Journal of the Optical
Society of America A, vol. 17, no. 3, pp. 401-414, 2000.

C. W. Chen and H. A. Zebker, "Phase unwrapping for large SAR interferograms:
statistical segmentation and generalized network models," IEEE Transactions
on Geoscience and Remote Sensing, vol. 40, no. 8, pp. 1709-1719, 2002.

D. C. Ghiglia and M. D. Pritt, "Two-Dimensional Phase Unwrapping: Theory,
Algorithms, and Software," Wiley, 1998.

The cost functions and their constants are those of the SNAPHU program of
the same authors.  This module reimplements them from their published
description and shares no code with it.

Notes
-----
The deviations from SNAPHU are the following, and all of them are
arithmetic, except the last two, which concern the model of the deformation
mode.

* SNAPHU stores the offset, the variance and the flat level as scaled short
  integers and truncates them, and it quantises the slope of the cost
  through the number of cycles.  This module keeps exact real numbers.  The
  two agree wherever the truncation is small compared with a turn.
* SNAPHU omits the flat level of an arc whose variance is below a floor of
  one short unit, which only matters for weights far above the ones an
  interferogram produces.
* The falloff branch beyond ``m`` is a real division here, where SNAPHU
  truncates an integer division.
* The box average is taken with :func:`scipy.ndimage.uniform_filter` in
  ``"mirror"`` mode, which reproduces SNAPHU's reflect padding without
  repeating the edge sample.  SNAPHU exits when the box does not fit in the
  gradient array; this module raises ``ValueError`` instead.
* SNAPHU clips the cost of an arc to a maximum, which is not reproduced: the
  objective reported here is unbounded like the model above.
* SNAPHU initialises the flow from a minimum spanning tree, controls the
  search with a temperature schedule, and can tile a large interferogram.
  Neither the schedule nor the tiling is reproduced.  The warm start used
  here is the exact solution of the convex model, and the refinement is a
  sequence of exact moves on the grid vertices.
* An arc of weight zero is free here, as in SNAPHU, where it is assigned a
  cost that is identically zero.  A ring of usable samples around a masked
  hole can then enclose a loop that no surviving square spans; the flow
  solver still returns a valid field, but the loop is only fixed up to whole
  turns.
* The flat level of a range edge of the deformation mode is scaled by
  ``defoazdzfactor`` in SNAPHU, which is one by default and is not a
  parameter of this module.
* The deformation mode of SNAPHU also shifts the offset of a range edge by
  the step that its scattering model predicts, and adds the variance of that
  prediction to ``s ** 2``.  Both need the sensor geometry, the intensity
  image and the local terrain slopes, and neither is modelled here, so a
  range edge is priced by its coherence alone, as in the smooth mode.  Only
  the weaker share of the average, the cap and the lower coherence threshold
  of the deformation mode are kept.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter

from .core import _axis_slice, _wrap
from .flow_nd import (
    ALIGN_MODES,
    _edge_shapes,
    grid_cycles,
    grid_edge_index,
    grid_edges,
    integrate_edges,
)
from .graph_flow import (
    ArcCost,
    QuadraticCost,
    curl_flow,
    residues_of_cycles,
)
from .reliability import _axis_confidence, _check_phase, _confidence

_TWO_PI = 2.0 * np.pi


#: Cost functions offered by :func:`stat_cost_unwrap`.
COST_MODES = ("smooth", "defo")

#: Ways of starting the refinement of the capped model.
START_MODES = ("smooth", "zero")

_EPSILON = 1e-12
_MIN_WEIGHT = 0.0

#: Share of the box average kept in the residual of an edge, as
#: ``(coherent, noise only)`` for the azimuth direction (the first axis) and
#: the range direction (the last axis), for the smooth cost model.
_SMOOTH_SHARE = ((1.0, 0.5), (1.0, 0.5))

#: The same shares for the deformation cost model, whose range direction
#: keeps half of the average, and a quarter for an edge that is noise alone.
_DEFO_SHARE = ((1.0, 0.5), (0.5, 0.25))


@dataclass
class StatCostParams:
    """Constants of the statistical cost model.

    The defaults are those of SNAPHU, which documents each of them in its man
    page.  They describe the sensor rather than the scene, so the same values
    are reused for every interferogram of one stack.

    Attributes
    ----------
    ncorrlooks : float
        Number of independent looks of the coherence estimate.
    rhosconst1, rhosconst2 : float
        Constants of the coherence threshold ``c1 / looks + c2``.
    cstd1, cstd2, cstd3 : float
        Constants of the exponent of the variance.
    defothreshfactor : float
        Factor of the coherence threshold of the deformation mode.
    sigsqcorr : float
        Floor of the variance, for the error of the coherence estimate.
    defomax : float
        Largest residual, in turns, that noise can explain.  A value of zero
        disables the cap and leaves the quadratic model.
    defolayconst : float
        Fraction whose logarithm is the flat level per unit weight.
    layfalloffconst : float
        Divisor of the quadratic that rises beyond ``defomax``.
    kperpdpsi, kpardpsi : int
        Width of the box average of the wrapped step, across and along the
        direction of the difference.  Both must be odd.
    """

    ncorrlooks: float = 23.8
    rhosconst1: float = 1.3
    rhosconst2: float = 0.14
    cstd1: float = 0.4
    cstd2: float = 0.35
    cstd3: float = 0.06
    defothreshfactor: float = 1.2
    sigsqcorr: float = 0.05
    defomax: float = 1.2
    defolayconst: float = 0.9
    layfalloffconst: float = 2.0
    kperpdpsi: int = 7
    kpardpsi: int = 7

    def __post_init__(self):
        if self.ncorrlooks <= 0.0:
            raise ValueError("ncorrlooks must be positive")
        if self.layfalloffconst <= 0.0:
            raise ValueError("layfalloffconst must be positive")
        if self.sigsqcorr <= 0.0:
            raise ValueError("sigsqcorr must be positive")
        if not 0.0 < self.defolayconst < 1.0:
            raise ValueError("defolayconst must lie between 0 and 1")
        if self.defomax < 0.0:
            raise ValueError("defomax must not be negative")
        for name in ("kperpdpsi", "kpardpsi"):
            size = int(getattr(self, name))
            if size < 1 or size % 2 == 0:
                raise ValueError("%s must be a positive odd number" % (name,))

    @property
    def rho0(self):
        """Coherence threshold of the smooth mode."""
        return self.rhosconst1 / self.ncorrlooks + self.rhosconst2

    @property
    def defocorrthresh(self):
        """Coherence threshold of the deformation mode."""
        return self.defothreshfactor * self.rho0

    @property
    def rhopow(self):
        """Exponent of the coherence in the variance of the residual."""
        return (2.0 * self.cstd1 + self.cstd2 * math.log(self.ncorrlooks)
                + self.cstd3 * self.ncorrlooks)


@dataclass
class StatCostInfo:
    """Report of :func:`stat_cost_unwrap`.

    Attributes
    ----------
    backend : str
        Solver that produced the field.
    costmode : str
        Cost function that was used.
    shape : tuple of int
        Shape of the grid.
    pixels : int
        Samples with a nonzero confidence.
    nodes : int
        Samples of the grid.
    edges : int
        Arcs of the grid, one per pair of neighbouring samples.
    cycles : int
        Loops that the jump residual had to be distributed over.
    residues : int
        Loops with a nonzero residue before unwrapping.
    components : int
        Connected components of the confident region.
    shelf_edges : int
        Arcs whose cost was capped.
    masked_edges : int
        Arcs of weight zero, whose cost is zero.
    max_jump : int
        Largest number of turns added to a single arc.
    objective : float
        Summed cost of the returned flow.
    initial_objective : float
        Summed cost of the flow the refinement started from, which is the
        returned flow itself when the cost was convex and no sweep was run.
    sweeps : int
        Sweeps of the refinement that lowered the cost.
    moves : int
        Squares whose jump was changed by the refinement.
    proven : bool
        Whether the objective is known to be the smallest one.
    ncorrlooks : float
        Looks used in the noise model.
    rho0 : float
        Coherence threshold of the smooth mode.
    defocorrthresh : float
        Coherence threshold of the deformation mode.
    """

    backend: str
    costmode: str
    shape: tuple
    pixels: int
    nodes: int
    edges: int
    cycles: int
    residues: int
    components: int
    shelf_edges: int
    masked_edges: int
    max_jump: int
    objective: float
    initial_objective: float
    sweeps: int
    moves: int
    proven: bool
    ncorrlooks: float
    rho0: float
    defocorrthresh: float


@dataclass
class _CostModel:
    """Flat per-arc arrays of the cost model."""

    curvature: np.ndarray
    centre: np.ndarray
    shelf: np.ndarray
    level: np.ndarray
    weight: np.ndarray
    gradients: np.ndarray


class DeformationCost(ArcCost):
    """Capped quadratic cost of one arc.

    The cost of a jump ``k`` on an arc is ``curvature * z ** 2`` with
    ``z = abs(k - centre)``, clipped to ``level`` where the arc is capped,
    and rising again with the falloff beyond ``defomax`` turns:

        min(curvature * z ** 2, level)                     z <= defomax
        curvature * (z - defomax) ** 2 / falloff + level   z >  defomax

    An arc whose ``shelf`` flag is false keeps the plain quadratic.  The
    capped form is continuous and has a continuous slope at ``defomax``, but
    its slope drops to zero at the start of the shelf, so the whole cost is
    not convex and the class reports ``convex`` as false as soon as one arc
    is capped.

    Parameters
    ----------
    curvature, centre : array_like of float
        Price of one turn squared, and the jump that costs nothing, one value
        per arc.
    shelf : array_like of bool, optional
        Which arcs are capped.  The default is none of them.
    level : array_like of float, optional
        Flat level of every capped arc.  The default is zero.
    defomax : float, optional
        Residual, in turns, at which the cap is reached and the falloff
        starts.
    falloff : float, optional
        Divisor of the quadratic that rises beyond ``defomax``.
    """

    def __init__(self, curvature, centre, *, shelf=None, level=None,
                 defomax=1.2, falloff=1.0):
        curvature = np.ascontiguousarray(curvature, dtype=np.float64)
        centre = np.ascontiguousarray(centre, dtype=np.float64)
        if curvature.shape != centre.shape:
            raise ValueError("curvature and centre must have the same shape")
        if falloff <= 0.0:
            raise ValueError("falloff must be positive")
        if defomax < 0.0:
            raise ValueError("defomax must not be negative")
        if shelf is None:
            shelf = np.zeros(curvature.shape, dtype=bool)
        else:
            shelf = np.ascontiguousarray(shelf, dtype=bool)
            if shelf.shape != curvature.shape:
                raise ValueError("shelf must have one value per arc")
        if level is None:
            level = np.zeros(curvature.shape, dtype=np.float64)
        else:
            level = np.ascontiguousarray(level, dtype=np.float64)
            if level.shape != curvature.shape:
                raise ValueError("level must have one value per arc")
        self.curvature = curvature
        self.centre = centre
        self.shelf = shelf
        self.level = level
        self.defomax = float(defomax)
        self.falloff = float(falloff)
        self.convex = not bool(np.any(shelf))

    def cost_at(self, arc, values):
        """Cost of ``values`` turns on the arcs ``arc``.

        Parameters
        ----------
        arc : array_like of int
            Arc of every value.
        values : array_like of int
            Whole turns added to the arc.

        Returns
        -------
        numpy.ndarray of float
            Cost of every value.
        """
        arc = np.asarray(arc, dtype=np.int64)
        size = np.abs(np.asarray(values, dtype=np.float64) - self.centre[arc])
        quadratic = self.curvature[arc] * size * size
        if not np.any(self.shelf):
            return quadratic
        level = self.level[arc]
        beyond = (self.curvature[arc]
                  * (size - self.defomax) ** 2 / self.falloff + level)
        # The flat shelf runs from the point where the quadratic reaches
        # ``level`` up to ``defomax``; the falloff takes over only past
        # ``defomax``.  Testing ``quadratic > level`` alone would apply the
        # falloff inside the shelf, where it falls with distance instead of
        # staying flat, and there would be a spurious minimum at ``defomax``.
        capped = np.where(size > self.defomax, beyond,
                          np.minimum(quadratic, level))
        return np.where(self.shelf[arc], capped, quadratic)

    def evaluate(self, flow):
        """Cost of a whole flow."""
        flow = np.asarray(flow, dtype=np.int64)
        return self.cost_at(np.arange(flow.size), flow)

    def marginal_up(self, flow):
        """Cost that one more turn on each arc would add."""
        flow = np.asarray(flow, dtype=np.int64)
        arc = np.arange(flow.size)
        return self.cost_at(arc, flow + 1) - self.cost_at(arc, flow)

    def marginal_down(self, flow):
        """Cost that one fewer turn on each arc would add."""
        flow = np.asarray(flow, dtype=np.int64)
        arc = np.arange(flow.size)
        return self.cost_at(arc, flow - 1) - self.cost_at(arc, flow)

    def start(self):
        """Round the centre to the nearest whole number of turns."""
        return np.rint(self.centre).astype(np.int64)


def _build_model(phase, confidence, coherence, params, threshold, flatten,
                 shelf, costmode):
    """Build the flat per-arc arrays of the cost model.

    Parameters
    ----------
    phase : numpy.ndarray
        Wrapped phase in radians, already flattened if ``flatten`` is given.
    confidence : numpy.ndarray
        Per-sample weight.
    coherence : numpy.ndarray
        Per-sample coherence.
    params : StatCostParams
        Constants of the model.
    threshold : float
        Coherence below which an arc is taken to carry noise alone.
    flatten : numpy.ndarray or None
        Estimate that was subtracted from the phase.
    shelf : bool
        Whether the cap may be used at all.
    costmode : {"smooth", "defo"}
        Cost model, which fixes the share of the average kept in the
        residual of an edge.

    Returns
    -------
    _CostModel
    """
    floor = math.log(1.0 / params.defolayconst)
    shares = _DEFO_SHARE if costmode == "defo" else _SMOOTH_SHARE
    shapes = _edge_shapes(phase.shape)
    curvature, centre, flags, level, weight, gradients = [], [], [], [], [], []
    for axis in range(phase.ndim):
        shape = shapes[axis]
        kernel = [params.kperpdpsi] * phase.ndim
        kernel[axis] = params.kpardpsi
        for other, size in enumerate(kernel):
            if (size - 1) // 2 > shape[other]:
                raise ValueError(
                    "the averaging box does not fit the gradient array of "
                    "shape %s; lower kpardpsi or kperpdpsi" % (shape,))
        step = _wrap(np.diff(phase, axis=axis)) / _TWO_PI
        average = uniform_filter(np.ascontiguousarray(step), size=kernel,
                                 mode="mirror")
        arc_coherence = 0.5 * (
            _axis_slice(coherence, axis, slice(0, -1))
            + _axis_slice(coherence, axis, slice(1, None)))
        noise_only = arc_coherence < threshold
        rho = np.where(noise_only, 0.0, arc_coherence)
        coherent_share, noisy_share = shares[axis == phase.ndim - 1]
        offset = np.where(noise_only, step - noisy_share * average,
                          step - coherent_share * average)
        if flatten is not None:
            offset = offset + np.diff(flatten, axis=axis) / _TWO_PI
        variance = ((1.0 / 6.0) * (1.0 - rho) ** params.rhopow
                    + params.sigsqcorr)
        arc_weight = _axis_confidence(confidence, axis).ravel()
        curvature.append((arc_weight / variance.ravel()))
        centre.append((-offset).ravel())
        enable = ((arc_weight > _MIN_WEIGHT)
                  & (params.defomax >= np.sqrt(floor * variance.ravel())))
        if not shelf:
            enable = np.zeros_like(enable)
        flags.append(enable)
        level.append(np.where(enable, arc_weight * floor, 0.0))
        weight.append(arc_weight)
        gradients.append((_TWO_PI * step).ravel())
    return _CostModel(
        curvature=np.concatenate(curvature),
        centre=np.concatenate(centre),
        shelf=np.concatenate(flags),
        level=np.concatenate(level),
        weight=np.concatenate(weight),
        gradients=np.concatenate(gradients))


def _star_moves(nodes, tail, head):
    """Arc and sign of the moves that raise one vertex at a time.

    Adding one turn to every arc that enters a vertex and taking one turn
    away from every arc that leaves it changes the jumps by the difference of
    a step of the integer potential between the two ends of the arc.  A move
    of that form is a gradient, so it leaves the closure of every loop alone:
    it can never change the residue of a loop and a flow stays feasible.
    Every two flows that close the same loops differ by a sum of such moves,
    which makes them the natural moves of a descent.

    Parameters
    ----------
    nodes : int
        Number of vertices.
    tail, head : array_like of int
        Endpoints of every arc of the graph.

    Returns
    -------
    numpy.ndarray
        Arc of every move, one row per vertex and one column per incident
        arc, padded with arc zero where a vertex has fewer neighbours.
    numpy.ndarray
        Sign of every move: ``+1`` when the arc enters the vertex, ``-1``
        when it leaves it and ``0`` on the padding.  A padded entry carries
        sign zero, so the jump it names is constant along the move and it can
        neither change the objective nor the best move.
    """
    tail = np.asarray(tail, dtype=np.int64)
    head = np.asarray(head, dtype=np.int64)
    nedges = int(tail.size)
    incident = np.concatenate([tail, head])
    counts = np.bincount(incident, minlength=nodes)
    width = int(counts.max()) if counts.size else 0
    order = np.argsort(incident, kind="stable")
    start = np.zeros(nodes + 1, dtype=np.int64)
    np.cumsum(counts, out=start[1:])
    vertex = incident[order]
    column = np.arange(incident.size) - start[vertex]
    moves = np.zeros((nodes, width), dtype=np.int64)
    signs = np.zeros((nodes, width), dtype=np.int64)
    moves[vertex, column] = order % nedges
    signs[vertex, column] = np.where(order < nedges, -1, 1)
    return moves, signs


def _colour_vertex_blocks(moves, signs, tail, head):
    """Split the vertices into blocks whose moves share no arc.

    Two stars share an arc only when their vertices are the two ends of that
    arc, so a two colouring of the vertices in which the ends of every arc
    have different colours puts the stars of one colour on arc disjoint arcs.
    The colouring is grown from the arcs themselves and never from the
    numbering of the vertices, and a graph that cannot be two coloured is
    reported so that the caller falls back to one vertex at a time, which is
    slower but never wrong.

    Parameters
    ----------
    moves, signs : numpy.ndarray
        Moves of every vertex, as returned by :func:`_star_moves`.
    tail, head : array_like of int
        Endpoints of every arc of the graph.

    Returns
    -------
    list of numpy.ndarray or None
        Index blocks, or ``None`` when the vertices cannot be two coloured.
    """
    tail = np.asarray(tail, dtype=np.int64)
    head = np.asarray(head, dtype=np.int64)
    colour = np.full(moves.shape[0], -1, dtype=np.int64)
    for seed in range(colour.size):
        if colour[seed] >= 0:
            continue
        colour[seed] = 0
        stack = [seed]
        while stack:
            vertex = stack.pop()
            want = 1 - int(colour[vertex])
            for arc, sign in zip(moves[vertex], signs[vertex]):
                if sign == 0:
                    continue
                other = int(tail[arc]) if sign > 0 else int(head[arc])
                if colour[other] < 0:
                    colour[other] = want
                    stack.append(other)
                elif colour[other] != want:
                    return None
    return [np.flatnonzero(colour == 0), np.flatnonzero(colour == 1)]


def _lower_block(flow, cost, curvature, centre, edges, signs, group, steps):
    """Lower the exact cost on every move of one block.

    Every move of a block touches its own arcs, so the moves do not
    interfere, and the whole block is then accepted or refused as one: a
    refused block is put back exactly as it was.  This keeps the objective
    non-increasing at every step whether or not the block really shares no
    arc.

    Returns
    -------
    int
        Number of moves whose jump was changed.
    """
    if group.size == 0:
        return 0
    move_edges = edges[group]
    move_signs = signs[group]
    active = move_signs != 0
    current = flow[move_edges]
    # The padding of a vertex carries no jump, so it must weigh on neither
    # the relaxed move nor the objective.
    price = np.where(active, curvature[move_edges], 0.0)
    target = centre[move_edges]
    numerator = np.sum(price * move_signs * (target - current), axis=1)
    denominator = np.sum(price, axis=1)
    safe = np.where(denominator > 0.0, denominator, 1.0)
    relaxed = np.where(denominator > 0.0, numerator / safe, 0.0)
    guess = np.rint(relaxed).astype(np.int64)
    turns = ((guess[:, None] + steps[None, :])[:, :, None]
             * move_signs[:, None, :])
    trial = current[:, None, :] + turns
    arc = np.broadcast_to(move_edges[:, None, :], trial.shape)
    total = cost.cost_at(arc, trial).sum(axis=2)
    # The column at ``window`` is the move that changes nothing at all.
    width = steps.size // 2
    here = total[:, width]
    best = np.argmin(total, axis=1)
    gain = here - total[np.arange(group.size), best]
    moving = np.flatnonzero(gain > _EPSILON * np.maximum(1.0, np.abs(here)))
    if moving.size == 0:
        return 0
    move = guess[moving] + steps[best[moving]]
    before = float(np.sum(cost.evaluate(flow)))
    update_edges = move_edges[moving][active[moving]]
    touched = np.unique(update_edges)
    saved = flow[touched].copy()
    # An arc can be named twice inside one move, and the padding names an arc
    # that the move must leave alone, so the jumps have to be accumulated.
    update = (move[:, None] * move_signs[moving])[active[moving]]
    np.add.at(flow, update_edges, update)
    after = float(np.sum(cost.evaluate(flow)))
    if after > before + _EPSILON * max(1.0, abs(before)):
        flow[touched] = saved
        return 0
    return int(moving.size)


def _cycle_descent(flow, cost, curvature, centre, nodes, tail, head, *,
                   window, max_sweeps):
    """Lower the exact cost with moves that keep every loop closed.

    The jumps of a flow are fixed up to the closure of the loops, so a move
    is legal only when it changes every closure by zero.  Raising one vertex
    and lowering every other vertex by the same turn is such a move: the two
    ends of an arc see opposite turns, which is exactly the difference of a
    potential, and a move of that form is a gradient.  A square boundary is
    not of that form: the turns of its four arcs add up to four turns inside
    the square instead of cancelling, so a square move is a vortex that
    rewrites the residues of the loops and leaves the problem behind.  A star
    touches only the arcs of one vertex, which makes an exact line search
    possible on it: the move is put to the whole number of turns nearest the
    real minimiser of the quadratic part, the neighbouring whole numbers are
    scored with the exact cost, and a sweep over all the vertices is repeated
    until no vertex helps.

    Parameters
    ----------
    flow : numpy.ndarray of int
        Whole turns per arc, changed in place.
    cost : DeformationCost
        Exact cost of a flow.
    curvature, centre : numpy.ndarray
        Quadratic model used to guess the move of a vertex.
    nodes : int
        Number of vertices.
    tail, head : numpy.ndarray
        Endpoints of every arc.
    window : int
        Half width of the search around the guessed move.
    max_sweeps : int
        Limit on the number of sweeps.

    Returns
    -------
    int
        Sweeps that lowered the cost.
    int
        Moves that changed a jump.
    """
    edges, signs = _star_moves(nodes, tail, head)
    steps = np.arange(-window, window + 1, dtype=np.int64)
    blocks = _colour_vertex_blocks(edges, signs, tail, head)
    moves = 0
    sweeps = 0
    for _ in range(max_sweeps):
        progressed = False
        if blocks is None:
            groups = (np.array([index]) for index in range(edges.shape[0]))
        else:
            groups = blocks
        for group in groups:
            taken = _lower_block(flow, cost, curvature, centre, edges, signs,
                                 group, steps)
            moves += taken
            progressed = progressed or taken > 0
        if not progressed:
            break
        sweeps += 1
    return sweeps, moves


def stat_cost_unwrap(phase, coherence, weight=None, *, mask=None, params=None,
                     costmode="defo", shelf=True, start="smooth", window=3,
                     max_sweeps=8, flatten=None, align="zero", max_iter=None,
                     return_info=False):
    """Unwrap a two dimensional interferogram with statistical costs.

    The method is the one of SNAPHU.  Every arc is priced by how well a jump
    explains the residual that is left when the wrapped step of the arc is
    compared with the average wrapped step of a box around it, and the price
    per turn falls with the coherence of the arc, so a noisy arc is cheap to
    jump and a clean one is expensive.  In the default deformation mode the
    price is also capped and then rises again, which allows a jump much larger
    than the noise to be charged at the level of a real deformation.  See the
    module docstring for the equations.

    Parameters
    ----------
    phase : array_like of float
        Wrapped phase in radians, two dimensional and at least two samples
        along every axis.
    coherence : array_like of float
        Coherence of the interferogram, the same shape as ``phase`` and
        between zero and one.  It is a property of the measurement, so it has
        to be supplied; :func:`parvaneh.reliability.pixel_reliability` can
        produce a stand-in when only the wrapped phase is available.
    weight : array_like of float, optional
        Extra per-sample weight, the same shape as ``phase``.  The default is
        one everywhere.  The price of an arc is the smaller weight of its two
        ends times ``1 / sigma ** 2``.
    mask : array_like of bool, optional
        Samples to ignore.  Every arc that touches one of them gets weight
        zero, which makes it free.
    params : StatCostParams, optional
        Constants of the noise model.  The default describes an interferogram
        with about twenty-four looks.
    costmode : {"defo", "smooth"}, optional
        ``"defo"`` is the deformation model: a lower coherence threshold, a
        smaller share of the averaged step in the range direction, and a
        cap.  ``"smooth"`` is the smooth model: the whole average, the higher
        threshold and no cap, whose cost is solved exactly.  Adding
        ``shelf=False`` to ``"defo"`` drops only the cap.
    shelf : bool, optional
        Whether the cap may be used at all, in the deformation mode.  Default
        true.
    start : {"smooth", "zero"}, optional
        Where the refinement of the capped model begins: at the exact
        solution of the convex model, or at no jump at all.
    window : int, optional
        Half width, in turns, of the search around the move that the convex
        model would make.  Default three.
    max_sweeps : int, optional
        Limit on the sweeps of the refinement over the vertices of the grid.
    flatten : array_like of float, optional
        An estimate of the unwrapped phase in radians, usually from a coarse
        reference or an earlier pass of lower resolution.  When it is given
        the wrapped phase is flattened with it before the priors are built,
        the estimate is added back to every arc as a step, and the estimate
        is added to the result, so that the phase trend of the estimate does
        not have to be recovered from the wrapped data.  This is the ``-e``
        option of SNAPHU.
    align : {"zero", "mean", "median"}, optional
        How to fix the additive constant of every connected component.
    max_iter : int, optional
        Augmentation limit of the network solver behind the convex model.  A
        limit too small to balance the supplies raises ``RuntimeError``.
    return_info : bool, optional
        Also return a :class:`StatCostInfo`.

    Returns
    -------
    numpy.ndarray
        Unwrapped phase in radians, ``NaN`` where the confidence is zero.
    StatCostInfo
        Only when ``return_info`` is true.

    Raises
    ------
    ValueError
        If an argument is unknown or ill shaped, or if the box average does
        not fit the gradient array.
    RuntimeError
        If ``max_iter`` stops the flow solver before the supplies of the
        convex model are balanced.

    Notes
    -----
    ``proven`` in the report says whether the objective is known to be the
    smallest one.  It is true for the quadratic model, which is convex and
    solved exactly, and false whenever the cap is used on at least one arc,
    because the capped cost is not convex and this method then reports only a
    local optimum of a descent that starts from the convex model.  The moves
    of that descent raise one vertex of the grid at a time, which is the
    difference of an integer potential across its arcs.  Such a move is a
    gradient, so it leaves the closure of every loop alone and every flow of
    the descent closes the loops exactly like the flow it started from.

    Examples
    --------
    >>> import numpy as np                          # doctest: +SKIP
    >>> phase = np.zeros((16, 16), dtype=float)     # doctest: +SKIP
    >>> coherence = np.full((16, 16), 0.8)          # doctest: +SKIP
    >>> field = stat_cost_unwrap(phase, coherence)  # doctest: +SKIP
    """
    if costmode not in COST_MODES:
        raise ValueError("costmode must be one of %s" % (COST_MODES,))
    if start not in START_MODES:
        raise ValueError("start must be one of %s" % (START_MODES,))
    if align not in ALIGN_MODES:
        raise ValueError("align must be one of %s" % (ALIGN_MODES,))
    if int(window) < 0:
        raise ValueError("window must not be negative")
    if int(max_sweeps) < 0:
        raise ValueError("max_sweeps must not be negative")
    if params is None:
        params = StatCostParams()
    array = _check_phase(phase)
    if array.ndim != 2:
        raise ValueError(
            "statistical costs are defined on a two dimensional grid")
    coherence = _check_coherence(coherence, array.shape)
    confidence = _confidence(array, mask, weight)
    estimate = None
    if flatten is not None:
        estimate = np.ascontiguousarray(flatten, dtype=np.float64)
        if estimate.shape != array.shape:
            raise ValueError("flatten must match phase.shape")
        if not np.isfinite(estimate).all():
            raise ValueError("flatten must be finite")
    threshold = params.defocorrthresh if costmode == "defo" else params.rho0
    inner = array if estimate is None else _wrap(array - estimate)
    model = _build_model(inner, confidence, coherence, params, threshold,
                         estimate, bool(shelf) and costmode == "defo",
                         costmode)
    quadratic = QuadraticCost(model.curvature, model.centre)
    exact = DeformationCost(model.curvature, model.centre, shelf=model.shelf,
                            level=model.level, defomax=params.defomax,
                            falloff=params.layfalloffconst)
    vertices, tail, head = grid_edges(array.shape)
    nodes = int(vertices.size)
    cycles = grid_cycles(array.shape, grid_edge_index(array.shape))
    if exact.convex:
        flow, info = curl_flow(nodes, tail, head, model.gradients,
                               cost=quadratic, cycles=cycles, method="flow",
                               max_iter=max_iter, return_info=True)
        start_objective = float(np.sum(exact.evaluate(flow)))
        sweeps = 0
        moves = 0
        proven = bool(info.proven)
        backend = "stat-costs-flow"
    else:
        if start == "zero":
            flow = np.zeros(model.curvature.size, dtype=np.int64)
        else:
            flow = curl_flow(nodes, tail, head, model.gradients,
                             cost=quadratic, cycles=cycles, method="flow",
                             max_iter=max_iter)
        flow = np.array(flow, dtype=np.int64)
        start_objective = float(np.sum(exact.evaluate(flow)))
        sweeps, moves = _cycle_descent(flow, exact, model.curvature,
                                       model.centre, nodes, tail, head,
                                       window=int(window),
                                       max_sweeps=int(max_sweeps))
        proven = False
        backend = "stat-costs-descent"
    objective = float(np.sum(exact.evaluate(flow)))
    gaps = residues_of_cycles(model.gradients, cycles)
    field, labels = _restore(inner, confidence, estimate, flow, array.shape,
                             align)
    info = StatCostInfo(
        backend=backend,
        costmode=costmode,
        shape=tuple(int(size) for size in array.shape),
        pixels=int(np.count_nonzero(confidence > _MIN_WEIGHT)),
        nodes=nodes,
        edges=int(model.curvature.size),
        cycles=int(len(cycles)),
        residues=int(np.count_nonzero(gaps)),
        components=int(labels.max()) + 1,
        shelf_edges=int(np.count_nonzero(exact.shelf)),
        masked_edges=int(np.count_nonzero(model.weight <= _MIN_WEIGHT)),
        max_jump=int(np.abs(flow).max()) if flow.size else 0,
        objective=objective,
        initial_objective=start_objective,
        sweeps=int(sweeps),
        moves=int(moves),
        proven=bool(proven),
        ncorrlooks=float(params.ncorrlooks),
        rho0=float(params.rho0),
        defocorrthresh=float(params.defocorrthresh),
    )
    if return_info:
        return field, info
    return field


def _check_coherence(coherence, shape):
    """Validate a coherence array and return it as a float array."""
    array = np.ascontiguousarray(coherence, dtype=np.float64)
    if array.shape != tuple(shape):
        raise ValueError("coherence must match phase.shape")
    if not np.isfinite(array).all():
        raise ValueError("coherence must be finite")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError("coherence must lie between 0 and 1")
    return array


def _jump_blocks(shape, flow):
    """Lay a flat flow out as one array of whole turns per axis."""
    blocks = []
    offset = 0
    for axes in _edge_shapes(shape):
        size = int(np.prod(axes))
        blocks.append(flow[offset:offset + size].reshape(axes))
        offset += size
    return blocks


def _restore(inner, confidence, estimate, flow, shape, align):
    """Integrate the jumps of a flat flow and add the estimate back."""
    field, labels = integrate_edges(inner, confidence,
                                    _jump_blocks(shape, flow), align=align)
    if estimate is not None:
        field = field + estimate
    return field, labels
