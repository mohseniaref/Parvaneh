"""Extended minimum cost flow unwrapping of a stack of interferograms.

A stack of interferograms of one scene holds more information than any single
interferogram of it: every sample of the scene is measured along the whole
acquisition history, so a set of measurements that cannot be the history of one
sample is inconsistent with itself, whatever the image alone suggests.  The
extended minimum cost flow (EMCF) method of Pepe and Lanari [2] uses that
information in a cheap way.  It first removes the whole turns that make the
stack inconsistent from one acquisition to the next, and then unwraps every
interferogram of the corrected stack in space, on its own, as an integer
network flow.  Both steps are the same kind of problem, and both are solved
here by :func:`parvaneh.graph_flow.curl_flow`.

The problem
-----------
A wrapped measurement is known modulo one turn, so unwrapping an interferogram
means choosing a whole number of turns for every pair of neighbouring samples.
Those pairs are the arcs of the image grid, and the only constraint on the
whole numbers is that their signed sum around every loop of the grid cancels
the residue of the wrapped differences around that loop, a whole number that
vanishes wherever the field is smooth.  That is the minimum cost flow
formulation of Costantini [1], which :mod:`parvaneh.network_flow` solves in two
dimensions.

A stack of interferograms adds a second family of loops.  The measurements of
one arc of the image are differences of one history of that arc, so the loops
of the acquisition network, the closed chains of acquisitions, have to close as
well: the signed sum of the measurements of one arc around such a loop must
vanish.  The corrections are whole turns added to whole interferograms, so the
temporal requirement has exactly the shape of the residue requirement of the
image, and [2] solves it with the same machinery.

The method in one paragraph
---------------------------
Time first.  For every arc of the image, round the residues of the temporal
loops of the measurements of that arc, solve an integer network flow over the
acquisitions with those residues as supplies, and add the whole turns of that
flow to the measurements.  Space second.  The corrected measurements of an arc
now close every temporal loop, so unwrap every interferogram of the corrected
stack in space as a flow of its own.  The temporal stage makes the stack
mutually consistent; the spatial stage turns each corrected interferogram into
a field.

The temporal network
--------------------
The acquisitions are the vertices of a small graph whose edges are the
interferograms.  An edge joins the earlier acquisition of its pair to the later
one, so the measurement of an interferogram is the phase of the later
acquisition minus the phase of the earlier one.  With ``L`` interferograms over
``E`` acquisitions the graph has ``L`` edges, and the default network here is
the hop three network, which joins acquisitions that are one, two or three
steps apart: ``L = 3 E - 6`` and the loops are ``2 E - 5`` triangles, for
``E >= 4``.

Let ``u[j]`` be the measurement of interferogram ``j`` at the arc of the image
being corrected, in radians, and let ``sigma[c, j]`` be ``+1`` when loop ``c``
walks edge ``j`` from its earlier acquisition to its later one and ``-1`` when
it walks it the other way.  The circulation of the loop is

    r[c] = rint( sum_j sigma[c, j] u[j] / (2 pi) )

and a whole number of turns ``k[j]`` is added to the measurements, so that the
corrected measurements close every loop:

    sum_j sigma[c, j] k[j] = -r[c]

This is the supply constraint of a network flow: loop ``c`` has to send the
residue ``r[c]`` away, and the turns ``k`` are the flow.

The temporal stage
------------------
Every arc of the image has its own measurements and therefore its own flow
problem, with the same graph and the same prices; the price of a turn is one
for every interferogram.  The flows of the arcs do not interact, so solving
them one after the other reaches the same minimum as solving them together, and
an arc whose residues all vanish is left alone: the cheapest way of closing a
loop that closes already is to add no turns at all, because a sum of absolute
values vanishes only when every term does.  The arcs are taken in blocks of
``link_batch`` so that the residues of a whole block are computed at once.

The spatial stage
-----------------
With the turns ``k`` of the temporal stage in place, interferogram ``j`` has
the corrected steps ``g[j] + 2 pi k[j]``, where ``g[j]`` is its wrapped step.
Adding whole turns does not change the residues of the image, so the spatial
problem of every interferogram has the supplies it had before and is solved
with the same engine.  The whole turns that reach
:func:`parvaneh.flow_nd.integrate_edges` are the sum of the two stages,
``k[j] + f[j]``, which closes every loop of the image by construction, so the
walk that integrates them does not depend on its order.

The cost of an arc
------------------
The temporal stage pays one for every turn of every interferogram, so it
minimises the number of turns it adds, which the report calls
``temporal_turns``.  The spatial stage pays the confidence of the two samples
that an arc joins, so it minimises a confidence weighted number of turns, which
is the objective of the other wrapped cost methods of the package: an arc that
the caller trusts little is allowed to absorb turns, and an arc whose
confidence is zero is free, which is what a masked sample means.  The two
stages therefore minimise different prices, and ``objective`` in the report is
the spatial one, summed over the interferograms.  With no mask and no weight
every confidence is one, the two prices agree, and both stages minimise the
plain number of turns.

Gauge
-----
Unwrapping fixes a field only up to a whole number of turns per connected group
of samples, because no measurement can see such a constant.  The constant of
every group is set here by ``align``, as in
:func:`parvaneh.flow_nd.flow_nd_unwrap`.  The temporal stage leaves a second
ambiguity of the same kind, since adding one turn to every measurement of one
interferogram is invisible in the wrapped data; the flow is the cheapest way of
closing the temporal loops, so that ambiguity is a tie of the objective rather
than a free choice.  Whatever the choice, the corrected measurements of a
returned field are congruent to the wrapped measurements modulo one turn at
every sample, so two fields that differ by such a gauge describe the same data.
The constants of two interferograms are not related to each other, so this
module returns one field per interferogram and not a series in time.

Two dimensions and three
------------------------
The measurements of a stack are one interferogram per acquisition pair, and an
interferogram is a plane: the leading axis here is the interferogram and not a
third axis of the image, which is what separates EMCF from
:mod:`parvaneh.flow_nd`, where the loops of one volume wind through time.  The
loops of the image have to be a flow network and the walk that integrates the
answer has to be order independent, and both hold for a plane, whose elementary
squares span every loop of the grid, while neither holds for a volume.  A
general graph of samples, such as the triangulation over a set of scatterers
that [2] also describes, is not built here; a caller who has one can pass its
loops through ``cycles`` for the acquisitions, but the image is still walked as
a grid.

References
----------
[1] M. Costantini, "A novel phase unwrapping method based on network
programming," *IEEE Transactions on Geoscience and Remote Sensing* 36(3)
(1998) 813-821.

[2] A. Pepe and R. Lanari, "On the extension of the minimum cost flow algorithm
for phase unwrapping of multitemporal differential SAR interferograms," *IEEE
Transactions on Geoscience and Remote Sensing* 44(9) (2006) 2374-2383.

[3] A. Pepe, L. D. Euillades, M. Manunta and R. Lanari, "New advances of the
extended minimum cost flow phase unwrapping algorithm for SBAS-DInSAR analysis
at full spatial resolution," *IEEE Transactions on Geoscience and Remote
Sensing* 49(10) (2011) 4062-4079.

[4] A. Pepe, Y. Yang, M. Manzo and R. Lanari, "Improved EMCF-SBAS processing
chain based on advanced techniques for the noise-filtering and selection of
small baseline multi-look DInSAR interferograms," *IEEE Transactions on
Geoscience and Remote Sensing* 53(8) (2015) 4394-4417.

[5] C. W. Chen and H. A. Zebker, "Network approaches to two-dimensional phase
unwrapping: intractability and two new algorithms," *Journal of the Optical
Society of America A* 17(3) (2000) 401-414.

[6] M. T. Calef, P. S. Agram, K. M. Olsen, S. J. Staniewicz, G. M. Gunter and
H. Fattahi, "spurt: spatial and temporal phase unwrapping for InSAR time
series," California Institute of Technology, 2024, released under BSD-3-Clause
OR Apache-2.0.

Notes
-----
The formulation is taken from the papers listed above and, for the details of
the networks, from the sources of the SPURT software [6], which is released
under a permissive licence and is used here only as a description of the
method; no code of it is copied, translated or called.  The parts of the method
that this module leaves out are listed below.

* The temporal network is built only for the interferograms that the caller
  gives, and the default is the hop three network of [2].  A network that the
  data itself suggests, such as a triangulation of the acquisition times or the
  small baseline network of [4], is passed through ``links``.  When the
  interferograms are not the hop three network, the loops that the temporal
  stage closes are those of a fundamental cycle basis of the network, as
  returned by :func:`parvaneh.graph_flow.graph_basis`; a network without a loop
  needs no temporal correction at all and is unwrapped in space alone.
* The temporal stage is one flow problem per arc of the image, each over the
  whole acquisition network, so its work grows with the number of arcs.  SPURT
  arranges the same solves as one large problem per block of links, which is
  the same work with fewer calls; the arcs are independent, so the two minimum
  costs agree, and this module keeps the smaller memory instead.
* The confidence of a sample prices the arcs of the image of every
  interferogram alike, while SPURT prices every arc of the image and of the
  acquisitions by one.  The two agree on a full scene with no mask and no
  weight, which is the case of the papers.
* The spatial loops are the elementary squares of the image grid.  The
  triangular network of a set of scatterers that [2] describes is not built,
  and neither is the Delaunay network of [6].
* The phase model term of the difference operator of SPURT, which is meant for
  simulated data, is not applied: the wrapped difference of two samples is used
  as it stands.
* The two stages are minimised one after the other, so the field is the exact
  minimum of each stage given the other, and ``objective`` is the cost of the
  field that is returned rather than a joint minimum over both stages.
* Both stages call the same solver and pass ``method`` on to it.  ``"flow"``
  requires the loops at hand to be a flow network.  The squares of a plane are
  one and the triangles of the hop three network are one; the loops of a
  fundamental basis of an arbitrary network need not be, and a caller who
  forces ``"flow"`` there is told so.
* Time is not unwrapped.  No series in time is assembled, and the temporal flow
  of an arc is only a correction of the measurements.  Unwrapping the phase of
  the acquisitions themselves is the subject of :mod:`parvaneh.space_time`.
"""

from dataclasses import dataclass

import numpy as np

from .core import _wrap
from .flow_nd import (ALIGN_MODES, _edge_shapes, grid_cycles,
                      grid_edge_index, grid_edges, integrate_edges)
from .graph_flow import LinearCost, curl_flow, graph_basis
from .reliability import _axis_confidence, _check_phase, _confidence

_TWO_PI = 2.0 * np.pi

#: Smallest number of acquisitions that the hop three network covers.
_HOP3_EPOCHS = 4

#: Largest gap, in acquisitions, of the default temporal network.
_HOP3_LAGS = (1, 2, 3)

#: Number of image arcs whose temporal problem is prepared at once.
_LINK_BATCH = 50000

#: Confidence above which a sample counts as measured.
_MIN_WEIGHT = 0.0


@dataclass
class EmcfInfo:
    """Bookkeeping for one extended minimum cost flow run.

    Every field counts the work of a stage, the size of what it was given, or
    both, and a caller can compare two runs of the same stack from them.  No
    field changes the returned phase.

    Attributes
    ----------
    backend : str
        Name of the method, always ``"emcf-flow"``.
    network : str
        ``"hop3"`` when the interferograms are the default network over the
        acquisitions, ``"links"`` when they are any other network.
    mode : str
        ``"stack"`` when the caller passed phases of the acquisitions and the
        interferograms were formed here, ``"interferograms"`` when the caller
        passed the measurements directly.
    shape : tuple of int
        Shape of one interferogram.
    epochs : int
        Number of acquisitions of the temporal network.
    interferograms : int
        Number of interferograms, that is, of temporal edges.
    pixels : int
        Number of samples of the scene that carry confidence.
    nodes : int
        Number of vertices of the image grid.
    edges : int
        Number of arcs of the image grid.
    cycles : int
        Number of loops that the temporal stage closed.
    temporal_solves : int
        Number of arcs whose temporal flow problem was solved.
    temporal_residues : int
        Sum of the absolute loop residues of the temporal stage.
    temporal_turns : int
        Sum of the absolute whole turns that the temporal stage added.
    residues : int
        Sum of the residue counts of the spatial solves.
    components : int
        Number of connected groups of the answer, over all interferograms.
    max_jump : int
        Largest absolute whole turn of the answer, over both stages.
    objective : float
        Confidence weighted number of turns of the spatial stage, summed over
        the interferograms.
    objective_per_interferogram : numpy.ndarray
        The same figure for every interferogram on its own.
    augmentations : int
        Number of augmenting walks of both stages together.
    proven : bool
        Whether every solve reached the optimum of its objective instead of
        stopping at a limit.
    """

    backend: str
    network: str
    mode: str
    shape: tuple
    epochs: int
    interferograms: int
    pixels: int
    nodes: int
    edges: int
    cycles: int
    temporal_solves: int
    temporal_residues: int
    temporal_turns: int
    residues: int
    components: int
    max_jump: int
    objective: float
    objective_per_interferogram: np.ndarray
    augmentations: int
    proven: bool


@dataclass
class _FlowTotals:
    """Solves, augmenting walks and proofs gathered over a stage."""

    solves: int = 0
    augmentations: int = 0
    proven: bool = True

    def add(self, result):
        """Fold the report of one solve into the totals."""
        self.solves += 1
        self.augmentations += int(result.augmentations)
        self.proven = bool(self.proven and result.proven)


def emcf_links(n_epochs):
    """Pairs of acquisitions of the default hop three network.

    An acquisition is joined to the next three acquisition steps, which is the
    network of the extended minimum cost flow method: enough loops to correct
    the measurements against each other, and few enough interferograms to keep
    the coherence of a pair high.

    Parameters
    ----------
    n_epochs : int
        Number of acquisitions.

    Returns
    -------
    numpy.ndarray
        One row per interferogram, ``(L, 2)``, holding the earlier acquisition
        first.  The rows are ordered by their first acquisition and then by
        their second one, ``L = 3 E - 6`` rows for ``E >= 4``.

    Raises
    ------
    ValueError
        If fewer than four acquisitions are given.

    Examples
    --------
    >>> emcf_links(5).tolist()
    [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4],
     [3, 4]]
    """
    epochs = int(n_epochs)
    if epochs < _HOP3_EPOCHS:
        raise ValueError("the hop three network needs at least four epochs")
    links = [(start, start + lag)
             for start in range(epochs)
             for lag in _HOP3_LAGS if start + lag < epochs]
    return np.asarray(links, dtype=np.int64)


def emcf_cycles(n_epochs):
    """Triangles of the default hop three network.

    The loops of the network are its triangles, which is what the temporal
    stage closes.

    Parameters
    ----------
    n_epochs : int
        Number of acquisitions.

    Returns
    -------
    list of tuple of numpy.ndarray
        Loops as ``(edge indices, signs)``.  The edge indices number the rows
        of :func:`emcf_links`, and a sign is ``+1`` when the loop walks an edge
        from its earlier acquisition to its later one and ``-1`` when it walks
        it the other way.

    Raises
    ------
    ValueError
        If fewer than four acquisitions are given.

    Examples
    --------
    >>> loops = emcf_cycles(4)
    >>> len(loops)
    3
    """
    epochs = int(n_epochs)
    return _hop3_cycles(emcf_links(epochs), epochs)


def emcf_unwrap(phase, links=None, *, mask=None, weight=None, cycles=None,
                method="auto", align="zero", max_iter=None,
                link_batch=_LINK_BATCH, return_info=False):
    """Unwrap a stack of acquisitions with the EMCF method.

    The measurements of the interferograms that ``links`` lists are formed
    here, wrapped to one turn, and are then corrected in time and unwrapped in
    space, as described in the module docstring.

    Parameters
    ----------
    phase : array_like
        Wrapped phase of a stack of acquisitions, in radians, with the
        acquisitions on the first axis, the epochs, and a two dimensional
        image behind them.  A complex stack of single look data is accepted as
        well and is turned into its argument, which is the convention of the
        method; every other backend of the package wants radians.
    links : array_like of int, optional
        Pairs of acquisitions, one row per interferogram, ``(L, 2)`` with the
        earlier acquisition first.  No pair may repeat.  The default is the hop
        three network of :func:`emcf_links`, which needs at least four
        acquisitions.
    mask : array_like of bool, optional
        Samples of the scene to leave out, with the shape of one
        interferogram.  A masked sample has no confidence, so every arc that
        touches it is free and the answer is ``NaN`` there.
    weight : array_like of float, optional
        Relative trust in every sample, with the shape of one interferogram
        and one everywhere by default.
    cycles : sequence of (array_like of int, array_like of int), optional
        Loops of the acquisition network, as pairs of edge indices and their
        ``+1``/``-1`` traversal signs, the indices numbering ``links`` and a
        sign describing the walk from the earlier acquisition to the later
        one.  A loop may not use an edge twice.  The triangles of
        :func:`emcf_cycles` are used when the interferograms are the hop three
        network, the loops of a fundamental cycle basis are used otherwise, and
        an empty sequence means that no loop is to be closed.
    method : {"auto", "flow", "ilp"}, optional
        Which solver of :func:`parvaneh.graph_flow.curl_flow` to use.  Both
        stages pass it on unchanged.
    align : {"zero", "mean", "median"}, optional
        How to fix the additive constant of every connected group of samples,
        as in :func:`parvaneh.flow_nd.integrate_edges`.  Every interferogram is
        aligned on its own, because the measurements of one acquisition pair
        say nothing about the level of another pair.
    max_iter : int, optional
        Augmentation limit of the network solver.
    link_batch : int, optional
        Number of arcs of the image whose temporal flow problem is prepared at
        once.
    return_info : bool, optional
        Also return an :class:`EmcfInfo`.

    Returns
    -------
    numpy.ndarray
        Unwrapped phase in radians, one plane per interferogram of ``links``
        and in the same order, ``NaN`` where the confidence is zero.
    EmcfInfo
        Only when ``return_info`` is true.

    Raises
    ------
    ValueError
        If an argument is unknown or ill shaped, if a pair of acquisitions is
        not a pair of the stack, or if the stack is not a two dimensional
        image of acquisitions.
    RuntimeError
        If ``max_iter`` stops the network solver before a loop is closed.

    Notes
    -----
    The phase is expected to be wrapped already and is used as it stands, as by
    every other backend of the package, so a step of more than one turn between
    neighbouring samples cannot be recovered by any of them.

    Examples
    --------
    >>> import numpy as np                                 # doctest: +SKIP
    >>> stack = np.zeros((5, 16, 16))                      # doctest: +SKIP
    >>> field = emcf_unwrap(stack)                          # doctest: +SKIP
    >>> field.shape                                        # doctest: +SKIP
    (9, 16, 16)
    """
    if align not in ALIGN_MODES:
        raise ValueError("align must be one of %s" % (ALIGN_MODES,))
    array = np.asarray(phase)
    if np.iscomplexobj(array):
        array = np.angle(array)
    array = _check_phase(array)
    if array.ndim != 3:
        raise ValueError("a stack needs an epoch axis and a two dimensional "
                         "image")
    epochs = int(array.shape[0])
    pairs = emcf_links(epochs) if links is None else _check_links(links)
    if np.any(pairs[:, 1] >= epochs):
        raise ValueError("a link names an acquisition outside the stack")
    confidence = _confidence(array[0], mask, weight)
    return _emcf_core(_form_interferograms(array, pairs), pairs, epochs,
                      confidence, cycles, mode="stack", method=method,
                      align=align, max_iter=max_iter, link_batch=link_batch,
                      return_info=return_info)


def emcf_unwrap_interferograms(phase, links, *, mask=None, weight=None,
                               cycles=None, method="auto", align="zero",
                               max_iter=None, link_batch=_LINK_BATCH,
                               return_info=False):
    """Unwrap measurements of a stack that is already interferometric.

    This is the same method as :func:`emcf_unwrap`, for a caller whose stack
    holds the differences of phase of pairs of acquisitions rather than the
    phases of the acquisitions, which is what a small baseline pipeline has.
    Nothing else changes: the temporal network is the one that ``links``
    describes, and the measurements are corrected and unwrapped in the same
    way.

    Parameters
    ----------
    phase : array_like
        Wrapped measurements, in radians, one interferogram per row of
        ``links`` and in the same order, with a two dimensional image behind
        them.  Complex measurements are accepted as well and are turned into
        their argument.
    links : array_like of int
        Pairs of acquisitions, one row per interferogram, ``(L, 2)`` with the
        earlier acquisition first, matching the measurements row by row.  No
        pair may repeat.
    mask, weight, cycles, method, align, max_iter, link_batch, return_info
        As in :func:`emcf_unwrap`.

    Returns
    -------
    numpy.ndarray
        Unwrapped phase in radians, with the same shape as ``phase``.
    EmcfInfo
        Only when ``return_info`` is true.

    Raises
    ------
    ValueError
        If an argument is unknown or ill shaped, or if the measurements are
        not a two dimensional image of interferograms.

    Notes
    -----
    The acquisitions of the network are the numbers in ``links``, of which
    there is one more than the largest number used.  A network without a loop,
    such as the chain of the classic small baseline configurations, has no
    temporal requirement at all and is unwrapped in space alone.

    Examples
    --------
    >>> import numpy as np                                 # doctest: +SKIP
    >>> links = emcf_links(5)                               # doctest: +SKIP
    >>> phase = np.zeros((links.shape[0], 16, 16))          # doctest: +SKIP
    >>> field = emcf_unwrap_interferograms(phase, links)    # doctest: +SKIP
    """
    if align not in ALIGN_MODES:
        raise ValueError("align must be one of %s" % (ALIGN_MODES,))
    array = np.asarray(phase)
    if np.iscomplexobj(array):
        array = np.angle(array)
    array = _check_phase(array)
    if array.ndim != 3:
        raise ValueError("the measurements need one interferogram per row of "
                         "links and a two dimensional image")
    pairs = _check_links(links)
    if pairs.shape[0] != array.shape[0]:
        raise ValueError("links must have one pair per interferogram")
    epochs = int(pairs[:, 1].max()) + 1
    confidence = _confidence(array[0], mask, weight)
    return _emcf_core(array, pairs, epochs, confidence, cycles,
                      mode="interferograms", method=method, align=align,
                      max_iter=max_iter, link_batch=link_batch,
                      return_info=return_info)


def _emcf_core(ifg, pairs, epochs, confidence, cycles, *, mode, method, align,
               max_iter, link_batch, return_info):
    """Correct a stack of measurements in time and unwrap it in space.

    ``ifg`` holds one wrapped measurement per row of ``pairs`` and one plane
    of the image per measurement, so the first axis is the interferogram.
    """
    if int(link_batch) < 1:
        raise ValueError("link_batch must be a positive number of arcs")
    interferograms, rows, cols = ifg.shape
    frame = (rows, cols)
    steps = _arc_steps(ifg)
    weights = _arc_weights(confidence)
    tails = np.ascontiguousarray(pairs[:, 0])
    heads = np.ascontiguousarray(pairs[:, 1])
    turns = np.zeros(steps.shape, dtype=np.int64)
    totals = _FlowTotals()
    temporal_solves = 0
    temporal_residues = 0
    temporal_turns = 0
    if _is_hop3_links(pairs, epochs):
        network = "hop3"
        loops = _hop3_cycles(pairs, epochs)
    else:
        network = "links"
        loops = graph_basis(epochs, tails, heads).cycles
    if cycles is not None:
        loops = _check_cycles(cycles, pairs.shape[0])
    if loops:
        price = LinearCost(np.ones(pairs.shape[0], dtype=np.float64))
        for start in range(0, steps.shape[1], int(link_batch)):
            stop = start + int(link_batch)
            block = steps[:, start:stop]
            residues = _temporal_residues(loops, block)
            temporal_residues += int(np.abs(residues).sum())
            for column in np.flatnonzero(np.any(residues != 0, axis=0)):
                flow, result = curl_flow(
                    epochs, tails, heads, block[:, column], cost=price,
                    cycles=loops, method=method, max_iter=max_iter,
                    return_info=True)
                turns[:, start + column] = flow
                temporal_solves += 1
                temporal_turns += int(np.sum(price.evaluate(flow)))
                totals.add(result)
    vertices, tail, head = grid_edges(frame)
    nodes = int(vertices.size)
    area_cycles = grid_cycles(frame, grid_edge_index(frame))
    price = LinearCost(weights)
    field = np.empty(ifg.shape, dtype=np.float64)
    per_interferogram = np.empty(interferograms, dtype=np.float64)
    residues = 0
    max_jump = 0
    components = 0
    for index in range(interferograms):
        gradients = steps[index] + _TWO_PI * turns[index]
        flow, result = curl_flow(nodes, tail, head, gradients, cost=price,
                                 cycles=area_cycles, method=method,
                                 max_iter=max_iter, return_info=True)
        per_interferogram[index] = float(np.sum(price.evaluate(flow)))
        residues += int(result.residues)
        totals.add(result)
        jumps = turns[index] + flow
        max_jump = max(max_jump, int(np.abs(jumps).max()))
        field[index], labels = integrate_edges(
            ifg[index], confidence, _jump_blocks(frame, jumps), align=align)
        components = max(components, int(labels.max()) + 1)
    info = EmcfInfo(
        backend="emcf-flow",
        network=network,
        mode=mode,
        shape=tuple(int(size) for size in frame),
        epochs=int(epochs),
        interferograms=int(interferograms),
        pixels=int(np.count_nonzero(confidence > _MIN_WEIGHT)),
        nodes=nodes,
        edges=int(tail.size),
        cycles=int(len(loops)),
        temporal_solves=int(temporal_solves),
        temporal_residues=int(temporal_residues),
        temporal_turns=int(temporal_turns),
        residues=int(residues),
        components=int(components),
        max_jump=int(max_jump),
        objective=float(per_interferogram.sum()),
        objective_per_interferogram=per_interferogram,
        augmentations=int(totals.augmentations),
        proven=bool(totals.proven),
    )
    if return_info:
        return field, info
    return field


def _check_links(links):
    """Check a list of pairs of acquisitions and normalise it to int64."""
    values = np.asarray(links)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("links must be an (L, 2) array of pairs")
    if values.shape[0] < 1:
        raise ValueError("links needs at least one interferogram")
    numbers = np.asarray(values, dtype=np.float64)
    if not np.isfinite(numbers).all() or not np.array_equal(
            numbers, np.rint(numbers)):
        raise ValueError("links must hold whole acquisition numbers")
    pairs = np.rint(numbers).astype(np.int64)
    if np.any(pairs < 0):
        raise ValueError("an acquisition number must not be negative")
    if np.any(pairs[:, 1] <= pairs[:, 0]):
        raise ValueError("every link joins an earlier acquisition to a later "
                         "one")
    if np.unique(pairs, axis=0).shape[0] != pairs.shape[0]:
        raise ValueError("links must not repeat a pair of acquisitions")
    return np.ascontiguousarray(pairs)


def _is_hop3_links(pairs, epochs):
    """Whether the interferograms are the default hop three network."""
    if epochs < _HOP3_EPOCHS or pairs.shape[0] != 3 * epochs - 6:
        return False
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    return bool(np.array_equal(pairs[order], emcf_links(epochs)))


def _hop3_cycles(pairs, epochs):
    """Triangles of the hop three network, as ``(edges, signs)`` loops."""
    position = {}
    for index in range(pairs.shape[0]):
        position[(int(pairs[index, 0]), int(pairs[index, 1]))] = index
    triangles = [(0, 1, 2), (0, 2, 3)]
    for start in range(1, epochs - 3):
        triangles.append((start, start + 2, start + 3))
        triangles.append((start, start + 3, start + 1))
    triangles.append((epochs - 3, epochs - 1, epochs - 2))
    cycles = []
    for first, second, third in triangles:
        edges = []
        signs = []
        # the loop walks first, then second, then third, then back to first
        for start, stop in ((first, second), (second, third),
                            (third, first)):
            if start < stop:
                low, high, sign = start, stop, 1
            else:
                low, high, sign = stop, start, -1
            if (low, high) not in position:
                raise ValueError("the hop three network is missing the link "
                                 "%d-%d" % (low, high))
            edges.append(position[(low, high)])
            signs.append(sign)
        cycles.append((np.asarray(edges, dtype=np.int64),
                       np.asarray(signs, dtype=np.int64)))
    return cycles


def _check_cycles(cycles, nedges):
    """Normalise loops given as pairs of edge indices and signs."""
    checked = []
    for loop in cycles:
        if len(loop) != 2:
            raise ValueError("a cycle is a pair of edge indices and signs")
        edges = np.asarray(loop[0], dtype=np.int64).reshape(-1)
        signs = np.asarray(loop[1], dtype=np.int64).reshape(-1)
        if edges.size != signs.size:
            raise ValueError("a cycle needs one sign per edge")
        if edges.size < 3:
            raise ValueError("a cycle needs at least three edges")
        if np.any(edges < 0) or np.any(edges >= nedges):
            raise ValueError("an edge index of a cycle is out of range")
        if np.unique(edges).size != edges.size:
            raise ValueError("a cycle must not use an edge twice")
        if np.any(np.abs(signs) != 1):
            raise ValueError("a sign of a cycle must be 1 or -1")
        checked.append((np.ascontiguousarray(edges),
                        np.ascontiguousarray(signs)))
    return checked


def _temporal_residues(loops, block):
    """Whole turn circulation of every loop over a block of image arcs."""
    residues = np.empty((len(loops), block.shape[1]), dtype=np.int64)
    for row, (edges, signs) in enumerate(loops):
        circulation = signs @ block[edges]
        residues[row] = np.rint(circulation / _TWO_PI).astype(np.int64)
    return residues


def _form_interferograms(stack, pairs):
    """Wrapped difference of phase of every pair of acquisitions."""
    return _wrap(stack[pairs[:, 1]] - stack[pairs[:, 0]])


def _arc_steps(phase):
    """Wrapped difference of every arc, one row per interferogram."""
    blocks = []
    for axis in range(1, phase.ndim):
        block = _wrap(np.diff(phase, axis=axis))
        blocks.append(block.reshape(phase.shape[0], -1))
    return np.concatenate(blocks, axis=1)


def _arc_weights(confidence):
    """Confidence of every arc of one interferogram, in edge order."""
    blocks = []
    for axis in range(confidence.ndim):
        block = _axis_confidence(confidence, axis)
        blocks.append(block.reshape(-1))
    return np.concatenate(blocks)


def _jump_blocks(shape, flow):
    """Lay a flat flow out as one array of whole turns per axis."""
    blocks = []
    offset = 0
    for axes in _edge_shapes(shape):
        size = int(np.prod(axes))
        blocks.append(flow[offset:offset + size].reshape(axes))
        offset += size
    return blocks
