"""Space-time phase unwrapping after StaMPS.

The other methods of this package unwrap one interferogram at a time.  This
module unwraps a stack of interferograms of the same scene, that is, the
measurements of the same pixels at several acquisition times.  The extra
dimension is not a third axis of the image grid.  It is the time axis of a
series of measurements of one arc, and it is used to predict what the
unwrapped phase difference of that arc should be, which turns a wrapped
measurement into something close to a measurement of the error.

The problem
-----------
An interferogram measures the phase difference of two samples of a scene,
each of them wrapped into one turn.  Unwrapping chooses, for every pair of
neighbouring samples, the whole number of turns that the measurement could
not record.  Those pairs are the *arcs* of the grid, and the only constraint
on the whole numbers is that their signed sum around every loop of the grid
matches the residue of the wrapped differences around that loop.

A stack of interferograms of one scene adds redundancy in time: all the
interferograms see the same scatterers, so the phase of an arc evolves
smoothly from one acquisition to the next, and that evolution can be
estimated from the whole stack even though every single measurement of it is
wrapped.  StaMPS unwraps such a stack in this way and calls the result
space-time unwrapping [1].  The idea does not replace the integer problem of
one interferogram, it supplies it with a prior.

The method in one paragraph
---------------------------
For every arc and every observation a prior is built: a prediction of the
unwrapped phase difference of that arc, and a variance for the error of the
prediction.  The prediction comes from the temporal evolution of the wrapped
arc differences, smoothed with a Gaussian window in time and, in the
interferogram mode, unwrapped along the time axis by adding up its principal
changes.  Every observation is then unwrapped on its own as an integer
network flow: the cheapest set of whole turns that keeps the corrected arc
differences near their priors all over the grid.  Time does not add edges to
that network.  It only tells every arc where the flow wants to be, and what
it costs to leave.

The temporal prediction
-----------------------
The interferogram mode is the one of [1].  For observation ``i`` of an arc,
with the measured arc differences ``steps`` and the acquisition days ``day``
of the observations:

1. the Gaussian weights of the observations around it are

       delta[t] = day[i] - day[t]
       w[t] = exp(-delta[t] ** 2 / (2 * time_win ** 2))

   normalised so that they sum to one, so the window is a weighted average
   rather than a filter that loses the level;

2. the measurements are averaged as complex numbers, which is a weighted
   average of the phase that the wrapping cannot spoil, and that average
   gives a reference phase

       mean = sum(w * exp(1j * steps))      phi = angle(mean)

3. the measurements are brought next to the reference, ``residual = wrap
   (steps - phi)``, and a straight line ``a + b * delta`` is fitted to them,
   weighted by ``w``, through the normal equations.  Only the intercept
   ``a`` is kept, exactly as in [1].  The slope would describe a change of
   the arc phase per unit time, that is a deformation rate, and the code of
   [1] deliberately leaves it out of the prior;

4. the prediction for observation ``i`` is ``phi + a``, one value per arc.

Those predictions are changes of one arc phase history between neighbouring
observations, not the history itself, so they are turned into a series by
adding up their principal changes,

    series[0] = wrap(angle[0])
    series[t] = series[t - 1] + wrap(angle[t] - angle[t - 1])

and its level is pinned afterwards, as described under *Gauge* below.

The series mode is the small baseline variant of [2].  Here the caller
supplies a design matrix ``D`` that relates the phase of the epochs of the
scene to the measurements: one row per observation, one column per epoch, so
that a pair with a one day baseline has a row with ``-1`` in the column of
its earlier epoch and ``+1`` in the column of its later one.

1. one series per arc is fitted to the whole stack at once,

       s[1:] = least squares solution of ``D[:, 1:] @ s = steps``
       s[0] = 0

   the first epoch is the reference, which is why its coefficient is fixed
   to zero.  This is the substitution ``G(:, 2:end) \\ angle(dph_space)'``
   of [2], where the first column of ``G`` is a column of ones;

2. the series of every arc is smoothed in time with the same Gaussian
   weights, now taken over the epochs rather than over the observations;

3. the prediction for an observation is the projection of the smoothed
   series, ``smooth = D @ s``.

The prior in whole turns
------------------------
Let ``k`` be the whole number of turns added to an arc difference, so that
the corrected difference is ``steps + 2 pi k``.  The prediction ``smooth``
and the residual of the prediction are related to the measurement by

    smooth = steps + 2 pi L + noise

which defines the integer ``L`` and makes the residual the principal value
``noise = wrap(steps - smooth)``.  The error of the corrected difference,
measured in turns against the prediction, is then

    (steps + 2 pi k - smooth) / (2 pi) = k - L - noise / (2 pi)

which vanishes for

    k = L + noise / (2 pi) = (smooth - steps) / (2 pi)

so the whole prior is a target number of turns and a variance:

    ==============  ============================================  =============
    quantity        definition                                    unit
    ==============  ============================================  =============
    ``steps``       wrapped difference of the two samples         radians
    ``smooth``      prediction of that difference                 radians
    ``noise``       ``wrap(steps - smooth)``                      radians
    ``centre``      ``(smooth - steps) / (2 pi)``                 turns
    ``variance``    sample variance of ``noise`` / (2 pi) ** 2    turns squared
    ``weight``      confidence of the arc, from the caller        none
    ``curvature``   ``weight`` divided by ``variance``            per turn^2
    ==============  ============================================  =============

The variance is estimated over the observations, so it is one number per arc
for the whole stack, while the target number of turns is one number per arc
and observation.  It is the sample variance, normalised by the number of
observations minus one, which is the statistic the costs of [1] are computed
from.

The cost of one arc
-------------------
The price of an arc is the squared error of the correction in turns, divided
by the variance and multiplied by the weight of the arc, which is

    curvature * (k - centre) ** 2

that is :class:`parvaneh.graph_flow.QuadraticCost`.  It is convex, so the
cheapest set of whole turns is found exactly by the network flow of
:func:`parvaneh.graph_flow.curl_flow`, and the same arc cost applies
independently to every observation.

SNAPHU, the solver behind [1] and [2], describes the same model in a cost
file [3]: an entry ``offset`` and an entry ``sigsq`` per arc and
interferogram, with ``centre = -offset / nshortcycle`` and ``variance =
sigsq * costscale / nshortcycle ** 2`` for the constants ``nshortcycle =
200`` and ``costscale = 100`` of [3].  Its price per whole turn is
``(nshortcycle * dk) ** 2 / sigsq``, which is ``n_edges * costscale`` times
ours.  The two objectives are positive multiples of each other, so they
choose the same flows.

The noise gate
--------------
An arc whose prediction error has a large spread is not trusted.  StaMPS
marks it when the sample standard deviation of the noise over the stack
exceeds 1.3 radians, and the arc then loses its statistics: the cost file
cannot store a large variance, because
a zero entry would make SNAPHU fail, so such an arc keeps the smallest entry
the format allows, ``sigsq = 1``, which is the floor of the variance, and
its offset is left at zero.  Reproducing that bookkeeping is what the
``"floor"`` gate does, and it is the default: a gated arc is asked for no
correction at all at the price of the floor variance.  The ``"free"`` gate
is the other reading of the same flag, and drops the cost of a gated arc
instead, which leaves it to the residue constraints alone.

Gauge
-----
The wrapped measurements do not fix the level of the prediction.  Adding a
whole turn to the level of one arc moves its target number of turns by one,
which does not change the cost of that arc alone but does change which sets
of turns satisfy the loops of the grid, so the level is a real part of the
model rather than a free gauge.  Both modes of StaMPS pin it, and so does
this module: the series mode sets the level of the first epoch to zero, and
the interferogram mode anchors its cumulative sum at the acquisitions
nearest to the master, so that the mean of the prediction there is within
half a turn of zero.  The epochs are the smallest positive day of the stack
and, when it is not the first one, the epoch before it as well.

Two dimensions and three
------------------------
The grid of the image must be two dimensional here.  The exact solver of
:func:`parvaneh.graph_flow.curl_flow` needs the dual network of the loops of
the grid, which only the plane has in the dense form the solver uses, and
the space-time idea lives entirely in the temporal prior in any case.  The
three dimensional counterpart is :mod:`parvaneh.flow_nd`, where time is
another axis of the same grid and the loops that the solver must satisfy
include those that wind through time.  There the third dimension adds
constraints; here it only adds information to the prior.

References
----------
[1] A. Hooper, P. Segall and H. Zebker, "Persistent scatterer
interferometry: a new operational model and tools for space-time analysis,"
*Journal of Geophysical Research* 112 (2007) B07407.

[2] A. Hooper, "A multi-temporal InSAR method incorporating both persistent
scatterer and small baseline approaches," *Geophysical Research Letters* 35
(2008) L16302.

[3] C. W. Chen and H. A. Zebker, "Phase unwrapping for large SAR
interferograms: statistical segmentation and generalized network models,"
*IEEE Transactions on Geoscience and Remote Sensing* 40(8) (2002) 1709-1719.

[4] A. Hooper, D. Bekaert, K. Spaans and M. Arikan, "Recent advances in SAR
interferometry time series analysis for measuring crustal deformation,"
*Tectonophysics* 514-517 (2012) 1-13.

[5] C. W. Chen and H. A. Zebker, "Network approaches to two-dimensional
phase unwrapping: intractability and two new algorithms," *Journal of the
Optical Society of America A* 17(3) (2000) 401-414.

Notes
-----
The formulation is taken from the MATLAB sources of StaMPS, which is
released under the GNU General Public License and is used here only as a
description of the method; no code of it is copied, translated or called.
The parts of the pipeline that this module leaves out are listed below.

* The look angle term of the interferogram mode, the ``la_flag`` of [1], is
  not applied.  It accounts for a phase ramp across range that a change of
  the look angle explains, and a caller who measured one can pass it as part
  of ``weight`` in the same way as any other model of the noise.
* The shakiness branch ``scf_flag`` of [1] and [2] is not ported, so the
  ``spread`` argument is the only hook for it and is zero by default.  What
  that branch does on top of the spread, resmoothing the noisy arcs by
  annealing, is not reproduced.
* Arcs whose prior is known in advance, the ``predef_ix`` of the statistical
  cost module, are not ported.
* One gate value serves both modes.  StaMPS narrows the gate to one radian
  in its small baseline mode.
* The Gaussian weights are renormalised over the entries that are used.
  MATLAB skips undefined values inside ``sum`` and ``std`` without
  renormalising; here every array has to be finite instead, and the caller
  expresses that a sample is unknown through ``mask``.
* The linear fit of the interferogram mode solves the normal equations,
  which is what the least squares call of [1] does for a well posed fit.  A
  degenerate window, with only one nonzero weight, falls back to the mean of
  the residuals; MATLAB would return a minimum norm fit there.
* The series fit is the minimum norm least squares solution for a design
  matrix of deficient rank, which is the documented behaviour of the
  backslash operator of MATLAB.
* A full grid gives every arc exactly one loop of each square it belongs to,
  so the occurrence weighting that SNAPHU applies through its ``n_edges``
  count is a constant here and is not applied.
* Every observation keeps its own additive constant per connected group, set
  by ``align``.  The temporal prior constrains the changes of an arc between
  observations, not the overall constant of one observation, so the
  constants of different observations are unrelated, exactly as in [1].
* The anchor of the interferogram mode uses the earliest acquisition of the
  stack when no acquisition is later than the master.  StaMPS selects the
  same acquisition by taking the smallest positive day, which fails on such
  a stack; this module keeps working and anchors at the first acquisition.
* An arc is priced by its variance and by the confidence the caller gave it,
  while StaMPS prices it by its variance alone.  The two agree whenever the
  confidence is one everywhere, which is the case of a full scene with no
  mask and no weight.  The confidence is applied on top of the statistics so
  that a caller with a model of the reliability of a sample can still use it,
  and it is the same factor that the other backends of the package use.
"""

from dataclasses import dataclass

import numpy as np

from .core import _wrap
from .flow_nd import (ALIGN_MODES, _edge_shapes, grid_cycles,
                      grid_edge_index, grid_edges, integrate_edges)
from .graph_flow import QuadraticCost, curl_flow
from .reliability import _axis_confidence, _check_phase, _confidence

_TWO_PI = 2.0 * np.pi

#: Ways of treating an arc whose prediction error was called noise.
GATE_MODES = ("floor", "free")

_EPSILON = 1e-12
_MIN_WEIGHT = 0.0


@dataclass
class SpaceTimeParams:
    """Constants of the temporal prior.

    The defaults are those of StaMPS, which describes them in its manual.
    They describe the acquisition geometry rather than the scene, so the same
    values serve a whole stack.

    Attributes
    ----------
    time_win : float
        Width, in days, of the Gaussian window that smooths an arc in time.
        It is the length over which the phase of an arc is expected to
        evolve in a straight line, so it belongs to the revisit interval of
        the satellite and the duration of the stack.
    noise_gate : float
        Standard deviation, in radians, above which the prediction error of
        an arc is called noise and the arc loses its statistics.
    min_variance : float
        Floor, in turns squared, of the variance of the prior.  It is what a
        cost file entry of one means in SNAPHU, and it exists because a zero
        entry would make that solver fail.
    """

    time_win: float = 180.0
    noise_gate: float = 1.3
    min_variance: float = 0.0025

    def __post_init__(self):
        if not self.time_win > 0.0:
            raise ValueError("time_win must be positive")
        if self.noise_gate < 0.0:
            raise ValueError("noise_gate must not be negative")
        if not self.min_variance > 0.0:
            raise ValueError("min_variance must be positive")


@dataclass
class SpaceTimePriors:
    """The prior that the temporal part of the method builds.

    Every array of arcs is flat and ordered the way
    :func:`parvaneh.flow_nd.grid_edges` lists the arcs of the image grid, so
    the arcs along the first axis of the image come first, then those along
    the second.  The shapes below write ``A`` for that number of arcs, ``T``
    for the number of observations and ``E`` for the number of epochs.

    Attributes
    ----------
    mode : str
        ``"interferogram"`` when the prior was built from the days of the
        observations, ``"series"`` when it was built from a design matrix.
    day : numpy.ndarray
        Days of the epochs, of length ``E``, increasing.
    steps : numpy.ndarray
        Measured arc differences in radians, of shape ``(T, A)``.
    smooth : numpy.ndarray
        Prediction of those differences, of shape ``(T, A)``, with the level
        pinned as described in the module docstring.
    noise : numpy.ndarray
        Residual of the prediction in radians, of shape ``(T, A)``.
    weight : numpy.ndarray
        Confidence of every arc and observation, of shape ``(T, A)``.
    variance : numpy.ndarray
        Variance of the residual in turns squared, of shape ``(T, A)``, at
        least the floor of the parameters and raised by any spread.
    centre : numpy.ndarray
        Whole number of turns that would make an arc agree with the
        prediction, of shape ``(T, A)``.  It is generally not an integer.
    curvature : numpy.ndarray
        Price of one turn of error, of shape ``(T, A)``, that is the weight
        of an arc divided by its variance.
    gated : numpy.ndarray
        Arcs whose prediction error was called noise, of shape ``(A,)``.
    """

    mode: str
    day: np.ndarray
    steps: np.ndarray
    smooth: np.ndarray
    noise: np.ndarray
    weight: np.ndarray
    variance: np.ndarray
    centre: np.ndarray
    curvature: np.ndarray
    gated: np.ndarray


@dataclass
class SpaceTimeInfo:
    """Report of :func:`space_time_unwrap`.

    Counts of nodes, arcs and loops describe one image grid, so they do not
    grow with the number of observations; the pixel count and the objective
    do.

    Attributes
    ----------
    backend : str
        Solver that produced the field.
    mode : str
        Which of the two temporal modes was used.
    shape : tuple of int
        Shape of the image grid.
    observations : int
        Measurements in the stack.
    epochs : int
        Epochs of the temporal model.  In the series mode these are the
        acquisitions the design matrix refers to; in the interferogram mode
        the temporal model is built on the observations themselves, so this
        repeats ``observations``.
    pixels : int
        Samples with a nonzero confidence, counted over the whole stack.
    nodes : int
        Samples of one image grid.
    edges : int
        Arcs of one image grid.
    cycles : int
        Loops that the whole turns had to satisfy.
    residues : int
        Residues of the loops, which are the supplies of the flow problem,
        summed in absolute value over the observations.
    components : int
        Connected groups of usable samples, the largest count over the
        observations, which is the count itself when the mask is the same
        everywhere.
    gated_edges : int
        Arcs whose prediction error was called noise.
    masked_edges : int
        Arcs of weight zero in every observation, whose cost is zero.
    max_jump : int
        Largest number of turns added to a single arc.
    objective : float
        Summed cost over the whole stack, which is dimensionless: the
        squared error of an arc in turns over its variance.
    objective_per_observation : numpy.ndarray
        The same cost for every observation.
    time_win : float
        Window used by the temporal smoothing, in days.
    noise_gate : float
        Threshold used to call the prediction error of an arc noise.
    min_variance : float
        Floor applied to the variance.
    augmentations : int
        Augmentations of the network solver, summed over the observations.
    proven : bool
        Whether the objective is known to be the smallest one, for every
        observation.
    """

    backend: str
    mode: str
    shape: tuple
    observations: int
    epochs: int
    pixels: int
    nodes: int
    edges: int
    cycles: int
    residues: int
    components: int
    gated_edges: int
    masked_edges: int
    max_jump: int
    objective: float
    objective_per_observation: np.ndarray
    time_win: float
    noise_gate: float
    min_variance: float
    augmentations: int
    proven: bool


def space_time_priors(phase, day, *, design=None, mask=None, weight=None,
                      spread=None, params=None, gate="floor"):
    """Build the temporal prior of every arc of a stack of interferograms.

    This is the first half of :func:`space_time_unwrap`, and it is exposed
    because the prior is interesting on its own: it is the measurement of the
    deformation rate of every arc that the wrapped stack contains, and it is
    the object a caller would inspect to decide whether the smoothing window
    and the gate suit a stack.

    Parameters
    ----------
    phase : array_like of float
        Wrapped phase of the stack in radians, of shape ``(observations,
        rows, columns)``.  The first axis is time.
    day : array_like of float
        Acquisition days.  One entry per observation in the interferogram
        mode, one entry per epoch of ``design`` in the series mode.  The
        values have to increase, because the Gaussian window and the
        cumulative sum both read them as a chronology.
    design : array_like of float, optional
        Design matrix of the series mode, of shape ``(observations,
        epochs)``.  Without it the interferogram mode is used.  The first
        column is the reference epoch, whose coefficient is fixed to zero.
    mask : array_like of bool, optional
        Samples to ignore, the same shape as ``phase``.  Every arc that
        touches one of them gets weight zero in that observation, which
        makes it free there.
    weight : array_like of float, optional
        Extra per-sample weight, the same shape as ``phase``.  The default is
        one everywhere.  The weight of an arc is the smaller weight of its
        two ends.
    spread : array_like of float, optional
        Extra variance of the arcs, in the units of the shakiness estimate of
        StaMPS, which is in turns: it is divided by six and added to the
        variance, as the cost file of [3] does.  It may be given per arc or
        per arc and observation.  The default is zero, which is what a run
        without the shakiness branch produces.
    params : SpaceTimeParams, optional
        Constants of the temporal model.
    gate : {"floor", "free"}, optional
        What an arc whose prediction error is called noise should become.
        ``"floor"`` reproduces the cost file of StaMPS, where such an arc is
        asked for no correction at the price of the floor variance.
        ``"free"`` drops its cost instead.

    Returns
    -------
    SpaceTimePriors
        The prior of every arc and observation.

    Raises
    ------
    ValueError
        If a shape, a day, a design matrix, a gate or a parameter is not
        usable.

    Examples
    --------
    >>> import numpy as np                                     # doctest: +SKIP
    >>> phase = np.zeros((8, 16, 16))                          # doctest: +SKIP
    >>> day = np.arange(8.0)                                   # doctest: +SKIP
    >>> priors = space_time_priors(phase, day)              # doctest: +SKIP
    >>> priors.centre.shape                                 # doctest: +SKIP
    (8, 480)
    """
    if gate not in GATE_MODES:
        raise ValueError("gate must be one of %s" % (GATE_MODES,))
    if params is None:
        params = SpaceTimeParams()
    array = _check_phase(phase)
    if array.ndim != 3:
        raise ValueError(
            "the space-time method takes one stack of interferograms, of "
            "shape (observations, rows, columns)")
    observations = array.shape[0]
    if design is None:
        mode = "interferogram"
        model = None
        epoch_day = _check_day(day, observations, "observation")
    else:
        mode = "series"
        model = np.ascontiguousarray(design, dtype=np.float64)
        if model.ndim != 2 or model.shape[0] != observations:
            raise ValueError(
                "design must have one row per observation of phase")
        if model.shape[1] < 2:
            raise ValueError("design must have at least two columns")
        if not np.isfinite(model).all():
            raise ValueError("design must be finite")
        epoch_day = _check_day(day, model.shape[1], "epoch")
    confidence = _confidence(array, mask, weight)
    steps = _arc_steps(array)
    weights = _arc_weights(confidence)
    if model is None:
        smooth = _smooth_interferograms(steps, epoch_day, params.time_win)
    else:
        smooth = _smooth_series(steps, model, epoch_day, params.time_win)
    noise, variance, centre, curvature, gated = _build_priors(
        steps, smooth, weights, spread, params, gate)
    return SpaceTimePriors(
        mode=mode,
        day=epoch_day,
        steps=steps,
        smooth=smooth,
        noise=noise,
        weight=weights,
        variance=variance,
        centre=centre,
        curvature=curvature,
        gated=gated,
    )


def space_time_unwrap(phase, day, *, design=None, mask=None, weight=None,
                      spread=None, params=None, gate="floor", align="zero",
                      max_iter=None, return_info=False):
    """Unwrap a stack of interferograms with a temporal prior.

    Every observation of the stack is unwrapped independently, as an integer
    network flow over the arcs of the image grid.  The arcs are priced by
    :func:`space_time_priors`, which is where the stack is used: the
    prediction of an arc and the variance of its error are shared by the
    whole stack, while the wrapped measurements of the observation being
    solved supply the loop constraints.

    Parameters
    ----------
    phase : array_like of float
        Wrapped phase of the stack in radians, of shape ``(observations,
        rows, columns)``.  The first axis is time.
    day : array_like of float
        Acquisition days, one per observation in the interferogram mode and
        one per epoch of ``design`` in the series mode, increasing.
    design : array_like of float, optional
        Design matrix of the series mode, of shape ``(observations,
        epochs)``.  Its first column is the reference epoch, whose
        coefficient is fixed to zero.  Without it the interferogram mode is
        used, which expects ``day`` to have one entry per observation.
    mask : array_like of bool, optional
        Samples to ignore, the same shape as ``phase``.
    weight : array_like of float, optional
        Extra per-sample weight, the same shape as ``phase``.  The default is
        one everywhere.
    spread : array_like of float, optional
        Extra variance of the arcs, in the units of the shakiness estimate of
        StaMPS, divided by six before it is added.  Zero by default.
    params : SpaceTimeParams, optional
        Constants of the temporal model.
    gate : {"floor", "free"}, optional
        What an arc whose prediction error is called noise should become.
        ``"floor"``, the default, reproduces StaMPS: no correction at the
        price of the floor variance.
    align : {"zero", "mean", "median"}, optional
        How to fix the additive constant of every connected group of an
        observation.  The temporal prior does not relate the constants of two
        observations, so every one of them is fixed on its own.
    max_iter : int, optional
        Augmentation limit of the network solver.  A limit too small to
        balance the supplies raises ``RuntimeError``.
    return_info : bool, optional
        Also return a :class:`SpaceTimeInfo`.

    Returns
    -------
    numpy.ndarray
        Unwrapped phase in radians, of the same shape as ``phase``, ``NaN``
        where the confidence is zero.
    SpaceTimeInfo
        Only when ``return_info`` is true.

    Raises
    ------
    ValueError
        If an argument is unknown or ill shaped, if ``day`` does not
        increase, or if ``design`` does not match the stack.
    RuntimeError
        If ``max_iter`` stops the network solver before the supplies of an
        observation are balanced.

    Notes
    -----
    The cost is convex, so the flow is the exact minimum of the prior cost
    of every observation, and ``proven`` in the report says whether the
    solver reached optimality rather than stopping at a limit.  A limit that
    it cannot reach raises, so a returned field always holds a proven
    optimum of the stated objective.

    Examples
    --------
    >>> import numpy as np                                 # doctest: +SKIP
    >>> phase = np.zeros((8, 16, 16))                      # doctest: +SKIP
    >>> day = np.arange(8.0)                               # doctest: +SKIP
    >>> field = space_time_unwrap(phase, day)               # doctest: +SKIP
    """
    if align not in ALIGN_MODES:
        raise ValueError("align must be one of %s" % (ALIGN_MODES,))
    if params is None:
        params = SpaceTimeParams()
    array = _check_phase(phase)
    priors = space_time_priors(array, day, design=design, mask=mask,
                               weight=weight, spread=spread, params=params,
                               gate=gate)
    frame = array.shape[1:]
    confidence = _confidence(array, mask, weight)
    vertices, tail, head = grid_edges(frame)
    nodes = int(vertices.size)
    cycles = grid_cycles(frame, grid_edge_index(frame))
    field = np.empty(array.shape, dtype=np.float64)
    per_observation = np.empty(array.shape[0], dtype=np.float64)
    augmentations = 0
    residues = 0
    max_jump = 0
    components = 0
    proven = True
    for index in range(array.shape[0]):
        cost = QuadraticCost(priors.curvature[index], priors.centre[index])
        flow, result = curl_flow(nodes, tail, head, priors.steps[index],
                                 cost=cost, cycles=cycles, method="flow",
                                 max_iter=max_iter, return_info=True)
        per_observation[index] = float(np.sum(cost.evaluate(flow)))
        augmentations += int(result.augmentations)
        residues += int(result.residues)
        proven = proven and bool(result.proven)
        if flow.size:
            max_jump = max(max_jump, int(np.abs(flow).max()))
        field[index], labels = integrate_edges(
            array[index], confidence[index], _jump_blocks(frame, flow),
            align=align)
        components = max(components, int(labels.max()) + 1)
    info = SpaceTimeInfo(
        backend="space-time-flow",
        mode=priors.mode,
        shape=tuple(int(size) for size in frame),
        observations=int(array.shape[0]),
        epochs=int(priors.day.size),
        pixels=int(np.count_nonzero(confidence > _MIN_WEIGHT)),
        nodes=nodes,
        edges=int(tail.size),
        cycles=int(len(cycles)),
        residues=residues,
        components=components,
        gated_edges=int(np.count_nonzero(priors.gated)),
        masked_edges=int(np.count_nonzero(
            np.all(priors.weight <= _MIN_WEIGHT, axis=0))),
        max_jump=max_jump,
        objective=float(per_observation.sum()),
        objective_per_observation=per_observation,
        time_win=float(params.time_win),
        noise_gate=float(params.noise_gate),
        min_variance=float(params.min_variance),
        augmentations=augmentations,
        proven=proven,
    )
    if return_info:
        return field, info
    return field


def _check_day(day, length, what):
    """Validate a vector of acquisition days and return it as an array."""
    values = np.ascontiguousarray(day, dtype=np.float64)
    if values.ndim != 1 or values.size != length:
        raise ValueError("day must have one entry per %s" % (what,))
    if not np.isfinite(values).all():
        raise ValueError("day must be finite")
    if values.size > 1 and np.any(np.diff(values) <= 0.0):
        raise ValueError("day must increase")
    return values


def _arc_steps(phase):
    """Wrapped difference of every arc, one row per observation."""
    blocks = []
    for axis in range(1, phase.ndim):
        block = _wrap(np.diff(phase, axis=axis))
        blocks.append(block.reshape(phase.shape[0], -1))
    return np.concatenate(blocks, axis=1)


def _arc_weights(confidence):
    """Confidence of every arc, one row per observation."""
    blocks = []
    for axis in range(1, confidence.ndim):
        block = _axis_confidence(confidence, axis)
        blocks.append(block.reshape(confidence.shape[0], -1))
    return np.concatenate(blocks, axis=1)


def _time_weights(day, index, time_win):
    """Normalised Gaussian weights around one acquisition day."""
    delta = day[index] - day
    weights = np.exp(-(delta * delta) / (2.0 * time_win * time_win))
    total = float(weights.sum())
    if total <= 0.0:
        weights = np.zeros(day.size, dtype=np.float64)
        weights[index] = 1.0
        return weights, delta
    return weights / total, delta


def _time_unwrapped(angles):
    """Add up the principal changes of a series of arc differences."""
    steps = np.empty_like(angles)
    steps[0] = _wrap(angles[0])
    steps[1:] = _wrap(np.diff(angles, axis=0))
    return np.cumsum(steps, axis=0)


def _reference_indices(day):
    """Epochs nearest to the master, chosen the way StaMPS chooses them."""
    positive = np.flatnonzero(day > 0.0)
    if positive.size == 0:
        return np.array([int(np.argmax(day))], dtype=np.int64)
    index = int(positive[np.argmin(day[positive])])
    if index > 0:
        return np.array([index - 1, index], dtype=np.int64)
    return np.array([index], dtype=np.int64)


def _anchor(smooth, day):
    """Pin the level of every arc near the epochs closest to the master."""
    index = _reference_indices(day)
    shift = np.mean(smooth[index], axis=0)
    return smooth - (shift - _wrap(shift))


def _smooth_interferograms(steps, day, time_win):
    """Predict every arc from the days of the observations."""
    angles = np.empty_like(steps)
    for index in range(steps.shape[0]):
        weights, delta = _time_weights(day, index, time_win)
        mean = weights @ np.exp(1j * steps)
        offset = np.angle(mean)
        residual = _wrap(steps - offset)
        s1 = float(weights @ delta)
        s2 = float(weights @ (delta * delta))
        b0 = weights @ residual
        b1 = (weights * delta) @ residual
        det = s2 - s1 * s1
        if abs(det) > _EPSILON:
            # The weights sum to one, so the first normal equation is the
            # mean of the residuals and the determinant only involves the
            # spread of the days that carry weight.
            b0 = (s2 * b0 - s1 * b1) / det
        angles[index] = offset + b0
    return _anchor(_time_unwrapped(angles), day)


def _smooth_series(steps, design, day, time_win):
    """Predict every arc from a design matrix and the days of the epochs."""
    series = np.zeros((design.shape[1], steps.shape[1]), dtype=np.float64)
    series[1:] = np.linalg.lstsq(design[:, 1:], steps, rcond=None)[0]
    smooth = np.empty_like(series)
    for index in range(series.shape[0]):
        weights, _ = _time_weights(day, index, time_win)
        smooth[index] = weights @ series
    return design @ smooth


def _spread_term(spread, shape):
    """Extra variance of every arc, in turns squared."""
    if spread is None:
        return 0.0
    values = np.ascontiguousarray(spread, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("spread must be finite")
    try:
        return np.broadcast_to(np.abs(values) / 6.0, shape)
    except ValueError:
        raise ValueError("spread must broadcast to the shape of the arcs")


def _build_priors(steps, smooth, weights, spread, params, gate):
    """Turn a prediction into the target and price of every arc."""
    noise = _wrap(steps - smooth)
    variance = np.maximum(np.var(noise, axis=0, ddof=1) / (_TWO_PI ** 2),
                          params.min_variance)
    variance = variance[None, :] + _spread_term(spread, noise.shape)
    centre = (smooth - steps) / _TWO_PI
    curvature = weights / variance
    gated = np.std(noise, axis=0, ddof=1) > params.noise_gate
    if gate == "floor":
        centre = np.where(gated[None, :], 0.0, centre)
        curvature = np.where(gated[None, :],
                             weights / params.min_variance, curvature)
    else:
        curvature = np.where(gated[None, :], 0.0, curvature)
    return noise, variance, centre, curvature, gated


def _jump_blocks(shape, flow):
    """Lay a flat flow out as one array of whole turns per axis."""
    blocks = []
    offset = 0
    for axes in _edge_shapes(shape):
        size = int(np.prod(axes))
        blocks.append(flow[offset:offset + size].reshape(axes))
        offset += size
    return blocks
