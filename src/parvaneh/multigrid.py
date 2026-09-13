"""Multigrid solution of the weighted least-squares unwrapping problem.

This module solves the same problem as :func:`parvaneh.core.unwrap`: it looks
for the field ``phi`` whose neighbouring differences are as close as possible to
the wrapped differences of ``phase``, with a nonnegative confidence on every
step.  The answer is the same; only the way of reaching it differs.  Where
:mod:`parvaneh.core` solves the normal equations with conjugate gradients
preconditioned by a discrete cosine transform, this module descends a hierarchy
of progressively coarser grids, the technique Pritt (1996) brought to
interferometric radar.

Why a hierarchy of grids
------------------------

Simple relaxation, say Gauss--Seidel, removes the rough part of an error
quickly and the smooth part very slowly.  After a few sweeps the error is
smooth, and one sweep only moves information one sample per step, so a bump that
spans the whole image needs a great many sweeps.  Multigrid turns that weakness
into a plan:

1. Smooth on the fine grid, which removes the sample-to-sample roughness of the
   error.
2. Restrict the remaining error -- its residual -- to a grid with half the
   samples along every axis.  A bump that covered 64 samples now covers 32, so
   on that grid it is *not* smooth any more and the smoother can bite again.
3. Solve the coarse problem by the same two steps, recursively, and add the
   correction back with interpolation.

One pass from the finest grid down to the coarsest and back is a V-cycle.  Its
cost is a constant multiple of a single sweep on the finest grid, because the
coarser grids shrink geometrically: in two dimensions the whole hierarchy holds
about a third of the samples of the finest grid, so smoothing once on every level
costs some four thirds of a fine sweep, and the default of two sweeps a side
costs a little over five.  The number of cycles needed for a given tolerance does
not grow with the image size either, which is exactly what plain relaxation
fails to achieve.  Measured on a noisy synthetic field (``sigma = 0.6`` rad, a
tolerance of ``1e-10``), the cycle count hardly moves: twelve cycles at
``16 x 16``, thirteen at ``128 x 128`` and ``256 x 256``, twelve at
``512 x 512``.  Plain relaxation of the finest grid never reaches that tolerance:
the same 500-sweep budget leaves a relative residual of ``2.5e-06`` at
``16 x 16`` and ``3.9e-04`` at ``512 x 512``.

The operator on each grid
-------------------------

Let :math:`L_\\ell` be the weighted discrete Laplacian of level :math:`\\ell`,
the operator :func:`parvaneh.core._apply_q_numpy` applies, built on a grid whose
samples are one unit apart.  The finest grid solves

.. math::

    L_0 \\phi_0 = f_0 ,

with :math:`f_0` the divergence of the wrapped differences, because that is
exactly the normal equation of the weighted least-squares problem.  One unit on
level :math:`\\ell` is :math:`2^\\ell` pixels of the original image, and a second
difference over a spacing :math:`h` estimates :math:`h^2` times the second
derivative.  The operator of level :math:`\\ell` that acts on the *same*
physical quantity as :math:`L_0` is therefore :math:`4^{-\\ell} L_\\ell`, and
each level carries the scalar factor :math:`c_\\ell = 4^{-\\ell}`.  A V-cycle is

.. math::

    \\phi_\\ell \\leftarrow \\phi_\\ell + P_\\ell \\phi_{\\ell+1},
    \\qquad c_{\\ell+1} L_{\\ell+1} \\phi_{\\ell+1}
    = R_\\ell \\left( f_\\ell - c_\\ell L_\\ell \\phi_\\ell \\right) ,

where :math:`R_\\ell` restricts and :math:`P_\\ell` prolongs.  The factor 4 per
level follows from the spacing alone and is the same in any dimension, which is
one reason this formulation extends to three dimensions without change:
:mod:`parvaneh.multigrid` loops over the axes, so a 3-D volume, a 2-D image and
a 1-D profile all go through the same code.

The transfer operators
----------------------

Both transfer operators are one-dimensional rules applied axis by axis, so the
same code serves a profile, an image and a volume.  Prolongation is linear
interpolation: the coarse samples land on the odd samples of the finer grid, the
even samples copy a coarse sample, and each odd sample between two coarse ones
averages them.  Restriction is the *transpose* of that interpolation, scaled by
one half per axis, which is the variational choice of multigrid theory
(Trottenberg et al. 2001, section 5.3): a fine sample feeds a coarse sample in
proportion to the influence that coarse sample has on it.  Away from the border
the resulting stencil is the full weighting :math:`\\tfrac14 [1\\; 2\\; 1]`, so
in two dimensions its tensor product is the :math:`\\tfrac{1}{16}`-weighted
nine-point restriction of Pritt (1996) and the prolongation is the bilinear
interpolation of his Figure 5.

The border needs a decision, because the stencils ask for neighbours that do not
exist.  Along each axis the coarse grid holds ``ceil(n / 2)`` samples and its
samples sit on the odd samples of the fine grid.  Prolongation repeats the last
coarse sample rather than inventing a slope, which is the discrete form of the
insulating boundary of the least-squares problem: a correction that is constant
along the edge stays constant along it.  Its transpose then makes the first and
last rows of the restriction sum to :math:`\\tfrac34` and :math:`\\tfrac54`
instead of one -- the fine samples sit half a cell inside the faces -- and the
next section deals with the consequence.

Weights on the coarse grids
---------------------------

The confidence of a pixel is a property of the finest grid only.  A coarse sample
takes the *row-normalised* restriction of the fine confidences,
:math:`R w / R \\mathbf{1}`, through the same :math:`R` the residuals go through.
The division is not cosmetic.  Plain restriction does not reproduce a constant:
applied to a field of ones it returns ``0.5625`` at the corner of an even-sized
grid and ``1.5625`` at the sample diagonally farthest from it.  Since the coarse
step weights are the squares of these numbers, every further level would darken
its own boundary again, and the coarse grid would be solving a *different*
equation from the fine one -- with weaker coupling at the edge -- so the
correction it hands back would be consistently too small.  The smoother cannot
repair what the coarse grid under-corrects, and the rate of a cycle would then
climb with the image size instead of staying flat.  With the division in place
that rate stays between ``0.17`` and ``0.23`` a cycle for every size from
``16 x 16`` to ``512 x 512``.

As in Pritt (1996), the coarse edge weights are then re-derived from those
averaged node weights with the same rule the finest grid uses,
:func:`parvaneh.core._edges`: the weight of a step is the smaller of the squared
confidences of its two endpoints.  Re-discretising instead of projecting the
operator algebraically ("Galerkin" coarsening, as in Trottenberg et al. 2001,
section 7) keeps every level a plain five-point weighted Laplacian, which is what
makes the cheap red-black smoother of the next paragraph possible.  In one
dimension the re-discretised operator and the projected one agree exactly,
because :math:`R L P = L / 4` sample by sample; in two and three dimensions the
projected operator also couples diagonal neighbours, so the five-point operator
only approximates it.  The approximation is the same on every level, which is why
the rate of the cycle stays nearly independent of the image size for a smooth
coefficient field.  A discontinuous one is another matter; the section "Where the
hierarchy stops helping" measures what happens there.

Geometric coarsening changes the *rate* of the iteration but not its fixed
point: a V-cycle still stops when the residual of the finest grid, which is the
gradient of the least-squares cost, vanishes.  An unweighted problem
(``weight=None, mask=None``) is the classical unweighted multigrid of Ghiglia
and Pritt (1998, chapter 5); weights or a mask make it the weighted multigrid
they recommend for real data.

The smoother
------------

Every grid is relaxed by red-black Gauss--Seidel.  Colour the samples like a
chessboard; red samples have only black neighbours, so all red updates of a
half-sweep may be written as one array expression that leaves the black values
untouched, and the black half-sweep follows.  A sample with zero diagonal, which
happens only when every incident step has weight zero, is left alone.  The
reported residual is :math:`\\| f_0 - L_0 \\phi_0 \\|_2` relative to
:math:`\\| f_0 \\|_2`, the quantity :func:`parvaneh.core.unwrap` reports too.

Where the hierarchy stops helping
---------------------------------

A cycle is the right answer for smooth coefficients, and much less so for
discontinuous ones.  The measurements below use a noisy synthetic scene
(``sigma = 0.6`` rad), two sweeps of smoothing a side, a tolerance of ``1e-10``
and a limit of 200 cycles; the second number in a cell is the relative residual
when the cycle did not converge::

    weight field                        128 x 128           256 x 256
    uniform                             13 cycles, 9e-11    13 cycles, 5e-11
    zero-weight lines (a mask)          79 cycles, 9e-11   155 cycles, 1e-10
    every 32nd line at 1e-2            200 cycles, 2e-06   200 cycles, 1e-06
    16 x 16 blocks at 2e-3             200 cycles, 2e-07   200 cycles, 1e-07

Four different behaviours are behind those numbers.

* **Uniform weights** are the textbook case: the cycle count does not grow with
  the image.  Twelve cycles at ``16 x 16``, thirteen at ``128 x 128`` and
  ``256 x 256``, twelve at ``512 x 512`` -- the same work for a thousand times
  the samples.  That is the hierarchy and not the smoother, because the same
  500-sweep budget spent on the finest grid alone leaves ``2.5e-06`` at
  ``16 x 16`` and ``3.9e-04`` at ``512 x 512``.  One cycle costs about ``5.3``
  finest-grid sweeps once the hierarchy is seven levels deep, so thirteen cycles
  is about seventy sweeps.

* A **mask**, that is zero-weight lines a couple of samples wide cutting the
  image into islands, still converges, but it needs more cycles as the image
  grows -- 16, 79 and 155 cycles at ``16 x 16``, ``128 x 128`` and ``256 x
  256``, and at ``512 x 512`` the run stops just short of the tolerance
  (``8e-09``) -- and it depends on how much smoothing each cycle does: 125, 79,
  58 and 40 cycles at ``128 x 128`` for one, two, four and ten sweeps a side.
  The smoother is doing work the hierarchy was supposed to do.

* **Thin high-contrast lines** are the real failure.  With every thirty-second
  row and column set to ``1e-2`` the residual freezes: after the first few
  cycles it falls by ``0.96`` a cycle and keeps that pace, so 200 cycles leave
  ``2e-06``.  Coarsening *re-derives* the edge weights from the averaged node
  weights, so a coarse grid carries the same weak lines as the grid above it,
  and the part of the error that lives along those lines never looks smooth on
  any level.  A blocky field at ``2e-3`` behaves the same way, stalling at
  ``2e-07`` on ``128 x 128``.  Neither stall is cosmetic: against
  :func:`parvaneh.core.unwrap`, which converges on both fields in 38 to 60
  iterations, the stalled answer is off by ``1.94`` against ``1.60`` rad RMS at
  ``128 x 128`` and ``1.76`` against ``1.34`` at ``256 x 256`` for the thin
  field, and by ``3.58`` against ``1.62`` and ``3.16`` against ``1.14`` for the
  blocky one.  A cycle that stalls is not a slow road to the same answer; it is
  a different, worse answer.

* **One sweep a side is not enough**, which is why the smoother defaults to two.
  On the blocky field a single sweep a side *diverges*: the relative residual
  reaches ``1e+44`` after 200 cycles, and a diverging cycle is worse than a
  stalling one.  Plain relaxation of the finest grid with
  no hierarchy at all is no substitute: its whole 500-sweep budget leaves
  ``1.8e-03`` (uniform), ``2.2e-03`` (mask), ``5.7e-04`` (thin) and ``4.6e-07``
  (blocky) at ``128 x 128``.  The single case where it wins outright in these
  measurements is a ``16 x 16`` mask, where it converges in 226 sweeps against
  16 cycles of the hierarchy.

The remedy known from multigrid theory is Galerkin, or algebraic, coarsening
(Trottenberg et al. 2001, section 7): project the *operator* onto the coarse
grid, :math:`L_{\\ell+1} = R L_\\ell P`, instead of projecting the coefficients
and re-discretising.  The projection couples diagonal neighbours, so a coarse
level is no longer a five-point stencil and the one-expression red-black
smoother of this module does not apply to it.  That is a different
implementation, not a parameter, and it is left to later work.

When the weights span several orders of magnitude, :func:`parvaneh.core.unwrap`
is the better tool, and the measurements say so plainly: its transform
preconditioner is built from a problem with constant coefficients, so the
contrast does not touch it.  On these fields it converges in one iteration with
uniform weights and in 38 to 60 iterations on the others, between ten and a
hundred times faster than the cycles above, and on the mask field it agrees with
this module to ``2e-08`` rad on the samples that carry weight while choosing a
different interpolation for the samples inside the cut.  What this module offers
instead is a cycle count that does not grow with the image for well-behaved
weights, agreement with the least-squares answer to the tolerance asked for, and
a formulation that extends to three dimensions, where a fast transform of the
same kind does not exist.  It is an alternative solver with a known weak spot,
not a replacement.

Every number in this section and in the two sections above comes from
``benchmarks/benchmark_multigrid.py``, which generates the scenes in memory and
whose module docstring lists the commands that produce them.

The singular, disconnected case
-------------------------------

The least-squares problem has no unique solution without a further condition:
adding a constant to every sample changes no difference, and a region cut off
from the rest by the mask may take any constant.  As in :mod:`parvaneh.core`,
the returned field has zero mean.  Zero-mean projection of the residual before
it is restricted and of every coarse right-hand side keeps the hierarchy
consistent, because a constant right-hand side has no solution on a grid with
insulating boundaries.  The component that lives on a masked-off island is
invisible to the residual and keeps whatever the smoother gave it (zero, since
the iteration starts from zero), the same freedom the transform solver leaves.

References
----------

* M. D. Pritt, "Phase unwrapping by means of multigrid techniques for
  interferometric SAR," *IEEE Transactions on Geoscience and Remote Sensing*
  34(3), 728--738, 1996.  doi:10.1109/36.499752.  The V-cycle, full-weighting
  restriction, bilinear prolongation, red-black smoother and the averaging of
  weights onto coarse grids.
* D. C. Ghiglia and M. D. Pritt, *Two-Dimensional Phase Unwrapping: Theory,
  Algorithms, and Software*, Wiley, 1998.  Chapter 5 places multigrid among the
  minimum-norm methods and discusses unweighted against weighted multigrid.
* W. L. Briggs, V. E. Henson and S. F. McCormick, *A Multigrid Tutorial*,
  2nd ed., SIAM, 2000.  The tutorial from which the V-cycle algorithm of this
  module is adapted.
* U. Trottenberg, C. W. Oosterlee and A. Schuller, *Multigrid*, Academic Press,
  2001.  Variable coefficients and the transfer operators.
* W. H. Press, S. A. Teukolsky, W. T. Vetterling and B. P. Flannery,
  *Numerical Recipes*, 3rd ed., Cambridge University Press, 2007, section 19.6,
  "Multigrid methods for boundary value problems".

Examples
--------

A ramp has no residues, so unwrapping it is exactly the least-squares solution,
and multigrid reproduces the transform solver:

>>> import numpy as np
>>> from parvaneh.core import unwrap
>>> ramp = 0.35 * np.add.outer(np.arange(8.0), np.arange(8.0))
>>> phase = (ramp + np.pi) % (2.0 * np.pi) - np.pi
>>> result = multigrid_unwrap(phase, max_cycles=100)
>>> bool(np.abs(result - unwrap(phase)).max() < 1e-5)
True
"""

from dataclasses import dataclass
import time

import numpy as np

from .core import _axis_slice, _divergence, _edges
from .reliability import _check_phase, _confidence

__all__ = ["MultigridInfo", "multigrid_unwrap"]


@dataclass
class MultigridInfo:
    """Bookkeeping for one multigrid run.

    Attributes
    ----------
    backend : str
        Engine that ran the iteration; always ``"numpy"`` for now.
    levels : int
        Number of grids in the hierarchy, one for the image itself.
    coarsest : tuple of int
        Shape of the coarsest grid.
    cycles : int
        V-cycles performed.
    sweeps : int
        Red-black Gauss--Seidel sweeps performed, summed over all levels and
        cycles.  This is the work measure to compare against other solvers.
    relative_residual : float
        ``‖f₀ - L₀φ₀‖₂ / ‖f₀‖₂`` after the last cycle.
    converged : bool
        Whether ``relative_residual`` reached ``tol``.
    seconds : float
        Wall-clock time of the iteration.
    residuals : tuple of float
        Relative residual after each cycle, oldest first.  A V-cycle is a
        contraction, so these normally fall by a roughly constant factor.
    """

    backend: str
    levels: int
    coarsest: tuple
    cycles: int
    sweeps: int
    relative_residual: float
    converged: bool
    seconds: float
    residuals: tuple


@dataclass
class _Level:
    """One grid of the hierarchy, finest first.

    ``edges``, ``diagonal`` and ``factor`` describe the operator; ``rhs`` and
    ``phi`` are the right-hand side and the current approximation, kept here so
    that a V-cycle can fill them in place instead of allocating.
    """

    edges: tuple
    diagonal: np.ndarray
    factor: float
    rhs: np.ndarray
    phi: np.ndarray


def _edge_weights(nodes):
    """Weights of the steps between neighbours, one array per axis.

    This is the rule of :func:`parvaneh.core._edges` applied to one field: the
    weight of a step is the smaller of the squared weights of its two ends, so a
    zero weight cuts the connection.
    """
    squared = np.asarray(nodes, dtype=np.float64) ** 2
    weights = []
    for axis in range(squared.ndim):
        low = _axis_slice(squared, axis, slice(0, -1))
        high = _axis_slice(squared, axis, slice(1, None))
        weights.append(np.minimum(low, high))
    return tuple(weights)


def _operator(field, edges):
    """Weighted discrete Laplacian of ``field``, the ``L`` of one grid."""
    return _divergence(tuple(weight * np.diff(field, axis=axis)
                             for axis, weight in enumerate(edges)))


def _diagonal(edges, shape):
    """Diagonal of the operator, minus the sum of the incident step weights.

    :func:`_divergence` is ``-Dᵀ`` for the forward-difference operator ``D``, so
    the operator of one grid is the *negative* of the usual weighted Laplacian
    and its diagonal is negative.  That sign is what the steepest-step update of
    Gauss--Seidel needs; a positive diagonal would make every sweep overshoot.
    """
    diagonal = np.zeros(shape, dtype=np.float64)
    for axis, weight in enumerate(edges):
        _axis_slice(diagonal, axis, slice(0, -1))[...] -= weight
        _axis_slice(diagonal, axis, slice(1, None))[...] -= weight
    return diagonal


def _checkerboard(shape):
    """``False`` on the red samples, ``True`` on the black ones.

    A sample is red when the sum of its indices is even, so every neighbour of a
    red sample is black.
    """
    black = np.zeros(shape, dtype=bool)
    for axis, size in enumerate(shape):
        view = [1] * len(shape)
        view[axis] = size
        black ^= (np.arange(size).reshape(view) % 2).astype(bool)
    return black


def _restrict_axis(field, axis):
    """Halve ``axis`` by the transpose of :func:`_prolong_axis`.

    Restriction is taken as ``R = P^T / 2`` rather than as an independently
    invented stencil, which is the *variational* choice of multigrid theory.  The
    payoff is a coarse-grid operator that matches the fine one: along a single
    axis with constant coefficient the identity ``R L P = L / 4`` holds sample by
    sample, which is the ``4 ** -level`` scaling that :func:`_build` gives each
    level operator.  An independently invented stencil instead leaves a mismatch
    between the coarse equation and the fine one, and the part of the correction
    that the mismatch loses is never removed -- the smoother has to clean up the
    smooth error on its own, which costs sweeps and buys nothing, so the V-cycle
    rate drifts upwards with the grid size.

    In two and three dimensions the identity holds only in the sense of the
    closest five-point stencil: the exact product ``R L P`` couples diagonal
    neighbours as well, and this module drops those terms when it re-derives a
    coarse operator from averaged coefficients (see
    :func:`_coarsen_confidence`).  The difference is the same on every level, so
    the iteration still converges at a rate that barely depends on the grid size.

    Away from the border the ``1/2`` makes the rows of ``R`` the usual full
    weighting ``[1 2 1] / 4``.  At the ends the rows are those the interpolation
    demands; they sum to ``3/4`` rather than one, because the fine grid, whose
    samples sit one half-cell inside the faces, is not symmetric either.
    """
    size = field.shape[axis]
    samples = (size + 1) // 2
    coarse = _axis_slice(field, axis, slice(0, None, 2)).copy()
    odd = _axis_slice(field, axis, slice(1, None, 2))
    count = odd.shape[axis]
    _axis_slice(coarse, axis, slice(0, count))[...] += 0.5 * odd
    _axis_slice(coarse, axis, slice(1, samples))[...] += 0.5 * _axis_slice(
        odd, axis, slice(0, samples - 1))
    if size == 2 * samples:
        # The interpolation of the last fine sample repeats the last coarse
        # sample, so that column of ``P`` carries weight one while only the
        # left half of it was added above.
        last = _axis_slice(odd, axis, slice(count - 1, count))
        _axis_slice(coarse, axis, slice(samples - 1, samples))[...] += 0.5 * last
    return 0.5 * coarse


def _restrict(field):
    """Restrict every axis of ``field``, the ``R`` of a V-cycle."""
    coarse = field
    for axis in range(field.ndim):
        coarse = _restrict_axis(coarse, axis)
    return coarse


def _prolong_axis(coarse, size, axis):
    """Double ``axis`` by linear interpolation, ``size`` samples in the result.

    Even fine samples copy a coarse sample and odd ones average their two coarse
    neighbours.  When ``size`` is even the last fine sample is odd and its second
    coarse neighbour lies outside the grid; it is taken as the mirror image of
    the last coarse sample, which repeats the field flat instead of inventing a
    slope.  A Neumann boundary is what prolongation should do, since the
    prolongation of a correction constant along the border must stay constant
    along it.
    """
    samples = coarse.shape[axis]
    if size not in (2 * samples, 2 * samples - 1):
        raise ValueError("a grid of %d samples cannot be prolonged to %d"
                         % (samples, size))
    if size == 2 * samples:
        mirror = _axis_slice(coarse, axis, slice(samples - 1, samples))
        extended = np.concatenate([coarse, mirror], axis=axis)
    else:
        extended = coarse
    pairs = size - samples
    low = _axis_slice(extended, axis, slice(0, pairs))
    high = _axis_slice(extended, axis, slice(1, pairs + 1))
    average = 0.5 * (low + high)
    shape = list(coarse.shape)
    shape[axis] = size
    fine = np.empty(shape, dtype=coarse.dtype)
    _axis_slice(fine, axis, slice(0, None, 2))[...] = coarse
    odd = size - samples
    _axis_slice(fine, axis, slice(1, 2 * odd, 2))[...] = _axis_slice(
        average, axis, slice(0, odd))
    return fine


def _prolong(coarse, shape):
    """Prolong ``coarse`` to ``shape``, the ``P`` of a V-cycle."""
    fine = coarse
    for axis, size in enumerate(shape):
        fine = _prolong_axis(fine, size, axis)
    return fine


def _residual(level):
    """``rhs - factor * L phi`` of one level."""
    return level.rhs - level.factor * _operator(level.phi, level.edges)


def _smooth(level, sweeps):
    """Red-black Gauss--Seidel sweeps of ``rhs = factor * L phi``.

    All red samples are updated first.  Their neighbours are black and stay
    untouched during the half-sweep, so the update is one array expression
    instead of a loop, and the same holds for the black half-sweep that follows.
    A sample whose diagonal is zero carries no information and is skipped.

    The equation being relaxed is ``rhs = factor * L phi``, so the division is
    by the diagonal of ``factor * L``; forgetting ``factor`` would multiply the
    correction of every coarse grid by ``4 ** level`` and diverge.
    """
    alive = level.diagonal != 0.0
    divisor = np.where(alive, level.factor * level.diagonal, -1.0)
    black = _checkerboard(level.rhs.shape)
    for _ in range(sweeps):
        for colour in (False, True):
            step = np.where(alive & (black == colour),
                            _residual(level) / divisor, 0.0)
            level.phi += step


def _v_cycle(levels, index, pre_smooth, post_smooth, coarse_sweeps):
    """One V-cycle of the hierarchy, starting at ``levels[index]``.

    The coarse grid is filled with the restricted residual of the finer one and
    relaxed, then its solution is interpolated back as a correction.  The
    recursion stops at the coarsest grid, which is relaxed in place.
    """
    level = levels[index]
    if index + 1 == len(levels):
        _smooth(level, coarse_sweeps)
        return
    _smooth(level, pre_smooth)
    restricted = _restrict(_zero_mean(_residual(level)))
    coarse = levels[index + 1]
    coarse.rhs[...] = _zero_mean(restricted)
    coarse.phi[...] = 0.0
    _v_cycle(levels, index + 1, pre_smooth, post_smooth, coarse_sweeps)
    level.phi += _prolong(coarse.phi, level.phi.shape)
    _smooth(level, post_smooth)


def _zero_mean(field):
    """Remove the average of ``field``.

    A constant is invisible to the operator, so a right-hand side with a
    nonzero average has no solution and its constant part only slows the
    iteration down.
    """
    return field - field.mean()


def _coarsen_confidence(nodes):
    """Average a confidence field onto the next coarser grid.

    The restriction does not reproduce a constant field: its rows sum to
    ``0.75`` at the first sample of a profile and to ``1.25`` at the last
    interior one, because the interpolation it transposes puts the outer sample
    half a cell inside the boundary.  Dividing by the restriction of an
    all-ones field turns ``R`` into a weighted average, so a constant survives
    coarsening exactly.

    That matters more than it looks.  The coarse edge weights are the squares of
    these numbers, so a uniform confidence of one would come out as ``0.5625``
    at the corner of an even-sized two-dimensional coarse grid and ``1.5625`` at
    the sample farthest from it, and every further level would dent its own
    boundary again.  The coarse grid would then be solving a *different*
    equation from the fine one -- with weaker coupling at the edge -- and the
    correction it hands back would be consistently too small.  The smoother
    cannot repair what the coarse grid under-corrects, and the rate of a cycle
    would climb with the image size instead of staying flat.  With the division,
    a constant-coefficient problem keeps the same coefficient on every level and
    the rate stays between ``0.17`` and ``0.23`` a cycle from ``16**2`` to
    ``512**2``.
    """
    return _restrict(nodes) / _restrict(np.ones_like(nodes))


def _build(phase, confidence, levels):
    """Build the hierarchy, finest grid first.

    The finest level is the weighted least-squares problem exactly as
    :func:`parvaneh.core.unwrap` states it.  Each level below coarsens the
    confidence field, re-derives its edge weights and gets its own zero initial
    guess; only the right-hand side of a coarse level is filled in later, by the
    V-cycle that reaches it.
    """
    edge_weights, differences = _edges(phase, confidence)
    finest = _Level(
        edges=edge_weights,
        diagonal=_diagonal(edge_weights, phase.shape),
        factor=1.0,
        rhs=_divergence(differences),
        phi=np.zeros(phase.shape, dtype=np.float64),
    )
    grids = [finest]
    nodes = confidence
    while (levels is None or len(grids) < levels) and min(nodes.shape) >= 3:
        nodes = _coarsen_confidence(nodes)
        weights = _edge_weights(nodes)
        grids.append(_Level(
            edges=weights,
            diagonal=_diagonal(weights, nodes.shape),
            factor=grids[-1].factor / 4.0,
            rhs=np.zeros(nodes.shape, dtype=np.float64),
            phi=np.zeros(nodes.shape, dtype=np.float64),
        ))
    return grids


def multigrid_unwrap(phase, weight=None, mask=None, *, levels=None,
                     max_cycles=50, tol=1e-8, pre_smooth=2, post_smooth=2,
                     coarse_sweeps=50, backend="numpy", return_info=False):
    """Unwrap a phase field by descending a hierarchy of coarser grids.

    The problem is the weighted least-squares problem of
    :func:`parvaneh.core.unwrap`, and the result agrees with that solver up to
    the tolerance reached; the difference is the way the linear system is
    solved.  Starting from a zero field, the routine relaxes the finest grid,
    restricts the residual it leaves to a grid with half the samples along every
    axis, solves that coarser problem recursively, and adds the interpolated
    correction back.  Such a pass is a V-cycle, and cycles repeat until the
    residual of the finest grid falls under ``tol``.

    Parameters
    ----------
    phase : array_like
        Array of wrapped phase in radians, any number of dimensions, all values
        finite and every axis at least two samples long.
    weight : array_like, optional
        Per-sample confidence in ``[0, 1]`` with ``phase.shape``.  The weight of
        a step is the smaller of its two endpoints, so a confident step is
        expensive to contradict.  ``None`` means every sample is equally
        reliable, which gives the unweighted multigrid of Pritt (1996).
    mask : array_like of bool, optional
        ``True`` where the phase may be used.  Masked samples are given zero
        weight, which removes them from the problem, and disconnected valid
        regions each keep an arbitrary constant offset.
    levels : int, optional
        Largest number of grids to use.  ``None`` coarsens until an axis would
        fall below two samples, which is the usual choice.  ``levels=1``
        relaxes the finest grid alone, which makes the gain of the hierarchy
        measurable but converges very slowly on a large image.
    max_cycles : int, optional
        Limit on the number of V-cycles.
    tol : float, optional
        Stop once ``‖f₀ - L₀φ₀‖₂ / ‖f₀‖₂``, the residual :mod:`parvaneh.core`
        reports as well, is at most this value.
    pre_smooth, post_smooth : int, optional
        Red-black Gauss--Seidel sweeps before the descent and after the
        correction.  At least one sweep in total is required; the defaults are
        the two before and two after of the classical V-cycle.  Two a side is
        also a practical floor: a piecewise-constant weight field at ``2e-3``
        diverges with one sweep a side and stalls with two, so lowering this is
        not a way to buy speed.
    coarse_sweeps : int, optional
        Sweeps spent on the coarsest grid.  Its size is fixed, so a generous
        number costs little.
    backend : {"numpy"}, optional
        Only the reference implementation is provided.
    return_info : bool, optional
        Return the result together with a :class:`MultigridInfo` record.

    Returns
    -------
    result : ndarray
        Unwrapped phase with zero mean, of the same shape as ``phase``.  Masked
        or zero-weight samples carry no information and hold whatever the
        iteration left there, ``0`` from the zero start; masking is applied by
        the caller, as in :mod:`parvaneh.core`.
    info : MultigridInfo
        Only returned when ``return_info`` is true.

    Raises
    ------
    ValueError
        If ``phase`` is not a finite array of at least two samples along every
        axis, if ``backend`` is unknown, if every weight is zero, or if a
        parameter is out of range.

    Examples
    --------
    A ramp is recovered exactly, up to the constant that no unwrapping method
    can determine:

    >>> import numpy as np
    >>> ramp = 0.3 * np.add.outer(np.arange(9.0), np.arange(11.0))
    >>> phase = (ramp + np.pi) % (2.0 * np.pi) - np.pi
    >>> result, info = multigrid_unwrap(phase, return_info=True)
    >>> bool(np.abs(result - (ramp - ramp.mean())).max() < 1e-6)
    True
    >>> bool(info.converged)
    True

    Notes
    -----
    The unweighted problem converges in fewer cycles than the weighted one,
    because a mask breaks the smoothness of the coefficients that full weighting
    transfers assume.  A discontinuous coefficient field is the hard case, and a
    mask does not necessarily prepare you for it: see "Where the hierarchy stops
    helping" in the module docstring for the measured plateaus near ``2e-06`` on
    thin high-contrast lines, where :func:`parvaneh.core.unwrap` remains the
    cheaper and more reliable solver.  The grid hierarchy is interesting when a
    transform is inconvenient, when only a rough solution of a very large problem
    is needed, or as a building block for the three-dimensional and multichannel
    problems where the transform does not apply.
    """
    phase = _check_phase(phase)
    if backend != "numpy":
        raise ValueError("backend must be 'numpy'")
    if levels is not None and levels < 1:
        raise ValueError("levels must be positive or None")
    if max_cycles < 1:
        raise ValueError("max_cycles must be positive")
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    if pre_smooth < 0 or post_smooth < 0:
        raise ValueError("pre_smooth and post_smooth must be nonnegative")
    if pre_smooth + post_smooth < 1:
        raise ValueError("a V-cycle needs at least one smoothing sweep")
    if coarse_sweeps < 1:
        raise ValueError("coarse_sweeps must be positive")
    confidence = _confidence(phase, mask, weight)
    if not np.any(confidence > 0.0):
        raise ValueError(
            "every sample has zero weight, so nothing can be unwrapped")

    grids = _build(phase, confidence, levels)
    reference = float(np.linalg.norm(grids[0].rhs))
    residuals = []
    converged = reference == 0.0
    started = time.perf_counter()
    if not converged:
        for _ in range(max_cycles):
            _v_cycle(grids, 0, pre_smooth, post_smooth, coarse_sweeps)
            relative = float(np.linalg.norm(_residual(grids[0])) / reference)
            residuals.append(relative)
            if relative <= tol:
                converged = True
                break
    seconds = time.perf_counter() - started

    result = grids[0].phi
    result -= result.mean()
    if not return_info:
        return result
    per_cycle = (len(grids) - 1) * (pre_smooth + post_smooth) + coarse_sweeps
    info = MultigridInfo(
        backend=backend,
        levels=len(grids),
        coarsest=tuple(int(size) for size in grids[-1].phi.shape),
        cycles=len(residuals),
        sweeps=len(residuals) * per_cycle,
        relative_residual=residuals[-1] if residuals else 0.0,
        converged=converged,
        seconds=seconds,
        residuals=tuple(residuals),
    )
    return result, info
