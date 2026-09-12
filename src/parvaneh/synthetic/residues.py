"""Residues of a wrapped phase field.

A *residue* (also called a *charge*) is the elementary obstruction to
unwrapping.  It is computed from the **wrapped** phase only, by circulating
around every :math:`2 \\times 2` block of pixels and asking whether the four
wrapped gradients add up to zero.  Where they do not, the phase cannot be
unwrapped consistently no matter which algorithm is used, and every robust
unwrapper (branch-cut, minimum-cost flow, SNAPHU) has to route around, through
or past that pixel.

Why this is worth a whole module
--------------------------------

The single most common way to get residues wrong is to compute them from the
*unwrapped* phase, or from gradients that were differenced before wrapping.
Both give an identically zero map and look plausible.  In this package residues
are always derived from the wrapped phase, exactly as an unwrapper would see
it, so that the ground-truth residue map and the unwrapper's own residue map
can be compared pixel by pixel.

The mathematics, for a reader who has not met it before
-------------------------------------------------------

Take a wrapped phase :math:`\\phi_{i,j} \\in [-\\pi, \\pi]` on a rectangular
grid, where :math:`i` indexes rows and :math:`j` indexes columns.  Define the
wrapped gradient between two neighbouring pixels as

.. math::

    \\Delta \\phi = \\operatorname{wrap}(\\phi_b - \\phi_a)
    \\quad\\text{with}\\quad
    \\operatorname{wrap}(x) = \\operatorname{atan2}(\\sin x, \\cos x),

so every gradient lands in :math:`[-\\pi, \\pi]`.  For the :math:`2 \\times 2`
block with corners

.. code-block:: text

    D = (i,   j)     C = (i,   j+1)      north-west    north-east
    A = (i+1, j)     B = (i+1, j+1)      south-west    south-east

Traverse the block counter-clockwise in map view -- east along the south edge,
north up the east edge, west along the north edge, south down the west edge --
and add the four wrapped gradients:

.. math::

    S = \\Delta\\phi_{A \\to B} + \\Delta\\phi_{B \\to C}
      + \\Delta\\phi_{C \\to D} + \\Delta\\phi_{D \\to A}.

Each term is in :math:`[-\\pi, \\pi]`, so :math:`S \\in (-4\\pi, 4\\pi)`.  When
the phase is noise-free the only achievable values are :math:`-2\\pi`,
:math:`0` and :math:`+2\\pi`, so the residue is

.. math::

    r = \\operatorname{round}\\!\\left(\\frac{S}{2\\pi}\\right) \\in \\{-1, 0, +1\\}.

A grid of zeros is "residue free": the wrapped phase is then the wrapping of a
globally continuous surface and a naive flood-fill unwrapper will succeed.
Every :math:`+1` must be joined to a :math:`-1` by a branch cut, or balanced by
flow, before unwrapping can proceed.  Residues are created by noise, by
aliasing (steep deformation, more than half a fringe per pixel), and by genuine
discontinuities such as a fault trace or the edge of a mask.

Aliasing is worth its own sentence, because it is the one failure mode this
module cannot see.  Wrapping maps every difference into
:math:`[-\\pi, \\pi]`, so a jump of :math:`\\pi + \\epsilon` and a jump of
:math:`-\\pi + \\epsilon` produce *identical* wrapped phases.  No function of
the wrapped phase -- residues included -- can therefore reveal aliasing.  The
residues reported here are the honest residues of the aliased data an unwrapper
would be handed.  A synthetic dataset, which knows the true phase, should
instead flag aliased pixels with :func:`aliasing_mask` and report scores on the
rest.

A subtlety worth knowing: residues do **not** always come in exactly balanced
pairs.  Summing the circulation over every block, interior edges cancel in
pairs, and what survives is one circulation around the outer boundary of the
image.  Hence

.. math::

    \\sum_{i,j} r_{i,j} = \\frac{1}{2\\pi} \\oint_{\\partial \\Omega}
    \\nabla \\phi \\cdot \\mathrm{d}\\boldsymbol{\\ell},

which is zero for a periodic domain, and zero for any field whose wrapped
phase happens to be single-valued on the boundary, but is otherwise an
arbitrary integer.  In practice the imbalance grows like the perimeter while
the number of charges grows like the area, so it is negligible for large
images.  :func:`boundary_circulation` returns the right-hand side, so the
identity can be checked rather than assumed.

Sign and orientation conventions
--------------------------------

The counter-clockwise traversal above is defined in the package's map
convention: :math:`x` (column index) increases to the **east** and :math:`y`
(row index) increases to the **south**, i.e. row 0 is the northernmost row,
matching ``imshow(origin="upper")``.  A positive residue is therefore a
positive counter-clockwise circulation in *map view*, comparable with any
textbook residue map drawn with north up.  Other codes differ by an overall
factor :math:`-1` depending on whether they traverse clockwise or think of rows
as increasing northwards, so compare residue *counts* and *balance* rather than
residue signs when cross-checking against another package.

References
----------
Ghiglia, D. C. and Pritt, M. D. (1998), *Two-Dimensional Phase Unwrapping:
Theory, Algorithms, and Software*, Wiley.  See the chapters on residues and
branch cuts; the circulation definition used here is the standard one.
"""

import numpy as np

from .wrapping import TWO_PI, wrap_phase

__all__ = [
    "wrapped_gradients",
    "phase_gradients",
    "aliasing_mask",
    "residue_map",
    "plaquette_valid_mask",
    "positive_residues",
    "negative_residues",
    "residue_balance",
    "residue_density",
    "boundary_circulation",
]


def _round_half_away(values):
    """Round ``values`` to the nearest integer, halves away from zero.

    ``np.round`` uses banker's rounding (``np.round(0.5) == 0.0``) and its
    tie-breaking rule has changed across NumPy versions, so it is unusable
    where reproducibility matters.  Every integer-cycle decision in this
    package goes through this helper.
    """
    array = np.asarray(values, dtype=np.float64)
    return np.copysign(np.floor(np.abs(array) + 0.5), array)


def _as_phase(phase_wrapped):
    """Coerce input to a 2-D ``float64`` array, raising on a bad shape."""
    array = np.asarray(phase_wrapped, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(
            "phase_wrapped must be a 2-D array of shape (ny, nx), got "
            "shape {}".format(array.shape))
    if array.shape[0] < 2 or array.shape[1] < 2:
        raise ValueError(
            "phase_wrapped must be at least 2 x 2 so that a plaquette "
            "exists, got shape {}".format(array.shape))
    return array


def _as_valid(phase_wrapped, mask):
    """Boolean array of usable pixels: finite phase and inside ``mask``.

    ``NaN`` is this package's marker for "no data", so a pixel that is ``NaN``
    is never usable and any block touching it yields no residue.
    """
    if mask is None:
        return np.isfinite(phase_wrapped)
    valid = np.asarray(mask, dtype=bool)
    if valid.shape != phase_wrapped.shape:
        raise ValueError(
            "mask must have the same shape as phase_wrapped, got {} and "
            "{}".format(valid.shape, phase_wrapped.shape))
    return np.logical_and(valid, np.isfinite(phase_wrapped))


def wrapped_gradients(phase_wrapped, mask=None):
    """Wrapped phase differences between neighbouring pixels.

    Parameters
    ----------
    phase_wrapped : array_like
        Wrapped phase in radians, shape ``(ny, nx)``.  ``NaN`` marks no data.
    mask : array_like of bool, optional
        Valid-data mask of the same shape.  Pixels outside the mask are
        treated as no data.

    Returns
    -------
    d_east : ndarray
        ``phi[:, 1:] - phi[:, :-1]`` wrapped into :math:`[-\\pi, \\pi]`,
        shape ``(ny, nx - 1)``.
    d_north : ndarray
        ``phi[:-1, :] - phi[1:, :]`` wrapped into :math:`[-\\pi, \\pi]`,
        shape ``(ny - 1, nx)``.  Row index increases southward, hence the
        reversed subtraction for a northward difference.

    Notes
    -----
    When either end of a difference is invalid the difference is ``NaN``
    rather than zero, so that a careless consumer cannot mistake "unknown" for
    "no change".  :func:`residue_map` turns those ``NaN`` entries into zero
    residues explicitly.

    Examples
    --------
    A constant phase has zero gradient everywhere:

    >>> phi = np.full((3, 4), 0.7)
    >>> east, north = wrapped_gradients(phi)
    >>> east.shape, north.shape
    ((3, 3), (2, 4))
    >>> bool(np.all(east == 0.0))
    True

    A gentle ramp does not wrap:

    >>> phi = np.array([[0.0, 0.2], [0.0, 0.2]])
    >>> [round(float(v), 6) for v in wrapped_gradients(phi)[0].ravel()]
    [0.2, 0.2]

    A steep step wraps into a small gradient of the opposite sign, which is
    exactly the ambiguity residues exist to detect:

    >>> phi = np.array([[0.0, 4.0], [0.0, 4.0]])
    >>> [round(float(v), 6) for v in wrapped_gradients(phi)[0].ravel()]
    [-2.283185, -2.283185]
    """
    phase = _as_phase(phase_wrapped)
    valid = _as_valid(phase, mask)

    d_east = wrap_phase(phase[:, 1:] - phase[:, :-1])
    d_north = wrap_phase(phase[:-1, :] - phase[1:, :])

    both = np.logical_and(valid[:, 1:], valid[:, :-1])
    d_east = np.where(both, d_east, np.nan)
    both = np.logical_and(valid[:-1, :], valid[1:, :])
    d_north = np.where(both, d_north, np.nan)
    return d_east, d_north


def phase_gradients(phase):
    """Unwrapped first differences of a *continuous* phase field.

    This is the continuous counterpart of :func:`wrapped_gradients`, and it is
    how a synthetic dataset detects aliasing: only a generator that knows the
    true phase can measure how steep that phase really is.

    Parameters
    ----------
    phase : array_like
        Continuous phase in radians, shape ``(ny, nx)``.  May be any real
        values, not restricted to :math:`[-\\pi, \\pi]`.

    Returns
    -------
    d_east : ndarray
        ``phi[:, 1:] - phi[:, :-1]``, shape ``(ny, nx - 1)``, in radians per
        pixel.
    d_north : ndarray
        ``phi[:-1, :] - phi[1:, :]``, shape ``(ny - 1, nx)``, in radians per
        pixel, positive northward.

    Examples
    --------
    >>> phi = np.array([[0.0, 0.4], [0.0, 0.4]])
    >>> [round(float(v), 6) for v in phase_gradients(phi)[0].ravel()]
    [0.4, 0.4]

    Unlike :func:`wrapped_gradients`, a jump larger than half a fringe stays
    large -- which is the whole point:

    >>> phi = np.array([[0.0, 4.0], [0.0, 4.0]])
    >>> [round(float(v), 6) for v in phase_gradients(phi)[0].ravel()]
    [4.0, 4.0]
    """
    array = np.asarray(phase, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(
            "phase must be a 2-D array of shape (ny, nx), got shape "
            "{}".format(array.shape))
    return (array[:, 1:] - array[:, :-1],
            array[:-1, :] - array[1:, :])


def aliasing_mask(phase, threshold=np.pi):
    """Flag pixels where the true phase gradient reaches half a fringe.

    A fringe is :math:`2\\pi` of phase.  If the phase changes by
    :math:`\\pi` or more from one pixel to the next, the wrapped phase cannot
    tell that jump apart from the same jump with the opposite sign, and every
    unwrapper -- not just the one being benchmarked -- will fail there.  A
    dataset that claims to be hard must say where this happens, so that scores
    can be reported on the remaining, genuinely unwrappable pixels.

    Parameters
    ----------
    phase : array_like
        Continuous phase in radians, shape ``(ny, nx)``.
    threshold : float, optional
        Gradient magnitude in radians per pixel at which a pixel is flagged.
        Defaults to :math:`\\pi`, half a fringe.

    Returns
    -------
    ndarray of bool
        Shape ``(ny, nx)``.  A pixel is ``True`` if it, or any of its four
        neighbours, is separated from it by a gradient of at least
        ``threshold`` in magnitude.

    Raises
    ------
    ValueError
        If ``phase`` is not 2-D, or ``threshold`` is not positive.

    Examples
    --------
    A gentle ramp is nowhere aliased:

    >>> y, x = np.mgrid[0:6, 0:6]
    >>> bool(aliasing_mask(0.2 * x + 0.1 * y).any())
    False

    Double the gradient and the steepest pixels trip the threshold:

    >>> mask = aliasing_mask(3.5 * x + 0.1 * y)
    >>> int(mask.sum())
    36

    Raising the threshold clears it again:

    >>> bool(aliasing_mask(3.5 * x + 0.1 * y, threshold=4.0).any())
    False
    """
    array = np.asarray(phase, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(
            "phase must be a 2-D array of shape (ny, nx), got shape "
            "{}".format(array.shape))
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError(
            "threshold must be a positive finite number of radians per pixel, "
            "got {!r}".format(threshold))

    d_east, d_north = phase_gradients(array)
    flagged = np.zeros(array.shape, dtype=bool)
    steep_east = np.abs(d_east) >= threshold
    flagged[:, :-1] = np.logical_or(flagged[:, :-1], steep_east)
    flagged[:, 1:] = np.logical_or(flagged[:, 1:], steep_east)
    steep_north = np.abs(d_north) >= threshold
    flagged[:-1, :] = np.logical_or(flagged[:-1, :], steep_north)
    flagged[1:, :] = np.logical_or(flagged[1:, :], steep_north)
    return flagged


def plaquette_valid_mask(phase_wrapped, mask=None):
    """Which :math:`2 \\times 2` blocks have four usable pixels.

    Parameters
    ----------
    phase_wrapped : array_like
        Wrapped phase in radians, shape ``(ny, nx)``.
    mask : array_like of bool, optional
        Valid-data mask of the same shape.

    Returns
    -------
    ndarray of bool
        Shape ``(ny - 1, nx - 1)``.  ``True`` where a residue could be
        computed, i.e. where all four corners of the block are finite and
        inside the mask.

    Examples
    --------
    >>> phi = np.zeros((3, 3))
    >>> phi[1, 1] = np.nan
    >>> plaquette_valid_mask(phi).astype(int).tolist()
    [[0, 0], [0, 0]]

    >>> phi[1, 1] = 0.0
    >>> plaquette_valid_mask(phi).astype(int).tolist()
    [[1, 1], [1, 1]]
    """
    phase = _as_phase(phase_wrapped)
    valid = _as_valid(phase, mask)
    return np.logical_and(
        np.logical_and(valid[:-1, :-1], valid[:-1, 1:]),
        np.logical_and(valid[1:, :-1], valid[1:, 1:]))


def residue_map(phase_wrapped, mask=None):
    """Residue (charge) of every :math:`2 \\times 2` block of pixels.

    Parameters
    ----------
    phase_wrapped : array_like
        Wrapped phase in radians, shape ``(ny, nx)``.  ``NaN`` marks no data.
    mask : array_like of bool, optional
        Valid-data mask of the same shape.

    Returns
    -------
    ndarray of ``int8``
        Shape ``(ny - 1, nx - 1)`` with entries in ``{-1, 0, +1}``.  Entry
        ``(i, j)`` describes the block whose corners are pixels ``(i, j)``,
        ``(i, j+1)``, ``(i+1, j)`` and ``(i+1, j+1)``, so the map can be
        overlaid on the wrapped-phase image without any offset.  Blocks that
        touch no data are reported as ``0``; use
        :func:`plaquette_valid_mask` if those must be told apart.

    Notes
    -----
    The circulation is summed in the order south edge east, east edge north,
    north edge west, west edge south -- counter-clockwise in map view.  See the
    module docstring for why that ordering, and only that ordering, gives a
    sign comparable with a textbook residue map.

    Aliasing **cannot** be detected here.  A pixel-to-pixel jump of more than
    half a fringe is indistinguishable from a jump of the opposite sign once
    the phase has been wrapped, so no function of the wrapped phase can spot
    it.  :func:`aliasing_mask` does the job from the continuous phase, which a
    synthetic dataset is blessed to have.

    Examples
    --------
    A noise-free ramp has no residues:

    >>> y, x = np.mgrid[0:5, 0:5]
    >>> phi = wrap_phase(0.3 * x + 0.2 * y)
    >>> residue_map(phi).astype(int).tolist()
    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    A vortex does.  Note the sign: rows run southward, so a vortex that turns
    counter-clockwise on the page is traversed clockwise in this convention and
    its charge is negative.  This is the documented sign convention in action,
    not a bug:

    >>> y, x = np.mgrid[-2:3, -2:3] * 1.0
    >>> residue_map(wrap_phase(np.arctan2(y, x))).astype(int).tolist()
    [[0, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    Invalid pixels suppress the residues they touch:

    >>> phi = np.zeros((3, 3))
    >>> phi[1, 1] = np.nan
    >>> int(np.abs(residue_map(phi)).sum())
    0

    Output dtype is ``int8``, which keeps large residue maps small:

    >>> residue_map(np.zeros((3, 3))).dtype
    dtype('int8')
    """
    phase = _as_phase(phase_wrapped)
    valid = _as_valid(phase, mask)
    return _residue_from_phase_and_validity(phase, valid)


def _residue_from_phase_and_validity(phase, valid):
    """Core circulation; ``phase`` must already be finite-or-``NaN`` float64.

    The charge is ``round(circulation / 2 pi)``, ties away from zero, which is
    the standard definition (Ghiglia and Pritt, 1998).  For noise-free data the
    circulation is exactly ``0`` or ``+/- 2 pi`` and the rounding is exact; for
    noisy data it is a deterministic way of collapsing a continuous
    circulation onto the nearest integer charge.
    """
    # Wrapped differences along each of the four edges, each shaped
    # (ny - 1, nx - 1) so that the four add element-wise.
    south = wrap_phase(phase[1:, 1:] - phase[1:, :-1])      # A -> B
    east = wrap_phase(phase[:-1, 1:] - phase[1:, 1:])       # B -> C
    north = wrap_phase(phase[:-1, :-1] - phase[:-1, 1:])    # C -> D
    west = wrap_phase(phase[1:, :-1] - phase[:-1, :-1])     # D -> A

    circulation = south + east + north + west
    circulation = np.where(np.isfinite(circulation), circulation, 0.0)

    block_ok = np.logical_and(
        np.logical_and(valid[:-1, :-1], valid[:-1, 1:]),
        np.logical_and(valid[1:, :-1], valid[1:, 1:]))
    residues = _round_half_away(circulation / TWO_PI)
    return np.where(block_ok, residues, 0.0).astype(np.int8)


def _dipole_phase():
    """Two opposite vortices: a compact, exactly two-charge test field.

    Kept as a helper so that the docstrings and the test suite share one field
    and cannot drift apart.
    """
    y, x = np.mgrid[-2:3, -2:3] * 1.0
    return wrap_phase(np.arctan2(y, x) - np.arctan2(y, x - 2.0))


def positive_residues(phase_wrapped, mask=None):
    """Boolean map of blocks carrying a ``+1`` charge.

    Parameters
    ----------
    phase_wrapped : array_like
        Wrapped phase in radians, shape ``(ny, nx)``.
    mask : array_like of bool, optional
        Valid-data mask of the same shape.

    Returns
    -------
    ndarray of bool
        Shape ``(ny - 1, nx - 1)``, ``True`` at positive residues.

    Examples
    --------
    A pair of equal and opposite vortices gives one charge of each sign:

    >>> phi = _dipole_phase()
    >>> int(positive_residues(phi).sum())
    1
    >>> int(negative_residues(phi).sum())
    1

    The positive charge sits two columns east of the negative one:

    >>> positive_residues(phi).astype(int).tolist()
    [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]]
    """
    return residue_map(phase_wrapped, mask=mask) == 1


def negative_residues(phase_wrapped, mask=None):
    """Boolean map of blocks carrying a ``-1`` charge.

    Parameters
    ----------
    phase_wrapped : array_like
        Wrapped phase in radians, shape ``(ny, nx)``.
    mask : array_like of bool, optional
        Valid-data mask of the same shape.

    Returns
    -------
    ndarray of bool
        Shape ``(ny - 1, nx - 1)``, ``True`` at negative residues.

    Examples
    --------
    >>> negative_residues(_dipole_phase()).astype(int).tolist()
    [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    """
    return residue_map(phase_wrapped, mask=mask) == -1


def residue_balance(residue, valid=None):
    """Count positive and negative charges in a residue map.

    This is the summary statistic a benchmark should record for every scene:
    a scene with a handful of balanced charges is easy, a scene with thousands
    is not, and a scene whose net charge is a large fraction of its absolute
    charge is usually a symptom of a bug rather than of hard data.

    Parameters
    ----------
    residue : array_like
        Integer residue map, typically the output of :func:`residue_map`.
    valid : array_like of bool, optional
        Optional mask on the *residue* grid, shape ``(ny - 1, nx - 1)``, to
        restrict the count to blocks that were computable.

    Returns
    -------
    dict
        Keys ``"positive"``, ``"negative"``, ``"total"`` (the net charge
        ``positive - negative``) and ``"absolute"`` (``positive +
        negative``).  All are Python ``int``.

    Raises
    ------
    ValueError
        If ``residue`` is not a 2-D map of whole numbers, or if ``valid`` has
        the wrong shape.  Passing a phase array instead of the output of
        :func:`residue_map` is an error rather than a silent rounding.

    Notes
    -----
    A non-zero ``"total"`` is not necessarily an error.  It equals the
    circulation of the wrapped gradient once around the outer boundary, in
    units of :math:`2\\pi`; see the module docstring and
    :func:`boundary_circulation`.  On a periodic domain, or when the mask
    covers the whole image and the wrapped phase is single-valued on the
    boundary, the total is exactly zero.

    Examples
    --------
    Opposite vortices balance exactly:

    >>> residue_balance(residue_map(_dipole_phase()))
    {'positive': 1, 'negative': 1, 'total': 0, 'absolute': 2}

    A noise-free ramp has no charges at all:

    >>> y, x = np.mgrid[0:6, 0:6]
    >>> residue_balance(residue_map(wrap_phase(0.4 * x - 0.1 * y)))["absolute"]
    0

    ``valid`` zeroes the blocks it excludes, so it works directly with the
    output of :func:`plaquette_valid_mask`.  Punching a hole in the pixel on
    which the negative charge depends removes that charge and leaves only the
    positive one, which is why the two counts no longer balance:

    >>> phi = _dipole_phase()
    >>> phi[1, 2] = np.nan
    >>> residue_balance(residue_map(phi),
    ...                 valid=plaquette_valid_mask(phi))["negative"]
    0

    Passing a phase array by mistake is an error, not a silent rounding into
    bogus charges:

    >>> residue_balance(_dipole_phase())
    Traceback (most recent call last):
        ...
    ValueError: residue must be an integer map in {-1, 0, +1}; got non-integral values.  Did you forget to call residue_map() first?
    """
    charges = _as_charge_map(residue)
    if valid is not None:
        keep = np.asarray(valid, dtype=bool)
        if keep.shape != charges.shape:
            raise ValueError(
                "valid must have the same shape as residue, got {} and "
                "{}".format(keep.shape, charges.shape))
        charges = np.where(keep, charges, 0)
    positive = int(np.count_nonzero(charges > 0))
    negative = int(np.count_nonzero(charges < 0))
    return {
        "positive": positive,
        "negative": negative,
        "total": positive - negative,
        "absolute": positive + negative,
    }


def residue_density(residue, valid=None):
    """Fraction of blocks that carry a charge.

    A compact summary of how hard an image is to unwrap: a residue density of
    :math:`10^{-5}` is trivial, a few :math:`10^{-2}` is hard, and a field
    whose density approaches the aliasing limit has no usable unwrapping
    solution at all.

    Parameters
    ----------
    residue : array_like
        Integer residue map, typically the output of :func:`residue_map`.
    valid : array_like of bool, optional
        Blocks to include in the denominator.  Defaults to every block.

    Returns
    -------
    float
        Density in ``[0, 1]``.  ``0.0`` when no block is valid, so the value
        is always finite and never a silent divide-by-zero.

    Raises
    ------
    ValueError
        If ``residue`` is not a 2-D map of whole numbers, or if ``valid`` has
        the wrong shape.  Passing a phase array instead of the output of
        :func:`residue_map` is an error rather than a silent rounding.

    Examples
    --------
    Two charges among sixteen blocks:

    >>> round(residue_density(residue_map(_dipole_phase())), 4)
    0.125

    Restricting the denominator to valid blocks raises the density of a
    half-masked field:

    >>> residue = np.zeros((4, 4), dtype=int)
    >>> residue[0, 0] = 1
    >>> valid = np.zeros((4, 4), dtype=bool)
    >>> valid[:2, :] = True
    >>> round(residue_density(residue, valid=valid), 4)
    0.125
    """
    charges = _as_charge_map(residue)
    if valid is None:
        denominator = charges.size
        numerator = int(np.count_nonzero(charges))
    else:
        keep = np.asarray(valid, dtype=bool)
        if keep.shape != charges.shape:
            raise ValueError(
                "valid must have the same shape as residue, got {} and "
                "{}".format(keep.shape, charges.shape))
        denominator = int(np.count_nonzero(keep))
        numerator = int(np.count_nonzero(np.where(keep, charges, 0)))
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _as_charge_map(residue):
    """Coerce a residue map to a 2-D ``int64`` array of charges.

    The input must already *be* a residue map, i.e. an array of whole numbers,
    not a phase array.  Silently rounding a continuous phase into integers
    would turn every pixel whose value exceeds a half into a spurious charge,
    so non-integral input is rejected with a message pointing at
    :func:`residue_map`.
    """
    array = np.asarray(residue)
    if array.ndim != 2:
        raise ValueError(
            "residue must be a 2-D integer map, got shape {}".format(
                array.shape))
    if array.dtype.kind == "f":
        if not np.all(np.isfinite(array)):
            raise ValueError(
                "residue must be finite; a residue map has no NaN entries, "
                "so this looks like a phase array rather than the output of "
                "residue_map()")
        rounded = np.round(array)
        if np.any(np.abs(array - rounded) > 1e-9):
            raise ValueError(
                "residue must be an integer map in {-1, 0, +1}; got "
                "non-integral values.  Did you forget to call residue_map() "
                "first?")
    return _round_half_away(array).astype(np.int64)


def boundary_circulation(phase_wrapped, mask=None):
    """Circulation of the wrapped gradient once around the image boundary.

    This is the net charge that a residue map can carry, so it is the natural
    companion of :func:`residue_balance`::

        int(residue_map(phi).sum())
            == int(round(boundary_circulation(phi) / (2 * pi)))

    to within floating-point error.  The returned value is in radians.

    Parameters
    ----------
    phase_wrapped : array_like
        Wrapped phase in radians, shape ``(ny, nx)``.
    mask : array_like of bool, optional
        Valid-data mask.  If supplied it must be ``True`` on the entire outer
        boundary, because a circulation needs a closed loop; a ``ValueError``
        is raised otherwise.

    Returns
    -------
    float
        Boundary circulation in radians, in an integer multiple of
        :math:`2\\pi` for a well-behaved wrapped field.

    Raises
    ------
    ValueError
        If the phase is not 2-D or smaller than ``2 x 2``, or if the mask does
        not cover the whole boundary.

    Examples
    --------
    A ramp closes on itself, so the circulation vanishes:

    >>> y, x = np.mgrid[0:6, 0:6]
    >>> round(boundary_circulation(wrap_phase(0.4 * x - 0.1 * y)), 12)
    0.0

    A vortex keeps a whole turn:

    >>> y, x = np.mgrid[-2:3, -2:3] * 1.0
    >>> round(boundary_circulation(wrap_phase(np.arctan2(y, x))) / (2 * np.pi))
    -1

    And the identity with :func:`residue_map` holds:

    >>> y, x = np.mgrid[0:17, 0:23]
    >>> rng = np.random.default_rng(0)
    >>> phi = wrap_phase(rng.normal(0.0, 2.0, (17, 23)))
    >>> int(residue_map(phi).sum())
    1
    >>> int(round(boundary_circulation(phi) / (2 * np.pi)))
    1
    """
    phase = _as_phase(phase_wrapped)
    if mask is not None:
        valid = _as_valid(phase, mask)
        if not (valid[0, :].all() and valid[-1, :].all()
                and valid[:, 0].all() and valid[:, -1].all()):
            raise ValueError(
                "mask must be True on the whole outer boundary to define a "
                "circulation; use a mask that keeps the image border")
    # Counter-clockwise in map view: east along the south row, north up the
    # east column, west along the north row, south down the west column.
    south = wrap_phase(phase[-1, 1:] - phase[-1, :-1]).sum()
    east = wrap_phase(phase[:-1, -1] - phase[1:, -1]).sum()
    north = wrap_phase(phase[0, :-1] - phase[0, 1:]).sum()
    west = wrap_phase(phase[1:, 0] - phase[:-1, 0]).sum()
    return float(south + east + north + west)
