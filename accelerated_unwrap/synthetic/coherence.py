"""Ground-truth coherence maps for a synthetic interferogram.

What coherence is
-----------------

A SAR interferogram is formed by multiplying one complex SAR image by the
complex conjugate of a second one acquired a few days or weeks later.  Write
the two per-pixel complex reflectivities as :math:`z_1` and :math:`z_2`.  Their
*interferometric coherence* is the normalised magnitude of their correlation,
averaged over some local neighbourhood or over several independent looks:

.. math::

    \\gamma = \\frac{\\left| \\left\\langle z_1 z_2^{*} \\right\\rangle \\right|}
                    {\\sqrt{\\left\\langle |z_1|^2 \\right\\rangle
                            \\left\\langle |z_2|^2 \\right\\rangle}}
    \\quad\\in [0, 1].

:math:`\\gamma = 1` means the two images are perfectly correlated and the
interferometric phase is trustworthy.  :math:`\\gamma = 0` means they are
uncorrelated and the phase is pure noise.  Everything in between is the
practical range of real InSAR.

Why a whole module for it
-------------------------

Coherence matters twice in a phase-unwrapping benchmark:

1. It controls **how much phase noise the interferogram contains**.  The
   relation between coherence and phase scatter is a hard, well-tested
   statistical law, and :mod:`accelerated_unwrap.synthetic.noise` uses it
   rather than inventing an arbitrary Gaussian.  A realistic dataset therefore
   needs a *plausible coherence map* first.
2. It is the **ground truth for every quality measure** an unwrapper produces.
   A quality-guided unwrapper, a coherence-weighted least-squares solver or a
   simple mask all decide where to trust the phase using a coherence estimate.
   If the benchmark does not know the true coherence, it cannot say whether
   those decisions were good ones.

So the maps here are not decoration.  They are the reference field against
which a coherence estimator, a quality map and a masking rule are all scored.

The patterns provided
---------------------

===================== ==================================================
Function              Physical situation it stands in for
===================== ==================================================
:func:`uniform_coherence`            ideal, homogeneous scene
:func:`coherence_gradient`           a scene becoming more vegetated,
                                     wetter or snow-covered northwards
:func:`gaussian_coherence_patch`     a lake, a forest block, a sand dune
:func:`decorrelation_stripe`         a swath of farmland stripped bare
                                     between the two acquisitions
:func:`fault_zone_coherence`         surface rupture and rubble along a
                                     fault trace
:func:`random_coherence`             natural, patchy variation
:func:`circular_no_data_mask`        water, layover, shadow, no data
===================== ==================================================

:func:`combine_coherence` merges several causes.  Coherence is *multiplied*
when independent decorrelation mechanisms act at the same time, because each
one removes a fraction of the correlation, and the losses compound
multiplicatively.  Taking the minimum, as some conventions do, gives a similar
result whenever the patterns overlap; multiplication also correctly gives the
low-coherence factor the dominant say.

Conventions
-----------

* Coherence is a dimensionless real number in ``[0, 1]``.
* Arrays have shape ``(grid.ny, grid.nx)`` in the package-wide ENU raster
  layout: row 0 is the northernmost row, column 0 the westernmost column.  See
  :mod:`accelerated_unwrap.synthetic.grid`.
* Distances and lengths are in metres, angles in degrees and carry a ``_deg``
  suffix.
* A pixel whose coherence is ``nan`` is meant to be treated as *invalid*, not
  as *bad*.  Nothing can be said about it at all.  Use :data:`NO_DATA` for it
  and see :func:`circular_no_data_mask`.

References
----------

Bamler, R. & Hartl, P. (1998).  Synthetic aperture radar interferometry.
*Inverse Problems* 14, R1-R54.  doi:10.1088/0266-5611/14/4/001

Hanssen, R. F. (2001).  *Radar Interferometry: Data Interpretation and Error
Analysis*.  Kluwer.  doi:10.1007/0-306-47633-9
"""

import numpy as np

from ._checks import positive_float as _positive_float

__all__ = [
    "COHERENCE_LEVELS",
    "NO_DATA",
    "coherence_from_level",
    "uniform_coherence",
    "coherence_gradient",
    "gaussian_coherence_patch",
    "decorrelation_stripe",
    "fault_zone_coherence",
    "random_coherence",
    "circular_no_data_mask",
    "combine_coherence",
]


#: The value used for pixels that carry no usable information.  ``nan`` is
#: deliberately chosen over ``0.0``: coherence ``0`` is a measurement saying
#: "the phase is pure noise", whereas ``nan`` says "there is no measurement".
NO_DATA = float("nan")


#: Named coherence levels used throughout the specification.  These are the
#: values a real interferogram is usually binned into when a scene is
#: classified by quality.
COHERENCE_LEVELS = {
    "very_high": 0.95,
    "high": 0.80,
    "moderate": 0.60,
    "low": 0.40,
    "very_low": 0.20,
}


def coherence_from_level(level):
    """Turn a named coherence level, or a number, into a coherence value.

    Parameters
    ----------
    level : str or float
        Either a key of :data:`COHERENCE_LEVELS` (``"very_high"``, ``"high"``,
        ``"moderate"``, ``"low"``, ``"very_low"``) or a number, which is
        returned after checking that it lies in ``[0, 1]``.

    Returns
    -------
    float
        The coherence value.

    Raises
    ------
    ValueError
        If ``level`` is an unknown name, or a number outside ``[0, 1]``.

    Examples
    --------
    >>> coherence_from_level("high")
    0.8
    >>> coherence_from_level("very_low")
    0.2
    >>> coherence_from_level(0.55)
    0.55
    >>> coherence_from_level("excellent")
    Traceback (most recent call last):
        ...
    ValueError: unknown coherence level 'excellent'; known levels are high, low, moderate, very_high, very_low
    >>> coherence_from_level(-0.5)
    Traceback (most recent call last):
        ...
    ValueError: coherence must lie in [0, 1], got -0.5
    """
    if isinstance(level, str):
        key = level.strip().lower().replace(" ", "_").replace("-", "_")
        if key not in COHERENCE_LEVELS:
            raise ValueError(
                "unknown coherence level {!r}; known levels are {}".format(
                    level, ", ".join(sorted(COHERENCE_LEVELS))))
        return COHERENCE_LEVELS[key]
    value = float(level)
    if not np.isfinite(value):
        raise ValueError("coherence must be finite, got {!r}".format(level))
    if value < 0.0 or value > 1.0:
        raise ValueError(
            "coherence must lie in [0, 1], got {!r}".format(level))
    return value


def _check_grid(grid):
    """Reject anything that is not a :class:`Grid`-like object with X and Y."""
    if not hasattr(grid, "X") or not hasattr(grid, "Y"):
        raise ValueError(
            "grid must be a Grid instance with X and Y arrays, got {!r}".format(
                type(grid).__name__))


def _check_coherence(value, name):
    """Validate a coherence value or array and return it as a float array."""
    array = np.asarray(value, dtype=np.float64)
    finite = np.isfinite(array)
    if finite.any():
        low = float(array[finite].min())
        high = float(array[finite].max())
        if low < 0.0 or high > 1.0:
            raise ValueError(
                "{} must lie in [0, 1], got range [{!r}, {!r}]".format(
                    name, low, high))
    return array


def _distance_to_segments(x, y, px, py):
    """Euclidean distance from each point to a polyline, ignoring ends.

    ``px`` and ``py`` are the vertices of the polyline.  The distance is the
    minimum over all segments of the point-to-segment distance; outside the
    along-track extent of a segment the perpendicular foot is clamped to the
    nearer endpoint.
    """
    px = np.asarray(px, dtype=np.float64).ravel()
    py = np.asarray(py, dtype=np.float64).ravel()
    if px.size != py.size:
        raise ValueError(
            "polyline x and y must have the same length, got {} and {}".format(
                px.size, py.size))
    if px.size < 2:
        raise ValueError(
            "polyline needs at least two vertices, got {}".format(px.size))

    best = np.full(np.shape(x), np.inf)
    for index in range(px.size - 1):
        x0, y0 = px[index], py[index]
        dx, dy = px[index + 1] - x0, py[index + 1] - y0
        length_squared = dx * dx + dy * dy
        if length_squared <= 0.0:
            continue
        t = ((x - x0) * dx + (y - y0) * dy) / length_squared
        t = np.clip(t, 0.0, 1.0)
        best = np.minimum(best, np.hypot(x - (x0 + t * dx),
                                         y - (y0 + t * dy)))
    return best


def uniform_coherence(grid, coherence=0.8):
    """A constant coherence over the whole grid.

    This is the right starting point for a clean benchmark sample: it keeps the
    statistics of the phase noise homogeneous, so a measured error can be
    attributed to the unwrapping algorithm rather than to the scene.

    Parameters
    ----------
    grid : Grid
        Sampling grid; the output has shape ``(grid.ny, grid.nx)``.
    coherence : float or str, optional
        Coherence value, or a key of :data:`COHERENCE_LEVELS`.
        Default ``0.8``.

    Returns
    -------
    ndarray
        Coherence map of shape ``(grid.ny, grid.nx)``, dtype ``float64``.

    Examples
    --------
    >>> from accelerated_unwrap.synthetic.grid import Grid
    >>> grid = Grid(nx=4, ny=3, spacing=10.0)
    >>> gamma = uniform_coherence(grid, "very_high")
    >>> gamma.shape, float(gamma[0, 0]), int(gamma.dtype.itemsize)
    ((3, 4), 0.95, 8)
    """
    _check_grid(grid)
    value = coherence_from_level(coherence)
    return np.full(grid.shape, value, dtype=np.float64)


def coherence_gradient(grid, low=0.2, high=0.95, azimuth_deg=0.0):
    """A planar, linear coherence gradient across the grid.

    The coherence varies linearly with position projected onto the direction
    given by ``azimuth_deg``, reaching exactly ``low`` where the projection is
    smallest and exactly ``high`` where it is largest.  The whole map therefore
    spans ``[low, high]`` no matter what size the grid is.

    A physical reading: the scene is progressively losing coherence towards one
    side, for example because vegetation cover, soil moisture or snow are
    increasing in that direction.

    Parameters
    ----------
    grid : Grid
        Sampling grid.
    low, high : float or str, optional
        Coherence at the two ends of the gradient.  ``low`` must not exceed
        ``high``.  Either may be a name from :data:`COHERENCE_LEVELS`.
        Defaults ``0.2`` and ``0.95``.
    azimuth_deg : float, optional
        Direction, measured in degrees clockwise from North, along which
        coherence increases.  ``0`` means coherence increases northwards,
        ``90`` eastwards, ``180`` southwards and ``270`` westwards.
        Default ``0.0``.

    Returns
    -------
    ndarray
        Coherence map of shape ``(grid.ny, grid.nx)``.

    Raises
    ------
    ValueError
        If ``low`` is greater than ``high``, or either lies outside ``[0, 1]``.

    Examples
    --------
    >>> from accelerated_unwrap.synthetic.grid import Grid
    >>> grid = Grid.centered(nx=5, ny=5, spacing=100.0)
    >>> gamma = coherence_gradient(grid, low=0.2, high=1.0)
    >>> bool(np.all(gamma >= 0.2 - 1e-12) and np.all(gamma <= 1.0 + 1e-12))
    True
    >>> float(gamma[0, 2]) > float(gamma[-1, 2])     # north is more coherent
    True
    >>> np.round(float(gamma[0, 2]) + float(gamma[-1, 2]), 6)
    1.2
    """
    _check_grid(grid)
    low_value = coherence_from_level(low)
    high_value = coherence_from_level(high)
    if low_value > high_value:
        raise ValueError(
            "low must not exceed high, got low={!r} and high={!r}".format(
                low, high))

    azimuth = float(azimuth_deg)
    if not np.isfinite(azimuth):
        raise ValueError("azimuth_deg must be finite, got {!r}".format(
            azimuth_deg))

    theta = np.deg2rad(azimuth)
    # Unit vector pointing in the direction of increasing coherence.
    axis_e, axis_n = np.sin(theta), np.cos(theta)
    projection = axis_e * grid.X + axis_n * grid.Y
    span = projection.max() - projection.min()
    if span == 0.0:
        return np.full(grid.shape, 0.5 * (low_value + high_value))
    fraction = (projection - projection.min()) / span
    return low_value + fraction * (high_value - low_value)


def gaussian_coherence_patch(grid, center_x, center_y, sigma, depth,
                             background=1.0):
    """A smooth, roughly circular patch of reduced coherence.

    The patch is a Gaussian dip in an otherwise flat background,

    .. math::

        \\gamma(x, y) = \\gamma_{\\text{bg}}
            - A \\exp\\!\\left(
                -\\frac{(x - x_0)^2 + (y - y_0)^2}{2 \\sigma^2}\\right),

    with the amplitude :math:`A` chosen so that the centre reaches exactly
    ``background - depth`` and the result is clipped to ``[0, 1]``.  Unlike a
    hard-edged disc this has no discontinuity, which is what a real lake, wood
    or dune field looks like once the coherence is estimated over a window.

    Parameters
    ----------
    grid : Grid
        Sampling grid.
    center_x, center_y : float
        Centre of the patch, in metres, in the grid's ENU frame.
    sigma : float
        Standard deviation of the Gaussian, in metres.  Larger values give a
        broader patch.  The visible patch is roughly :math:`4\\sigma` across.
    depth : float
        How far below ``background`` the coherence falls at the centre.
    background : float or str, optional
        Coherence far from the patch.  Default ``1.0``.

    Returns
    -------
    ndarray
        Coherence map of shape ``(grid.ny, grid.nx)``.

    Railings
    --------
    The true Gaussian is not clipped: if ``background - depth`` is negative the
    map is truncated at ``0`` over a central plateau.  That is intentional, so
    that asking for a very deep patch gives a genuine no-information core
    rather than a negative coherence.

    Examples
    --------
    >>> from accelerated_unwrap.synthetic.grid import Grid
    >>> grid = Grid.centered(nx=21, ny=21, spacing=100.0)
    >>> gamma = gaussian_coherence_patch(grid, 0.0, 0.0, 200.0, 0.6)
    >>> round(float(gamma[10, 10]), 6)         # the centre of the patch
    0.4
    >>> round(float(gamma[0, 0]), 6)           # far corner, background
    1.0
    >>> bool(np.all(gamma >= 0.0))
    True
    """
    _check_grid(grid)
    sigma_value = _positive_float(sigma, "sigma")
    background_value = coherence_from_level(background)
    depth_value = _positive_float(depth, "depth")
    dx = np.asarray(grid.X, dtype=np.float64) - float(center_x)
    dy = np.asarray(grid.Y, dtype=np.float64) - float(center_y)
    radius_squared = dx * dx + dy * dy
    attenuation = np.exp(-radius_squared / (2.0 * sigma_value ** 2))
    gamma = background_value - depth_value * attenuation
    return np.clip(gamma, 0.0, 1.0)


def decorrelation_stripe(grid, width, coherence=0.2, azimuth_deg=0.0,
                         offset=0.0, background=1.0):
    """An infinite straight band of reduced coherence.

    A stripe is what a long, narrow surface change looks like: a river valley
    that floods, a ploughed strip, an agricultural swath that is harvested
    between the two acquisitions.  It is also a deliberately awkward shape for
    a quality-guided unwrapper, because the low-quality region is not compact
    and cannot be routed around in one move.

    Parameters
    ----------
    grid : Grid
        Sampling grid.
    width : float
        Width of the band in metres, measured across the band.
    coherence : float or str, optional
        Coherence inside the band.  Default ``0.2``.
    azimuth_deg : float, optional
        Azimuth of the band's *long axis*, degrees clockwise from North.
        Default ``0.0``, i.e. the band runs north-south.  The band is
        perpendicular to this direction.
    offset : float, optional
        Signed perpendicular distance, in metres, from the grid origin to the
        centre line of the band.  Default ``0.0`` centres it on the origin.
        A positive offset shifts the band in the direction
        ``azimuth_deg + 90`` (to the east for a north-south band).
    background : float or str, optional
        Coherence outside the band.  Default ``1.0``.

    Returns
    -------
    ndarray
        Coherence map of shape ``(grid.ny, grid.nx)``.

    Examples
    --------
    >>> from accelerated_unwrap.synthetic.grid import Grid
    >>> grid = Grid(nx=11, ny=11, spacing=100.0, x_min=-500.0, y_max=500.0)
    >>> gamma = decorrelation_stripe(grid, width=300.0, coherence=0.3)
    >>> round(float(gamma[5, 5]), 6)           # on the centre line x = 0
    0.3
    >>> round(float(gamma[5, 0]), 6)           # x = -500, outside the band
    1.0
    >>> round(float(gamma[5, 8]), 6)           # x = +300, just outside
    1.0
    """
    _check_grid(grid)
    width_value = _positive_float(width, "width")
    inside = coherence_from_level(coherence)
    outside = coherence_from_level(background)
    theta = np.deg2rad(float(azimuth_deg))
    # Normal to the band: rotate the along-band unit vector by -90 degrees.
    normal_e, normal_n = np.cos(theta), -np.sin(theta)
    distance = normal_e * grid.X + normal_n * grid.Y - float(offset)
    gamma = np.where(np.abs(distance) <= 0.5 * width_value, inside, outside)
    return np.asarray(gamma, dtype=np.float64)


def fault_zone_coherence(grid, trace_x, trace_y, width, coherence=0.2,
                         background=1.0):
    """Reduced coherence in a band following an arbitrary fault trace.

    Real rupture zones are not straight.  The trace is given as a polyline, and
    every pixel within ``width / 2`` of the polyline has its coherence lowered.
    This is the pattern to use when the deformation source *is* the
    decorrelation: a surface-breaking fault destroys coherence along its trace
    while the surrounding far field stays good, so the aliasing and the
    discontinuity coincide, which is the hardest realistic case for branch-cut
    and minimum-cost-flow unwrappers.

    Parameters
    ----------
    grid : Grid
        Sampling grid.
    trace_x, trace_y : array_like
        Vertices of the fault trace in metres, in the grid's ENU frame.  At
        least two vertices are required.  They need not lie inside the grid; the
        distance is evaluated everywhere.
    width : float
        Full width of the damaged band in metres.
    coherence : float or str, optional
        Coherence inside the band.  Default ``0.2``.
    background : float or str, optional
        Coherence outside the band.  Default ``1.0``.

    Returns
    -------
    ndarray
        Coherence map of shape ``(grid.ny, grid.nx)``.

    Raises
    ------
    ValueError
        If fewer than two trace vertices are given, or if ``trace_x`` and
        ``trace_y`` differ in length.

    Examples
    --------
    >>> from accelerated_unwrap.synthetic.grid import Grid
    >>> grid = Grid(nx=11, ny=11, spacing=100.0, x_min=-500.0, y_max=500.0)
    >>> gamma = fault_zone_coherence(grid, [0.0, 0.0], [-400.0, 400.0],
    ...                              width=200.0, coherence=0.25)
    >>> round(float(gamma[5, 5]), 6)           # a pixel on the trace
    0.25
    >>> round(float(gamma[5, 0]), 6)           # far from the trace
    1.0
    """
    _check_grid(grid)
    width_value = _positive_float(width, "width")
    inside = coherence_from_level(coherence)
    outside = coherence_from_level(background)
    distance = _distance_to_segments(grid.X, grid.Y, trace_x, trace_y)
    gamma = np.where(distance <= 0.5 * width_value, inside, outside)
    return np.asarray(gamma, dtype=np.float64)


def random_coherence(grid, mean=0.75, sigma=0.12, correlation_length=500.0,
                     rng=None):
    """A smooth, spatially correlated random coherence field.

    Real coherence maps are patchy at the scale of the estimation window and
    the land cover, not white noise.  This function draws a white Gaussian
    field on the grid, smooths it with a Gaussian kernel of standard deviation
    ``correlation_length`` metres, rescales it to zero mean and unit variance,
    and then maps it to coherence by

    .. math::

        \\gamma = \\text{clip}\\left(\\mu + \\sigma w, 0, 1\\right),

    where :math:`w` is the smoothed, rescaled field.

    This is a *statistical* model, not a physical one.  It reproduces the right
    spatial texture and nothing more; do not read a physical meaning into a
    particular realisation.  Its purpose is to give the noise and masking
    machinery something heterogeneous and reproducible to work on.

    Parameters
    ----------
    grid : Grid
        Sampling grid.
    mean : float, optional
        Mean coherence of the field.  Default ``0.75``.
    sigma : float, optional
        Standard deviation before clipping.  Default ``0.12``.
    correlation_length : float, optional
        Smoothing length in metres.  Default ``500.0``.
    rng : numpy.random.Generator, optional
        Random source.  A default generator is created if omitted, in which
        case the result differs between calls.  Pass
        ``np.random.default_rng(seed)`` for a reproducible field.

    Returns
    -------
    ndarray
        Coherence map of shape ``(grid.ny, grid.nx)``, clipped to ``[0, 1]``.

    Examples
    --------
    >>> from accelerated_unwrap.synthetic.grid import Grid
    >>> grid = Grid(nx=64, ny=64, spacing=100.0)
    >>> rng = np.random.default_rng(20240101)
    >>> gamma = random_coherence(grid, rng=rng)
    >>> float(gamma.mean()) > 0.6 and float(gamma.mean()) < 0.9
    True
    >>> same = random_coherence(grid, rng=np.random.default_rng(20240101))
    >>> bool(np.array_equal(gamma, same))
    True
    """
    _check_grid(grid)
    if rng is None:
        rng = np.random.default_rng()
    mean_value = _check_coherence(float(mean), "mean")
    sigma_value = _positive_float(sigma, "sigma")
    length_value = _positive_float(correlation_length, "correlation_length")
    spacing = float(grid.spacing)
    if spacing <= 0.0:
        raise ValueError(
            "grid.spacing must be positive, got {!r}".format(grid.spacing))

    white = rng.normal(size=grid.shape)
    sigma_pixels = length_value / spacing
    if sigma_pixels < 1e-3:
        field = white
    else:
        # Imported lazily so that the rest of the module has no SciPy
        # requirement beyond the package's declared dependency.
        from scipy.ndimage import gaussian_filter
        field = gaussian_filter(white, sigma_pixels, mode="reflect")
    spread = float(field.std())
    if spread > 0.0:
        field = (field - field.mean()) / spread
    else:                                    # pragma: no cover - degenerate
        field = np.zeros(grid.shape)
    return np.clip(mean_value + sigma_value * field, 0.0, 1.0)


def circular_no_data_mask(grid, center_x, center_y, radius):
    """A boolean mask of a circular region that carries no data at all.

    Water, radar shadow, layover and image edges all produce regions where the
    interferometric phase is not merely noisy but meaningless.  Such pixels are
    ``True`` in this mask.  Recording them separately from a low coherence
    value is essential: an unwrapper may legitimately decide to low-pass a
    coherence-0.2 pixel and trust it a little, but it must never interpolate
    across a no-data pixel without saying so.

    Parameters
    ----------
    grid : Grid
        Sampling grid.
    center_x, center_y : float
        Centre of the region, in metres, in the grid's ENU frame.
    radius : float
        Radius of the region, in metres.

    Returns
    -------
    ndarray
        Boolean array of shape ``(grid.ny, grid.nx)``; ``True`` where there is
        no data.

    Examples
    --------
    >>> from accelerated_unwrap.synthetic.grid import Grid
    >>> grid = Grid.centered(nx=21, ny=21, spacing=100.0)
    >>> mask = circular_no_data_mask(grid, 0.0, 0.0, 250.0)
    >>> bool(mask[10, 10])
    True
    >>> bool(mask[0, 0])
    False
    >>> int(mask.sum())
    21
    """
    _check_grid(grid)
    radius_value = _positive_float(radius, "radius")
    dx = np.asarray(grid.X, dtype=np.float64) - float(center_x)
    dy = np.asarray(grid.Y, dtype=np.float64) - float(center_y)
    return (dx * dx + dy * dy) <= radius_value ** 2


def combine_coherence(*maps):
    """Combine several coherence maps by multiplication.

    Independent decorrelation mechanisms each remove a fraction of the
    correlation, so their effects compound multiplicatively:

    .. math::

        \\gamma_{\\text{total}} = \\prod_i \\gamma_i .

    ``nan`` propagates: one no-data map makes the pixel no-data overall, which
    is the correct behaviour.

    Parameters
    ----------
    *maps : array_like
        Two or more coherence arrays of identical shape.

    Returns
    -------
    ndarray
        The product, of the same shape as the inputs.

    Raises
    ------
    ValueError
        If fewer than two maps are given, or their shapes differ.

    Notes
    -----
    Multiplication is the physically motivated choice, but it is worth knowing
    what it does numerically.  Two moderately degraded maps, each at 0.6, give
    0.36 -- a combined coherence nobody would call moderate.  That is correct
    and it is easy to underestimate: decorrelation is unforgiving.

    Examples
    --------
    >>> a = np.array([[1.0, 0.5], [0.8, 0.0]])
    >>> b = np.array([[0.5, 0.5], [0.5, 1.0]])
    >>> combine_coherence(a, b).tolist()
    [[0.5, 0.25], [0.4, 0.0]]
    """
    if len(maps) < 2:
        raise ValueError(
            "combine_coherence needs at least two maps, got {}".format(
                len(maps)))
    arrays = [np.asarray(m, dtype=np.float64) for m in maps]
    shape = arrays[0].shape
    for index, array in enumerate(arrays[1:], start=1):
        if array.shape != shape:
            raise ValueError(
                "all coherence maps must have the same shape; map 0 has {} "
                "but map {} has {}".format(shape, index, array.shape))
    result = np.ones(shape, dtype=np.float64)
    for array in arrays:
        result = result * array
    return result
