"""The Mogi point-source model for volcano and reservoir deformation.

What the model is
-----------------
Mogi (1958) asked a very simple question: if you pump a small spherical
pressure source into an elastic half-space, what does the ground surface do?
He found that the answer depends on the source only through its *volume
change* ``dV`` [m^3] and its depth ``d`` [m] -- not on its radius or its
pressure -- as long as the source is small compared with ``d``.  That means
the model has a wonderfully small parameter set, which is why it is still the
workhorse model for inflating volcanoes and compacting reservoirs 65 years
later.

The equations
-------------
Let ``(x0, y0)`` be the horizontal position of the source and define

.. math::

    r^2 = (x - x_0)^2 + (y - y_0)^2, \\qquad R^2 = r^2 + d^2 ,

so that ``R`` is the straight-line distance from the observation point to the
source.  With :math:`\\nu` the Poisson ratio of the half-space, the surface
displacement in East-North-Up components is

.. math::

    u_E &= \\frac{(1-\\nu)\\,\\Delta V}{\\pi} \\frac{x - x_0}{R^3} \\\\
    u_N &= \\frac{(1-\\nu)\\,\\Delta V}{\\pi} \\frac{y - y_0}{R^3} \\\\
    u_U &= \\frac{(1-\\nu)\\,\\Delta V}{\\pi} \\frac{d}{R^3}

Three properties are worth memorising, because they are exactly what makes the
model useful as a *test* for an unwrapping algorithm:

1. **Sign.**  With a positive :math:`\\Delta V` (inflation) the vertical
   component points up everywhere, since ``d > 0`` and ``R > 0``.  Deflation
   is simply a negative :math:`\\Delta V`.
2. **Symmetry.**  The vertical field is radially symmetric about
   ``(x0, y0)``; the horizontal field is radial, with magnitude zero directly
   above the source and growing without bound for a shallow source.
3. **Decay.**  Far from the source ``u ~ 1 / r^2``.  This is a *long-range*
   field, quite different from the ``1 / r`` decay of a dislocation.  If your
   unwrapping algorithm produces an error that lives on the fringes far from
   the source, this is the shape to compare against.

The peak uplift, directly above the source, is

.. math::

    u_U(x_0, y_0) = \\frac{(1 - \\nu)\\,\\Delta V}{\\pi d^2}

which is a very convenient closed form to check an implementation against.

A note on the factor in front
-----------------------------
Different papers define the strength of the source differently, and reviewers
of synthetic datasets should be told which convention is used.  This module
follows the normalisation requested in the specification for this package:
``(1 - nu) * dV / pi``.  Some texts instead write the whole thing in terms of a
source radius ``a`` and pressure change ``dP`` and end up with a factor
``3 / (4 pi)`` in front of a differently-defined volume.  The physics
(``1 / R^3`` decay, radial symmetry, ``d`` in the vertical component) is
identical; only the interpretation of the scalar changes.  Whenever you
compare absolute amplitudes with another code, compare the *ratio* of the
components, not their absolute values.

Assumptions and limitations
---------------------------
* The half-space is **homogeneous, isotropic and linearly elastic**.
* The source is a point.  This fails close to the source when the source
  radius is comparable with its depth; use a finite source
  (:mod:`parvaneh.synthetic.okada`) in that regime.
* **No topography**: the free surface is flat.  On a real volcano this is a
  significant simplification.
* The model is *static*: no viscoelastic relaxation, no magma compressibility
  over time.
* Only surface displacements are returned, because that is what a satellite
  measures.

References
----------
Mogi, K. (1958).  Relations between the eruptions of various volcanoes and the
deformations of the ground surfaces around them.  *Bulletin of the Earthquake
Research Institute, University of Tokyo*, 36, 99-134.
"""

import numpy as np

from ._checks import finite_float as _finite
from ._checks import poisson_ratio as _validate_poisson_ratio
from ._checks import positive_float as _require_positive

__all__ = [
    "MOGI_STYLES",
    "mogi_point_displacement",
    "mogi_displacement",
    "mogi_displacement_multi",
    "random_mogi_source",
]


# Parameter ranges used when sampling a source of a given "style".  These are
# ranges, not fixed values, and every scenario samples *within* the range with
# a given random seed, so no two datasets are identical unless you ask for
# them to be.  The ranges are chosen so that a 512 x 512 grid at 30 m spacing
# (about 15 km across) contains a few fringes but not so many that the field
# aliases into pure noise.
MOGI_STYLES = {
    "inflation": {
        "depth_range": (2000.0, 6000.0),
        "volume_range": (5.0e6, 40.0e6),
        "centre_range": 0.25,
    },
    "deflation": {
        "depth_range": (2000.0, 6000.0),
        "volume_range": (-40.0e6, -5.0e6),
        "centre_range": 0.25,
    },
    "shallow": {
        "depth_range": (500.0, 1500.0),
        "volume_range": (0.5e6, 5.0e6),
        "centre_range": 0.25,
    },
    "deep": {
        "depth_range": (8000.0, 15000.0),
        "volume_range": (40.0e6, 200.0e6),
        "centre_range": 0.25,
    },
    "weak": {
        "depth_range": (3000.0, 5000.0),
        "volume_range": (0.5e6, 2.0e6),
        "centre_range": 0.25,
    },
    "strong": {
        "depth_range": (3000.0, 5000.0),
        "volume_range": (40.0e6, 80.0e6),
        "centre_range": 0.25,
    },
    "centred": {
        "depth_range": (3000.0, 5000.0),
        "volume_range": (5.0e6, 20.0e6),
        "centre_range": 0.0,
    },
    "edge": {
        "depth_range": (3000.0, 5000.0),
        "volume_range": (5.0e6, 20.0e6),
        "centre_range": 1.0,
    },
    "multiple": {
        "depth_range": (2000.0, 8000.0),
        "volume_range": (2.0e6, 30.0e6),
        "centre_range": 0.8,
    },
}


def mogi_point_displacement(x, y, source_x, source_y, depth, delta_volume,
                            poisson_ratio=0.25):
    """Displacement at ``(x, y)`` from a single Mogi source.

    This is the scalar workhorse; :func:`mogi_displacement` is a thin wrapper
    over it that accepts a :class:`~parvaneh.synthetic.grid.Grid`.

    Parameters
    ----------
    x, y : array_like
        Observation point coordinates in metres, East and North.  Any shape,
        and broadcastable against each other.
    source_x, source_y : float
        Horizontal position of the source, in metres.
    depth : float
        Depth of the source below the free surface, in metres.  Must be
        strictly positive: a zero-depth source makes the field singular.
    delta_volume : float
        Source volume change in cubic metres.  Positive means inflation.
    poisson_ratio : float, optional
        Poisson ratio of the half-space.  ``0.25`` is the usual choice for
        crustal rock; ``0.5`` (perfectly incompressible) is not allowed.

    Returns
    -------
    u_e, u_n, u_u : ndarray
        East, North and Up displacement components in metres, with the
        broadcast shape of ``x`` and ``y``.

    Notes
    -----
    The computation is done in ``float64`` throughout.  ``R`` is never zero
    because ``depth > 0``, so no singularity guard is needed beyond the
    validation above.
    """
    source_x = _finite(source_x, "source_x")
    source_y = _finite(source_y, "source_y")
    depth = _require_positive(depth, "depth")
    delta_volume = _finite(delta_volume, "delta_volume")
    poisson_ratio = _validate_poisson_ratio(poisson_ratio)

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    dx = x - source_x
    dy = y - source_y

    # R^3 rather than R**3: one multiply instead of a power, and it makes the
    # intent (a distance cubed) obvious at a glance.
    r_squared = dx * dx + dy * dy + depth * depth
    r_cubed = r_squared * np.sqrt(r_squared)

    scale = (1.0 - poisson_ratio) * delta_volume / (np.pi * r_cubed)

    u_e = scale * dx
    u_n = scale * dy
    u_u = scale * depth
    return u_e, u_n, u_u


def mogi_displacement(grid, source_x, source_y, depth, delta_volume,
                      poisson_ratio=0.25):
    """Mogi displacement field sampled on a :class:`Grid`.

    Parameters
    ----------
    grid : Grid
        Sampling grid.  Its :attr:`~Grid.X` and :attr:`~Grid.Y` arrays are
        used, so the returned arrays have shape ``(grid.ny, grid.nx)``.
    source_x, source_y, depth, delta_volume, poisson_ratio
        See :func:`mogi_point_displacement`.  ``poisson_ratio`` and
        ``delta_volume`` may be scalars or, for the volume, arrays matching
        the grid if you want a spatially varying source (rare, but harmless).

    Returns
    -------
    u_e, u_n, u_u : ndarray
        ENU displacement components in metres, each of shape
        ``(grid.ny, grid.nx)``.

    Examples
    --------
    >>> from parvaneh.synthetic.grid import Grid
    >>> grid = Grid.centered(nx=3, ny=3, spacing=1000.0)
    >>> u_e, u_n, u_u = mogi_displacement(grid, 0.0, 0.0, 2000.0, 1.0e7)
    >>> round(float(u_u[1, 1]), 6)          # directly above the source
    0.596831
    """
    _check_grid(grid)
    u_e, u_n, u_u = mogi_point_displacement(
        grid.X, grid.Y, source_x, source_y, depth, delta_volume,
        poisson_ratio=poisson_ratio)
    return u_e, u_n, u_u


def mogi_displacement_multi(grid, sources, poisson_ratio=0.25):
    """Sum the displacement of several Mogi sources.

    The elastic half-space is linear, so the displacement of a collection of
    sources is exactly the sum of the individual displacements.  Nothing is
    approximated here, which makes this a clean way to build a "hard" test
    case with several overlapping deformation patterns.

    Parameters
    ----------
    grid : Grid
        Sampling grid.
    sources : sequence of dict
        Each dictionary holds the keyword arguments of
        :func:`mogi_displacement`: ``source_x``, ``source_y``, ``depth`` and
        ``delta_volume``, and optionally ``poisson_ratio``.  A dict missing a
        required key raises :exc:`ValueError` naming the key.
    poisson_ratio : float, optional
        Default Poisson ratio for sources that do not specify their own.

    Returns
    -------
    u_e, u_n, u_u : ndarray
        Total ENU displacement, shape ``(grid.ny, grid.nx)``.

    Raises
    ------
    ValueError
        If ``sources`` is empty or a source dictionary is incomplete.

    Examples
    --------
    Multiple sources are additive, so a pair of identical sources cancels:

    >>> from parvaneh.synthetic.grid import Grid
    >>> grid = Grid.centered(nx=4, ny=4, spacing=500.0)
    >>> src = dict(source_x=0.0, source_y=0.0, depth=3000.0,
    ...            delta_volume=1.0e7)
    >>> cancel = dict(src, delta_volume=-1.0e7)
    >>> u_e, u_n, u_u = mogi_displacement_multi(grid, [src, cancel])
    >>> bool(np.allclose(u_u, 0.0))
    True
    """
    _check_grid(grid)
    if sources is None:
        raise ValueError("sources must be a non-empty sequence of dicts")
    sources = list(sources)
    if not sources:
        raise ValueError("sources must be a non-empty sequence of dicts")

    required = ("source_x", "source_y", "depth", "delta_volume")
    u_e = np.zeros(grid.shape, dtype=np.float64)
    u_n = np.zeros(grid.shape, dtype=np.float64)
    u_u = np.zeros(grid.shape, dtype=np.float64)

    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ValueError(
                "sources[{}] must be a dict, got {!r}".format(
                    index, type(source).__name__))
        missing = [key for key in required if key not in source]
        if missing:
            raise ValueError(
                "sources[{}] is missing {}".format(
                    index, ", ".join(sorted(missing))))
        e_i, n_i, u_i = mogi_displacement(
            grid,
            source["source_x"], source["source_y"], source["depth"],
            source["delta_volume"],
            poisson_ratio=source.get("poisson_ratio", poisson_ratio))
        u_e += e_i
        u_n += n_i
        u_u += u_i

    return u_e, u_n, u_u


def random_mogi_source(grid, style="inflation", rng=None):
    """Sample a random Mogi source whose deformation fits on ``grid``.

    The parameters are drawn from the ranges in :data:`MOGI_STYLES` rather
    than hard-coded, so a batch of datasets covers a genuine variety of
    amplitudes and wavelengths.  Sampling is done with the supplied
    :class:`numpy.random.Generator`, which means the result is reproducible
    from a seed (see the package-level documentation on reproducibility).

    Parameters
    ----------
    grid : Grid
        Grid the source will be evaluated on.  It is used to place the source
        horizontally: ``centre_range`` in :data:`MOGI_STYLES` is the fraction
        of the half-width over which the source centre is allowed to roam,
        measured from the grid centre.  ``0`` puts the source dead centre,
        ``1`` allows it to reach the grid edge.
    style : str, optional
        Key of :data:`MOGI_STYLES`.  Default ``"inflation"``.
    rng : numpy.random.Generator, optional
        Random source.  A default generator is created if omitted, in which
        case the result differs between calls.

    Returns
    -------
    dict
        Keyword arguments ready to be passed to :func:`mogi_displacement`,
        plus a ``style`` entry recording how it was generated.

    Raises
    ------
    ValueError
        If ``style`` is not a known key of :data:`MOGI_STYLES`.

    Examples
    --------
    >>> from parvaneh.synthetic.grid import Grid
    >>> grid = Grid.centered(nx=64, ny=64, spacing=100.0)
    >>> rng = np.random.default_rng(20240101)
    >>> src = random_mogi_source(grid, "centred", rng=rng)
    >>> src["source_x"] == 0.0 and src["source_y"] == 0.0
    True
    """
    _check_grid(grid)
    if style not in MOGI_STYLES:
        raise ValueError(
            "unknown Mogi style {!r}; expected one of {}".format(
                style, ", ".join(sorted(MOGI_STYLES))))
    if rng is None:
        rng = np.random.default_rng()

    spec = MOGI_STYLES[style]
    depth = float(rng.uniform(*spec["depth_range"]))
    volume = float(rng.uniform(*spec["volume_range"]))

    # Allowed offset from the grid centre, as a fraction of the half-width.
    reach = float(spec["centre_range"])
    half_width = 0.5 * grid.width
    half_height = 0.5 * grid.height
    if reach <= 0.0:
        source_x = grid.center[0]
        source_y = grid.center[1]
    else:
        source_x = grid.center[0] + float(rng.uniform(-1.0, 1.0)) * reach * half_width
        source_y = grid.center[1] + float(rng.uniform(-1.0, 1.0)) * reach * half_height

    return {
        "source_x": source_x,
        "source_y": source_y,
        "depth": depth,
        "delta_volume": volume,
        "style": style,
    }


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------
def _check_grid(grid):
    """Reject anything that is not a :class:`Grid`-like object with X and Y."""
    if not hasattr(grid, "X") or not hasattr(grid, "Y"):
        raise ValueError(
            "grid must be a Grid instance with X and Y arrays, got {!r}".format(
                type(grid).__name__))
