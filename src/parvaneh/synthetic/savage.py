"""The Savage-Burford screw-dislocation model for interseismic deformation.

What the problem is
-------------------
Two blocks of crust slide past each other along a vertical fault.  Deep down
the rock is hot enough to creep, so it slides steadily at the long-term slip
rate ``V``.  Near the surface the rock is cold and brittle, and friction locks
the fault: the two sides are stuck together and the relative motion has to be
taken up by *elastic strain* in the surrounding rock instead.  An InSAR
interferogram spanning a few years sees that strain, not the slip.

This locked-versus-creeping division is the single most important idea in
interseismic geodesy, and the model of Savage and Burford (1973) captures it
with two numbers: the long-term slip rate ``V`` and the *locking depth* ``D``,
which is the depth at which the fault stops being stuck.

The equations
-------------
Measuring ``x`` as the perpendicular distance from the fault trace, the
fault-parallel surface velocity is

.. math::

    v(x) = \\frac{V}{\\pi} \\arctan\\!\\left(\\frac{x}{D}\\right) .

That is the whole model.  Three consequences deserve to be memorised:

1. **Far field.**  As :math:`x \\to +\\infty`, :math:`v \\to +V/2`; as
   :math:`x \\to -\\infty`, :math:`v \\to -V/2`.  The two plates end up moving
   at :math:`\\pm V/2`, so their *relative* velocity is the full :math:`V`.
   Half the slip rate is "lost" to the choice of reference frame, which is why
   an interferogram is only ever sensitive to the relative motion.
2. **Across the fault.**  :math:`v(0) = 0`.  The velocity is continuous through
   the fault trace: there is no slip at the surface, because the fault is
   locked.  This is exactly what distinguishes an interseismic field from a
   coseismic one, where the surface displacement *jumps* by the slip.
3. **Decay.**  The strain is concentrated near the fault and decays over a
   distance of order ``D``.  A locking depth of 5 km and a locking depth of
   20 km produce the same far-field velocity but a completely different fringe
   pattern near the fault, which is the observable that lets you measure
   ``D`` from space.  Note also that :math:`v` saturates: at a fixed distance
   from the fault, the velocity you see grows only as :math:`\\arctan(1/D)`.

Sign convention
---------------
Strike is measured in degrees clockwise from North, exactly as elsewhere in
this package, so strike 0 means the fault trace runs north-south and strike 90
means it runs east-west.  Define the "right-of-strike" direction as
``strike + 90`` degrees, i.e. the direction you get by turning clockwise from
the strike direction:

* strike ``0`` (north)  -> right-of-strike is **east**
* strike ``90`` (east)  -> right-of-strike is **south**
* strike ``180`` (south) -> right-of-strike is **west**
* strike ``270`` (west) -> right-of-strike is **north**

The perpendicular coordinate ``x`` is positive on the right-of-strike side,
and the velocity everywhere points along ``+strike`` with the magnitude given
by the formula above.  So with a positive ``V`` the block on the
right-of-strike side slides in the ``+strike`` direction and the other block in
the ``-strike`` direction.

For a fault striking north (``strike = 0``) that means the **eastern** block
moves north.  Stand on the western side and look across the fault: the far side
moves to your left, so this is a **sinistral** (left-lateral) sense of motion.
To build a dextral fault, negate ``slip_rate``.

One trap worth flagging, because it catches people out: reversing ``strike`` by
180 degrees changes nothing at all.  The trace is the same line, "right of
strike" flips, and the two flips cancel exactly.  So ``strike = 0`` and
``strike = 180`` produce identical velocity fields and the same dextral /
sinistral label; if you want the opposite sense of motion you must negate
``slip_rate``.  For the same reason this module lets ``slip_rate`` be
**signed** rather than trying to encode the sense in the strike.  Every
function documents its convention and none of them guess.

Speed versus displacement
-------------------------
The screw-dislocation model is naturally a *velocity* field, because the
interseismic process is a slow steady creep.  Radio interferometry, however,
measures a *displacement* accumulated between two acquisition dates, so the
time interval has to be supplied.  That is why this module has both

* ``savage_velocity`` -> metres per year, and
* ``savage_displacement`` -> metres, the velocity multiplied by an interval.

Nothing here is a coseismic model.  A coseismic (earthquake) displacement is a
*step*, and comes from the Mogi or Okada models; mixing the two up is a
classic source of sign and amplitude errors, so the two live in different
modules and never call each other.

Assumptions and limitations
---------------------------
* **Vertical fault**, so the trace at the surface is directly above the
  locked patch.  A dipping fault makes the surface velocity field asymmetric
  and needs a different model.
* **Screw dislocation**: pure strike-slip.  There is no vertical motion at all
  (``v_U`` is identically zero) and no fault-normal motion (``v_N`` is zero in
  the fault frame).  Real subduction and normal faults violate this badly, so
  do not use this module for a megathrust.
* **Two-dimensional**: the model assumes the fault is infinitely long, so the
  velocity does not vary along strike.  Real faults end, and the ends of a
  segment look different; see the note in :func:`savage_velocity`.
* **Steady state**: the model gives the velocity field once the earthquake
  cycle has settled, and says nothing about the postseismic transient after a
  large event.
* **Uniform elastic half-space**, same as everywhere else in this package.

References
----------
Savage, J. C., & Burford, R. O. (1973).  Geodetic determination of relative
plate motion in central California.  *Journal of Geophysical Research*,
78(5), 832-845.  doi:10.1029/JB078i005p00832

Savage, J. C. (1983).  A dislocation model of strain accumulation and release
at a subduction zone.  *Journal of Geophysical Research*, 88(B6), 4984-4996.
"""

import numpy as np

from ._checks import finite_float as _finite
from ._checks import positive_float as _require_positive

__all__ = [
    "SAVAGE_STYLES",
    "savage_velocity_profile",
    "savage_velocity",
    "savage_displacement",
    "random_savage_source",
]


# Parameter ranges used when sampling an interseismic fault of a given
# "style".  As with the Mogi styles these are ranges, not fixed values: a
# scenario samples within them from a seeded generator, so a batch of datasets
# covers a spread of locking depths and slip rates rather than one number
# repeated.  The depths bracket the range that is actually observed on
# continental strike-slip faults (roughly 5-20 km).
SAVAGE_STYLES = {
    "shallow_locking": {
        "locking_depth_range": (3000.0, 6000.0),
        "slip_rate_range": (0.02, 0.04),
        "strike_range": (0.0, 360.0),
        "centre_range": 0.25,
    },
    "deep_locking": {
        "locking_depth_range": (12000.0, 25000.0),
        "slip_rate_range": (0.02, 0.04),
        "strike_range": (0.0, 360.0),
        "centre_range": 0.25,
    },
    "slow_slip": {
        "locking_depth_range": (8000.0, 15000.0),
        "slip_rate_range": (0.002, 0.008),
        "strike_range": (0.0, 360.0),
        "centre_range": 0.25,
    },
    "fast_slip": {
        "locking_depth_range": (8000.0, 15000.0),
        "slip_rate_range": (0.06, 0.12),
        "strike_range": (0.0, 360.0),
        "centre_range": 0.25,
    },
    "north_south": {
        "locking_depth_range": (8000.0, 15000.0),
        "slip_rate_range": (0.02, 0.04),
        "strike_range": (0.0, 0.0),
        "centre_range": 0.25,
    },
    "east_west": {
        "locking_depth_range": (8000.0, 15000.0),
        "slip_rate_range": (0.02, 0.04),
        "strike_range": (90.0, 90.0),
        "centre_range": 0.25,
    },
    "oblique": {
        "locking_depth_range": (8000.0, 15000.0),
        "slip_rate_range": (0.02, 0.04),
        "strike_range": (30.0, 60.0),
        "centre_range": 0.25,
    },
}


def savage_velocity_profile(x, slip_rate, locking_depth):
    """The one-dimensional Savage-Burford velocity profile ``v(x)``.

    This is the scalar heart of the model.  Everything else in the module is
    bookkeeping to place this profile on a map and to multiply it by time.

    .. math::

        v(x) = \\frac{V}{\\pi} \\arctan\\!\\left(\\frac{x}{D}\\right)

    Parameters
    ----------
    x : array_like
        Signed perpendicular distance from the fault trace, in metres,
        positive on the right-of-strike side (see the module docstring).
        Any shape.
    slip_rate : float
        Long-term slip rate ``V`` of the fault, in metres per year.  Signed:
        a negative value reverses the sense of motion.
    locking_depth : float
        Locking depth ``D`` in metres.  Must be strictly positive.

    Returns
    -------
    ndarray
        Fault-parallel velocity in metres per year, the same shape as ``x``.

    Raises
    ------
    ValueError
        If ``slip_rate`` is not finite or ``locking_depth`` is not positive.

    Examples
    --------
    The profile is odd in ``x``, so the two sides move in opposite directions:

    >>> savage_velocity_profile(0.0, 0.03, 10000.0)
    0.0
    >>> round(float(savage_velocity_profile(5000.0, 0.03, 10000.0)), 6)
    0.004428
    >>> round(float(savage_velocity_profile(-5000.0, 0.03, 10000.0)), 6)
    -0.004428

    Far from the fault the two plates move at ``+V/2`` and ``-V/2``:

    >>> far = float(savage_velocity_profile(1.0e9, 0.03, 10000.0))
    >>> bool(abs(far - 0.015) < 1.0e-6)
    True
    >>> bool(abs(float(savage_velocity_profile(-1.0e9, 0.03, 10000.0)) + 0.015)
    ...      < 1.0e-6)
    True

    The value at ``x = D`` is a fixed fraction of the far field, independent of
    both parameters:

    >>> round(float(savage_velocity_profile(10000.0, 0.03, 10000.0)), 9)
    0.0075

    A halved locking depth makes the strain concentrate over half the distance,
    but leaves the velocities at a fixed *fraction* of the far field untouched:

    >>> ratio_a = savage_velocity_profile(5000.0, 1.0, 5000.0)
    >>> ratio_b = savage_velocity_profile(5000.0, 1.0, 10000.0)
    >>> bool(ratio_a > ratio_b)
    True
    """
    slip = _finite(slip_rate, "slip_rate")
    depth = _require_positive(locking_depth, "locking_depth")
    x_array = np.asarray(x, dtype=np.float64)
    return (slip / np.pi) * np.arctan(x_array / depth)


def savage_velocity(grid, strike_deg, locking_depth, slip_rate,
                    center_x=0.0, center_y=0.0):
    """Fault-parallel interseismic velocity over a map.

    The one-dimensional profile of :func:`savage_velocity_profile` is applied
    to the perpendicular distance from the fault trace, and the resulting
    scalar is then pointed along the strike direction.  That is the whole
    ``Savage -> map`` recipe:

    1. project each observation point onto the fault normal to get ``x``,
    2. evaluate the profile,
    3. point the result along ``+strike``.

    Because the model is two-dimensional, the velocity does not change along
    the trace.  On a real fault the ends of the locked segment taper off and
    the pattern rotates near a step-over or a bend; if a benchmark needs that,
    compose several segments with :func:`savage_velocity` and add the results.

    Parameters
    ----------
    grid : Grid
        Sampling grid.
    strike_deg : float
        Strike of the fault in degrees clockwise from North.
    locking_depth : float
        Locking depth ``D`` in metres.  Must be strictly positive.
    slip_rate : float
        Long-term slip rate ``V`` in metres per year.  Signed.
    center_x, center_y : float, optional
        A point that the fault trace passes through, in metres in the grid's
        ENU frame.  Default ``(0, 0)``, which is the grid centre for a grid
        built with :meth:`Grid.centered`.

    Returns
    -------
    v_e, v_n, v_u : ndarray
        Velocity in metres per year, shape ``(grid.ny, grid.nx)``.  ``v_u`` is
        identically zero: a screw dislocation produces no vertical motion.

    Raises
    ------
    ValueError
        If ``grid`` is not grid-like, if ``strike_deg`` is not finite, or if
        the locking depth or slip rate are rejected by the validators.

    Examples
    --------
    A north-south fault striking ``0`` has "right of strike" to the east, so
    the eastern half of the map moves north and the western half moves south:

    >>> from parvaneh.synthetic.grid import Grid
    >>> grid = Grid.centered(nx=5, ny=5, spacing=1000.0)
    >>> v_e, v_n, v_u = savage_velocity(grid, 0.0, 10000.0, 0.03)
    >>> bool(np.all(v_n[0, 3:] > 0.0))          # east of the fault: northwards
    True
    >>> bool(np.all(v_n[0, :2] < 0.0))          # west of the fault: southwards
    True
    >>> bool(np.allclose(v_e, 0.0)) and bool(np.allclose(v_u, 0.0))
    True

    Rotating the strike by 90 degrees rotates the whole field by 90 degrees, so
    the magnitudes are the same but rearranged -- and now it is ``v_n`` that
    vanishes instead of ``v_e``:

    >>> e_e, e_n, e_u = savage_velocity(grid, 90.0, 10000.0, 0.03)
    >>> bool(np.allclose(e_n, 0.0))
    True
    >>> bool(np.allclose(np.sort(np.abs(e_e).ravel()),
    ...                  np.sort(np.abs(v_n).ravel())))
    True

    Reversing the strike by 180 degrees changes nothing: the fault trace is the
    same line, "right of strike" flips, and the two flips cancel.

    >>> f_e, f_n, _ = savage_velocity(grid, 180.0, 10000.0, 0.03)
    >>> bool(np.allclose(f_n, v_n))
    True

    Reversing the sign of the slip rate, by contrast, reverses the sense of
    motion:

    >>> g_e, g_n, _ = savage_velocity(grid, 0.0, 10000.0, -0.03)
    >>> bool(np.allclose(g_n, -v_n))
    True
    """
    _check_grid(grid)
    strike = _finite(strike_deg, "strike_deg")
    depth = _require_positive(locking_depth, "locking_depth")
    slip = _finite(slip_rate, "slip_rate")
    origin_x = _finite(center_x, "center_x")
    origin_y = _finite(center_y, "center_y")

    theta = np.deg2rad(strike)
    # Unit vector along the fault trace, in (East, North) components.  An
    # azimuth theta measured clockwise from North has east component sin(theta)
    # and north component cos(theta).
    strike_e, strike_n = np.sin(theta), np.cos(theta)
    # Unit vector 90 degrees clockwise from the strike, i.e. the
    # right-of-strike direction.  Rotating an azimuth clockwise by 90 degrees
    # maps (sin, cos) to (cos, -sin).
    normal_e, normal_n = np.cos(theta), -np.sin(theta)

    # Signed perpendicular distance, positive on the right-of-strike side.
    x = (np.asarray(grid.X, dtype=np.float64) - origin_x) * normal_e \
        + (np.asarray(grid.Y, dtype=np.float64) - origin_y) * normal_n

    speed = savage_velocity_profile(x, slip, depth)
    v_e = speed * strike_e
    v_n = speed * strike_n
    v_u = np.zeros(grid.shape, dtype=np.float64)
    return v_e, v_n, v_u


def savage_displacement(grid, strike_deg, locking_depth, slip_rate,
                        interval, center_x=0.0, center_y=0.0):
    """Interseismic displacement accumulated over a time interval.

    Simply the velocity field of :func:`savage_velocity` multiplied by
    ``interval``.  It is the same physics; the reason for a separate function
    is that interferometric phase depends on the *displacement* between two
    acquisition dates, and a caller who forgets to multiply by time gets a
    field that is wrong by a factor of years.

    The result is the *interseismic* displacement: a smooth accumulation.  It
    is **not** the coseismic offset of an earthquake, which would be a step
    across the fault.  See :mod:`parvaneh.synthetic.mogi` and
    :mod:`parvaneh.synthetic.okada` for those.

    Parameters
    ----------
    grid : Grid
        Sampling grid.
    strike_deg : float
        Strike of the fault in degrees clockwise from North.
    locking_depth : float
        Locking depth ``D`` in metres.
    slip_rate : float
        Long-term slip rate ``V`` in metres per year.  Signed.
    interval : float
        Time between the two acquisitions traversed by the interferogram, in
        years.  Must be non-negative; zero gives an empty interferogram.
    center_x, center_y : float, optional
        A point on the fault trace, in metres.  Default ``(0, 0)``.

    Returns
    -------
    u_e, u_n, u_u : ndarray
        Displacement in metres, shape ``(grid.ny, grid.nx)``.  ``u_u`` is
        identically zero.

    Raises
    ------
    ValueError
        If ``interval`` is negative, or if any argument is rejected by
        :func:`savage_velocity`.

    Examples
    --------
    >>> from parvaneh.synthetic.grid import Grid
    >>> grid = Grid.centered(nx=5, ny=5, spacing=1000.0)
    >>> u_e, u_n, u_u = savage_displacement(grid, 0.0, 10000.0, 0.03,
    ...                                     interval=2.0)
    >>> v_e, v_n, _ = savage_velocity(grid, 0.0, 10000.0, 0.03)
    >>> bool(np.allclose(u_n, 2.0 * v_n))
    True

    A zero-length interval gives no displacement, however fast the fault is:

    >>> z_e, z_n, _ = savage_displacement(grid, 0.0, 10000.0, 5.0,
    ...                                   interval=0.0)
    >>> bool(np.allclose(z_n, 0.0))
    True
    """
    interval_value = float(interval)
    if not np.isfinite(interval_value) or interval_value < 0.0:
        raise ValueError(
            "interval must be a non-negative finite number of years, got "
            "{!r}".format(interval))
    v_e, v_n, v_u = savage_velocity(grid, strike_deg, locking_depth, slip_rate,
                                   center_x=center_x, center_y=center_y)
    return (v_e * interval_value, v_n * interval_value,
            v_u * interval_value)


def random_savage_source(grid, style="deep_locking", rng=None):
    """Sample a random interseismic fault whose pattern fits on ``grid``.

    Parameters are drawn from the ranges in :data:`SAVAGE_STYLES`, so a batch
    of datasets explores a real spread of locking depths and slip rates
    instead of repeating one hand-picked fault.

    Parameters
    ----------
    grid : Grid
        Grid the fault will be evaluated on.  ``centre_range`` in
        :data:`SAVAGE_STYLES` is the fraction of the grid half-width over
        which the trace is allowed to wander, measured from the grid centre;
        ``0`` pins the trace to the centre.
    style : str, optional
        Key of :data:`SAVAGE_STYLES`.  Default ``"deep_locking"``.
    rng : numpy.random.Generator, optional
        Random source.  A default generator is created if omitted, in which
        case the result differs between calls.

    Returns
    -------
    dict
        Keyword arguments ready for :func:`savage_velocity` or
        :func:`savage_displacement`, plus a ``style`` entry recording how it
        was generated.

    Raises
    ------
    ValueError
        If ``style`` is not a known key of :data:`SAVAGE_STYLES`.

    Examples
    --------
    >>> from parvaneh.synthetic.grid import Grid
    >>> grid = Grid.centered(nx=64, ny=64, spacing=100.0)
    >>> src = random_savage_source(grid, "north_south",
    ...                            rng=np.random.default_rng(20240101))
    >>> (src["strike_deg"], src["style"])
    (0.0, 'north_south')
    >>> sorted(src)
    ['center_x', 'center_y', 'locking_depth', 'slip_rate', 'strike_deg', 'style']

    The dictionary is ready to be unpacked straight into
    :func:`savage_velocity` once the bookkeeping ``style`` entry is dropped:

    >>> parameters = dict(src)
    >>> del parameters["style"]
    >>> v_e, v_n, v_u = savage_velocity(grid, **parameters)
    >>> bool(np.all(np.isfinite(v_e)))
    True
    """
    _check_grid(grid)
    if style not in SAVAGE_STYLES:
        raise ValueError(
            "unknown Savage style {!r}; expected one of {}".format(
                style, ", ".join(sorted(SAVAGE_STYLES))))
    if rng is None:
        rng = np.random.default_rng()

    spec = SAVAGE_STYLES[style]
    locking_depth = float(rng.uniform(*spec["locking_depth_range"]))
    slip_rate = float(rng.uniform(*spec["slip_rate_range"]))
    strike = float(rng.uniform(*spec["strike_range"]))

    reach = float(spec["centre_range"])
    half_width = 0.5 * grid.width
    half_height = 0.5 * grid.height
    if reach <= 0.0:
        center_x = grid.center[0]
        center_y = grid.center[1]
    else:
        center_x = grid.center[0] + float(rng.uniform(-1.0, 1.0)) * reach * half_width
        center_y = grid.center[1] + float(rng.uniform(-1.0, 1.0)) * reach * half_height

    return {
        "strike_deg": strike,
        "locking_depth": locking_depth,
        "slip_rate": slip_rate,
        "center_x": center_x,
        "center_y": center_y,
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
