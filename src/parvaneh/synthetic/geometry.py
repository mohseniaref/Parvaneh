"""SAR line-of-sight geometry and displacement-to-phase conversion.

This module is deliberately separate from the deformation models (:mod:`mogi`,
:mod:`okada`, :mod:`savage`).  A deformation model answers the question
"how did the ground move in three dimensions?".  A SAR system answers a
different question: "what component of that motion did the radar actually
see, and what phase does it produce?".  Mixing the two is the classic way to
end up with an unexplainable factor of two or a mysterious minus sign, so the
package keeps them apart and lets you compose them explicitly.

The line-of-sight vector
------------------------
A satellite in a near-polar orbit does not measure East, North and Up motion.
It measures the change in distance between the ground and the satellite along
the **line of sight** (LOS).  We describe that direction by a unit vector

.. math::

    \\hat{l} = (l_E, l_N, l_U)

which points **from the ground towards the satellite**.  Writing the surface
displacement as :math:`\\mathbf{u} = (u_E, u_N, u_U)`, the LOS displacement is
the projection

.. math::

    d_{LOS} = \\hat{l} \\cdot \\mathbf{u} = l_E u_E + l_N u_N + l_U u_U

Sign convention (read this twice)
---------------------------------
With :math:`\\hat{l}` pointing ground-to-satellite:

* :math:`d_{LOS} > 0` means the ground moved **towards the satellite**, which
  *shortens* the range.
* :math:`d_{LOS} < 0` means the ground moved **away from the satellite**,
  which *lengthens* the range.

Interferometric phase grows with range **decrease**, so

.. math::

    \\Phi_{def} = +\\frac{4\\pi}{\\lambda}\\, d_{LOS}

which can also be written :math:`\\Phi_{def} = -\\frac{4\\pi}{\\lambda}\\Delta r`
with :math:`\\Delta r = -d_{LOS}` the range change.  Both are the same
statement.

.. warning::
   Exactly **one** knob encodes the sign convention.  If a downstream package
   reports phase with the opposite sign, pass ``sign=-1.0`` to
   :func:`displacement_to_phase`.  Do **not** additionally pass
   ``los_sign=-1.0`` to :func:`project_to_los`: the two flips cancel and you
   get the original phase back, an error that is very hard to see in a plot.
   Edit the sign, never the physics.

The factor 4 arises because the radar signal travels from the satellite to the
ground and back, so a range change :math:`\\Delta r` changes the path length by
:math:`2\\Delta r`, and phase is :math:`2\\pi/\\lambda` per unit path length.

Angles and sensors
------------------
Every angle parameter is in **degrees** and carries a ``_deg`` suffix;
everything else is SI (metres, radians).  The incidence angle is measured from
the local vertical (zenith) at the ground point, and azimuths are measured
clockwise from North, so due East is :math:`90^\\circ`.

Worked example
--------------
>>> from parvaneh.synthetic.geometry import (los_unit_vector,
...                                                    project_to_los,
...                                                    displacement_to_phase)
>>> l_e, l_n, l_u = los_unit_vector(incidence_deg=30.0, look_azimuth_deg=270.0)
>>> round(l_e, 6), round(abs(l_n), 6), round(l_u, 6)
(-0.5, 0.0, 0.866025)

The satellite is due west of the ground point and 30 degrees above the
horizon, so the unit vector has a westward horizontal component and a large
upward component.  The north component is zero exactly; ``abs`` is applied
above only because IEEE arithmetic reports it as ``-0.0``.  Now suppose the
ground rises by 1 cm:

>>> d_los = project_to_los(u_e=0.0, u_n=0.0, u_u=0.01,
...                        incidence_deg=30.0, look_azimuth_deg=270.0)
>>> round(d_los, 6)
0.00866

Uplift moves the ground towards the satellite, so :math:`d_{LOS} > 0`, and the
corresponding phase for Sentinel-1 (C-band) is

>>> round(displacement_to_phase(d_los, wavelength=0.0554658), 4)
1.9621

i.e. about 1.96 radians, or roughly a third of a fringe, per centimetre of
uplift here.  That ratio is worth remembering: one full 2 pi fringe of
Sentinel-1 phase corresponds to ``lambda / 2 = 2.77 cm`` of LOS motion,
because the phase contains the factor 4 pi rather than 2 pi.

References
----------
* Hanssen, R. F. (2001). *Radar Interferometry: Data Interpretation and Error
  Analysis*. Kluwer. (Chapter 2 covers the imaging geometry and the
  displacement-to-phase relation.)
* ESA Sentinel-1 fact sheet for the 5.405 GHz carrier frequency.
* Rosen, P. A. et al. (2000). Synthetic aperture radar interferometry,
  *Proceedings of the IEEE*, 88(3), 333-382.
"""

import numpy as np

from ._checks import finite_float as _finite_float
from ._checks import positive_float as _positive_float
from ._checks import sign as _sign

SPEED_OF_LIGHT = 299792458.0

#: Carrier wavelength in metres for common SAR missions, keyed by a short
#: lower-case name.  Values are computed as ``c / f`` from the published
#: centre frequency, not copied from rounded tables, so they are exactly
#: consistent with :data:`SPEED_OF_LIGHT`.
WAVELENGTHS = {
    # C-band, 5.405 GHz.  Sentinel-1A/1B/1C IW and EW modes.
    "sentinel1": SPEED_OF_LIGHT / 5.405e9,
    # C-band, 5.3 GHz.  ERS-1/2 and Envisat ASAR.
    "ers": SPEED_OF_LIGHT / 5.3e9,
    "envisat": SPEED_OF_LIGHT / 5.3e9,
    # L-band, 1.27 GHz.  ALOS PALSAR.
    "alos": SPEED_OF_LIGHT / 1.27e9,
    "palsar": SPEED_OF_LIGHT / 1.27e9,
    # L-band, 1.2575 GHz.  NISAR.
    "nisar": SPEED_OF_LIGHT / 1.2575e9,
}


class SarGeometry:
    """Rigid SAR acquisition geometry: wavelength, incidence and look azimuth.

    This is a small value object so that geometry can be carried around with a
    dataset and written to metadata in one piece.  Angles are stored in
    degrees (matching the public API) and converted to radians on use.

    Parameters
    ----------
    wavelength : float
        Radar carrier wavelength in metres.  Use :func:`wavelength_for` to
        look one up by sensor name.
    incidence_deg : float
        Incidence angle from the local vertical at the ground.  Sentinel-1
        IW varies from about 30 degrees (near range) to 46 degrees (far
        range), with roughly 39 degrees at mid-swath.
    heading_deg : float or None
        Azimuth of the satellite's horizontal velocity, clockwise from North.
        Exactly one of ``heading_deg`` and ``look_azimuth_deg`` must be given.
    look_azimuth_deg : float or None
        Azimuth from the ground point towards the satellite, clockwise from
        North.  Convenient when a processing product already supplies it.
    right_looking : bool
        Almost every civil SAR is right-looking, meaning the antenna points to
        the right of the flight direction.  This is what links ``heading_deg``
        to ``look_azimuth_deg``.
    sign : float
        ``+1.0`` (default) for phase that increases when the ground moves
        towards the satellite; ``-1.0`` flips the sign of the reported phase.
    """

    def __init__(self, wavelength, incidence_deg, heading_deg=None,
                 look_azimuth_deg=None, right_looking=True, sign=1.0):
        self.wavelength = _positive_float(wavelength, "wavelength")
        self.incidence_deg = _finite_float(incidence_deg, "incidence_deg")
        if not 0.0 <= self.incidence_deg <= 90.0:
            raise ValueError("incidence_deg must lie in [0, 90], got {!r}"
                             .format(incidence_deg))
        if (heading_deg is None) == (look_azimuth_deg is None):
            raise ValueError("give exactly one of heading_deg or "
                             "look_azimuth_deg")
        self.right_looking = bool(right_looking)
        self.sign = _sign(sign)
        if heading_deg is not None:
            self.heading_deg = _finite_float(heading_deg, "heading_deg")
            self.look_azimuth_deg = look_azimuth_from_heading(
                self.heading_deg, right_looking=self.right_looking)
        else:
            self.look_azimuth_deg = _finite_float(look_azimuth_deg,
                                                  "look_azimuth_deg")
            self.heading_deg = heading_from_look_azimuth(
                self.look_azimuth_deg, right_looking=self.right_looking)

    @property
    def los_vector(self):
        """``(l_E, l_N, l_U)`` unit vector from the ground towards the satellite.

        This is pure geometry and is therefore **not** affected by
        :attr:`sign`, which only flips the sign of the reported phase.
        """
        return los_unit_vector(self.incidence_deg, self.look_azimuth_deg)

    @property
    def pass_direction(self):
        """``"ascending"`` or ``"descending"``, inferred from the heading.

        An ascending pass travels broadly northwards, a descending pass
        broadly southwards.  Note that the *look* direction is perpendicular
        to the heading, so an ascending pass looks to the west and a
        descending pass looks to the east.
        """
        northward = np.cos(np.radians(self.heading_deg))
        return "ascending" if northward >= 0.0 else "descending"

    def project(self, u_e, u_n, u_u):
        """Project an ENU displacement field onto this geometry's line of sight.

        The result is always in the standard convention, positive towards the
        satellite.  :attr:`sign` is applied by :meth:`phase`, not here, so
        that ``phase(project(u))`` flips exactly once.
        """
        return project_to_los(u_e, u_n, u_u, self.incidence_deg,
                              look_azimuth_deg=self.look_azimuth_deg)

    def phase(self, d_los):
        """Convert a standard-convention LOS displacement to phase in radians."""
        return displacement_to_phase(d_los, self.wavelength, self.sign)

    def phase_from_displacement(self, u_e, u_n, u_u):
        """Convenience: project an ENU displacement and convert it to phase.

        Equivalent to ``self.phase(self.project(u_e, u_n, u_u))``.
        """
        return self.phase(self.project(u_e, u_n, u_u))

    def to_dict(self):
        """Return a plain, JSON/YAML-friendly dict of the geometry."""
        return {
            "wavelength_m": self.wavelength,
            "incidence_angle_deg": self.incidence_deg,
            "heading_deg": self.heading_deg,
            "look_azimuth_deg": self.look_azimuth_deg,
            "right_looking": self.right_looking,
            "pass_direction": self.pass_direction,
            "los_unit_vector_enu": list(self.los_vector),
            "sign": self.sign,
        }

    def __repr__(self):
        return ("SarGeometry(wavelength={:.7g}, incidence_deg={:.4g}, "
                "heading_deg={:.4g}, pass={!r}, sign={!r})".format(
                    self.wavelength, self.incidence_deg, self.heading_deg,
                    self.pass_direction, self.sign))


def wavelength_for(sensor):
    """Return the carrier wavelength in metres for a named sensor.

    Parameters
    ----------
    sensor : str
        One of the keys of :data:`WAVELENGTHS`, case insensitive:
        ``sentinel1``, ``ers``, ``envisat``, ``alos``, ``palsar``, ``nisar``.

    Returns
    -------
    float
        Wavelength in metres.

    Examples
    --------
    >>> round(wavelength_for("sentinel1"), 7)
    0.0554658
    """
    key = str(sensor).strip().lower().replace("-", "").replace("_", "")
    for name, value in WAVELENGTHS.items():
        if name.replace("_", "") == key:
            return float(value)
    raise ValueError("unknown sensor {!r}; known sensors are {}"
                     .format(sensor, ", ".join(sorted(set(WAVELENGTHS)))))


def los_unit_vector(incidence_deg, look_azimuth_deg, los_sign=1.0):
    """Build the ENU unit vector pointing from the ground to the satellite.

    Parameters
    ----------
    incidence_deg : float or array_like
        Incidence angle from the local vertical, in degrees.
    look_azimuth_deg : float or array_like
        Azimuth from the ground point towards the satellite, clockwise from
        North, in degrees.
    los_sign : float
        ``+1.0`` for the standard ground-to-satellite convention.  ``-1.0``
        returns the opposite direction, which flips the sign of every LOS
        projection; provided so that callers can reproduce a differently
        oriented product without rewriting the physics.

        .. warning::
           Flipping the LOS vector *and* passing ``sign=-1.0`` to
           :func:`displacement_to_phase` cancels out and gives you back the
           original phase.  Choose one place to encode your sign convention
           and leave the other at its default.

    Returns
    -------
    (l_e, l_n, l_u) : tuple of numpy.ndarray
        The three ENU components.  Scalars come back as 0-d arrays; pass them
        through ``float()`` if you want a Python number.
    """
    inc = np.radians(np.asarray(incidence_deg, dtype=np.float64))
    azi = np.radians(np.asarray(look_azimuth_deg, dtype=np.float64))
    horizontal = np.sin(inc)
    sign = _sign(los_sign)
    return (sign * horizontal * np.sin(azi),
            sign * horizontal * np.cos(azi),
            sign * np.cos(inc))


def look_azimuth_from_heading(heading_deg, right_looking=True):
    """Convert a flight heading into the ground-to-satellite look azimuth.

    A right-looking radar images a swath to the right of the flight
    direction, so the satellite-to-ground direction has azimuth
    ``heading + 90`` degrees.  The ground-to-satellite direction, which is
    what :func:`los_unit_vector` wants, is that plus 180 degrees, giving
    ``heading + 270`` mod 360.

    Examples
    --------
    A Sentinel-1 ascending pass travels about 12.6 degrees west of due north,
    so the satellite sits to the west-south-west of the ground point:

    >>> round(look_azimuth_from_heading(-12.6), 4)
    257.4

    A left-looking radar would put it on the other side:

    >>> round(look_azimuth_from_heading(-12.6, right_looking=False), 4)
    77.4
    """
    heading = _finite_float(heading_deg, "heading_deg")
    offset = 270.0 if right_looking else 90.0
    return (heading + offset) % 360.0


def heading_from_look_azimuth(look_azimuth_deg, right_looking=True):
    """Invert :func:`look_azimuth_from_heading`."""
    azimuth = _finite_float(look_azimuth_deg, "look_azimuth_deg")
    offset = 270.0 if right_looking else 90.0
    return (azimuth - offset) % 360.0


def ground_track_heading(inclination_deg=98.18, altitude_m=693000.0,
                         latitude_deg=0.0, earth_rotation=True):
    """Azimuth of the satellite's ground track, clockwise from North.

    The satellite's velocity in the inertial frame is combined with the
    eastward motion of the Earth's surface to give the direction in which the
    footprint actually sweeps across the ground.  This is the "heading" that
    enters the look-geometry calculation, and computing it from orbital
    parameters is preferable to quoting a number you cannot check.

    The orbital speed is taken from a circular orbit,
    ``v = sqrt(GM / (R + h))``, and the ground point's eastward speed is
    ``omega * R * cos(latitude)``.

    Parameters
    ----------
    inclination_deg : float
        Orbital inclination.  Sun-synchronous orbits such as Sentinel-1's are
        retrograde, with ``inclination_deg`` slightly greater than 90.
    altitude_m : float
        Mean orbital altitude above the Earth's surface, in metres.
    latitude_deg : float
        Geodetic latitude of the ground point, in degrees.  The ground-track
        azimuth of a real orbit varies along the pass because both the
        convergence of the meridians and the Earth's rotation term change.
    earth_rotation : bool
        Set to ``False`` to neglect the Earth's rotation, which makes the
        function directly testable against hand calculations.

    Returns
    -------
    float
        Heading in degrees, measured clockwise from North, in ``[0, 360)``.
        For an ascending pass of a Sun-synchronous orbit this is a small
        negative angle such as ``-12`` degrees, i.e. a track heading slightly
        west of due north.

    Notes
    -----
    The result is the **ascending** ground track when the satellite is moving
    northwards.  The descending track of the same orbit is
    ``heading + 180`` degrees modulo 360.

    Examples
    --------
    A prograde equatorial orbit should sweep due east, i.e. 90 degrees:

    >>> round(ground_track_heading(inclination_deg=0.0,
    ...                            earth_rotation=False), 6)
    90.0

    A retrograde polar orbit should sweep due north:

    >>> round(ground_track_heading(inclination_deg=90.0,
    ...                            earth_rotation=False), 6)
    0.0
    """
    gm = 3.986004418e14
    earth_radius = 6371000.0
    omega = 7.292115e-5
    alt = _positive_float(altitude_m, "altitude_m")
    inc = np.radians(_finite_float(inclination_deg, "inclination_deg"))
    lat = np.radians(_finite_float(latitude_deg, "latitude_deg"))
    if not -90.0 <= latitude_deg <= 90.0:
        raise ValueError("latitude_deg must lie in [-90, 90], got {!r}"
                         .format(latitude_deg))

    speed = np.sqrt(gm / (earth_radius + alt))
    east = speed * np.cos(inc)
    north = speed * np.sin(inc)
    if earth_rotation:
        east -= omega * earth_radius * np.cos(lat)
    return float(np.degrees(np.arctan2(east, north)) % 360.0)


def project_to_los(u_e, u_n, u_u, incidence_deg, heading_deg=None,
                   look_azimuth_deg=None, right_looking=True, los_sign=1.0):
    """Project an ENU displacement field onto the SAR line of sight.

    Parameters
    ----------
    u_e, u_n, u_u : array_like
        Displacement components in metres, East, North and Up.  They must be
        broadcast-compatible; in normal use all three have the grid shape
        ``(ny, nx)``.
    incidence_deg : float or array_like
        Incidence angle(s) in degrees.  A scalar applies everywhere.  An array
        of the grid shape lets you model the range-dependent incidence angle
        of a real swath.
    heading_deg, look_azimuth_deg : float or array_like or None
        Give exactly one.  See :class:`SarGeometry`.
    right_looking : bool
        Only used when ``heading_deg`` is given.
    los_sign : float
        ``+1.0`` for the standard ground-to-satellite convention.

    Returns
    -------
    numpy.ndarray
        LOS displacement in metres.  Positive means motion towards the
        satellite.

    Examples
    --------
    Pure uplift with a 30 degree incidence angle and the satellite due west
    gives ``d_LOS = h * cos(30 deg)``:

    >>> round(float(project_to_los(0.0, 0.0, 1.0, 30.0,
    ...                            look_azimuth_deg=270.0)), 6)
    0.866025

    Pure eastward motion with the satellite due west gives zero, because
    motion along the look azimuth's perpendicular is invisible:

    >>> round(float(project_to_los(1.0, 0.0, 0.0, 30.0,
    ...                            look_azimuth_deg=270.0)), 6)
    -0.5
    """
    if (heading_deg is None) == (look_azimuth_deg is None):
        raise ValueError("give exactly one of heading_deg or look_azimuth_deg")
    if look_azimuth_deg is None:
        look_azimuth_deg = look_azimuth_from_heading(heading_deg,
                                                     right_looking=right_looking)
    l_e, l_n, l_u = los_unit_vector(incidence_deg, look_azimuth_deg,
                                    los_sign=los_sign)
    return (l_e * np.asarray(u_e, dtype=np.float64)
            + l_n * np.asarray(u_n, dtype=np.float64)
            + l_u * np.asarray(u_u, dtype=np.float64))


def displacement_to_phase(d_los, wavelength, sign=1.0):
    """Convert a LOS displacement in metres to interferometric phase in radians.

    Implements ``Phi = sign * (4 pi / wavelength) * d_los``.  With the default
    convention and ``sign = +1`` the phase increases when the ground moves
    towards the satellite, which is the usual presentation of a processed
    interferogram.

    Parameters
    ----------
    d_los : array_like
        Line-of-sight displacement in metres.
    wavelength : float
        Radar carrier wavelength in metres.
    sign : float
        ``+1.0`` or ``-1.0``; see the module docstring.

    Returns
    -------
    numpy.ndarray
        Phase in radians.  This is **not** wrapped; it is the continuous
        physical phase, which can easily exceed 2 pi.

    Examples
    --------
    >>> round(float(displacement_to_phase(0.01, 0.0554658)), 4)
    2.2656
    """
    lam = _positive_float(wavelength, "wavelength")
    return _sign(sign) * (4.0 * np.pi / lam) * np.asarray(d_los, dtype=np.float64)


def phase_to_displacement(phase, wavelength, sign=1.0):
    """Invert :func:`displacement_to_phase`, returning metres.

    Examples
    --------
    >>> round(float(phase_to_displacement(2.2656, 0.0554658)), 4)
    0.01
    """
    lam = _positive_float(wavelength, "wavelength")
    return np.asarray(phase, dtype=np.float64) * lam / (4.0 * np.pi * _sign(sign))


def fringe_per_metre(wavelength):
    """Return the number of 2 pi phase fringes per metre of LOS motion.

    Since ``Phi = (4 pi / lambda) * d_LOS``, one whole fringe needs
    ``d_LOS = lambda / 2``, so the answer is simply ``2 / lambda``.

    This is a handy scale check: Sentinel-1 produces about 36 fringes per
    metre, i.e. 0.36 fringes per centimetre, so a whole-fringe unwrapping
    error is a sub-centimetre mistake.

    Examples
    --------
    >>> round(float(fringe_per_metre(0.0554658)), 3)
    36.058
    """
    return 2.0 / _positive_float(wavelength, "wavelength")
