"""Interferometric phase noise: the physically correct simulation.

The one-line summary
--------------------

Real InSAR phase noise is **not** Gaussian and its amplitude is **not** a free
parameter.  Given the coherence :math:`\\gamma` and the number of independent
looks :math:`L` used to form the interferogram, the distribution of the phase
error is completely determined.  This module simulates that distribution
exactly, by simulating the interferogram itself.

A benchmark that adds hand-tuned Gaussian phase noise measures an unwrapper
against a noise model that no real interferogram ever follows.  A benchmark
that generates the phase error from :math:`(\\gamma, L)` measures it against
reality.  That is why this module exists.

The physical model
------------------

Each SAR pixel is a sum over a great many unresolved scatterers inside a
resolution cell.  By the central limit theorem the resulting complex
reflectivity is a *circular complex Gaussian* random variable.  Write the
master and slave reflectivities as :math:`z_1` and :math:`z_2`, each normalised
so that :math:`\\langle |z_k|^2 \\rangle = 1`.  Their complex correlation
coefficient is

.. math::

    \\gamma = \\langle z_1 z_2^{*} \\rangle \\in \\mathbb{C},
    \\qquad |\\gamma| \\le 1,

and the *coherence* is its magnitude :math:`|\\gamma|`.  Interferometric phase
is the argument of the interferogram

.. math::

    S = \\sum_{\\ell = 1}^{L} z_{1,\\ell} z_{2,\\ell}^{*},
    \\qquad \\psi = \\arg S .

If the reflectivities were noiseless the interferogram would be
:math:`z_1 z_2^{*} = \\exp(i \\phi_{\\text{true}})`, so :math:`\\psi` *is* the
phase error: the observed phase is :math:`\\phi_{\\text{true}} + \\psi`.
Writing the phase of the coherence as zero (the usual convention, which just
shifts all phases by a constant) it is enough to simulate

.. math::

    a, b \\sim \\mathcal{CN}(0, 1) \\text{ independent}, \\quad
    z_1 = a, \\quad
    z_2 = \\gamma a + \\sqrt{1 - \\gamma^{2}}\\, b,

which satisfies :math:`\\langle |z_2|^2 \\rangle = 1` and
:math:`\\langle z_1 z_2^{*} \\rangle = \\gamma` by construction.  Repeating this
:math:`L` times, averaging and taking the argument gives one sample of the
phase error for that :math:`(\\gamma, L)` pair.  That is exactly what
:func:`phase_noise` does, and it is a simulation of the physics rather than an
approximation of its statistics.

The exact single-look distribution
----------------------------------

For :math:`L = 1` the phase error has a closed-form probability density
(Goodman 1963; Lee, Hoppel, Mango & Miller 1994; Tough, Blacknell & Quegan
1995).  For :math:`\\psi \\in (-\\pi, \\pi]`,

.. math::

    p(\\psi) = \\frac{1 - \\gamma^{2}}{2\\pi (1 - \\beta^{2})}
        \\left[ 1 + \\frac{\\beta\\,(\\pi - \\arccos \\beta)}
                          {\\sqrt{1 - \\beta^{2}}} \\right],
    \\qquad \\beta = \\gamma \\cos \\psi .

:func:`single_look_phase_pdf` evaluates it.  Two sanity checks are built into
the formula and are verified in the test suite: at :math:`\\gamma = 0`,
:math:`\\beta = 0` and the density is the uniform :math:`1/2\\pi`, as it must be
when there is no correlation at all; and the density integrates to one.  The
most useful thing about having it is that the simulated histogram can be
compared against a formula derived independently of the simulation, which is
what turns "we added some noise" into a validated noise model.

The approximate phase scatter
-----------------------------

The many-look limit has the well-known approximation (Rodriguez & Martin 1992;
Bamler & Hartl 1998)

.. math::

    \\sigma_{\\psi} \\approx \\sqrt{\\frac{1 - \\gamma^{2}}{2 L \\gamma^{2}}}
    \\quad \\text{radians}.

:func:`phase_noise_std` returns it, clipped at the standard deviation of a
uniform distribution on :math:`(-\\pi, \\pi]`, namely
:math:`\\pi/\\sqrt{3} \\approx 1.8138` rad, which is the largest value any
wrapped phase can have.

Two cautions.  The formula diverges as :math:`\\gamma \\to 0`, which is
unphysical -- a wrapped phase cannot scatter more than a uniform phase does --
and that is what the clip repairs.  Less widely advertised is that it is a
*large-:math:`L`* result: it comes from a central-limit argument that needs many
independent looks before the phase error is anywhere near Gaussian.  It is
convenient for :math:`L \\gtrsim 16` and misleading for :math:`L = 1`.  The exact
single-look standard deviations, obtained by integrating
:func:`single_look_phase_pdf`, are

================  ==============  ===============
:math:`\\gamma`   exact           formula, clipped
================  ==============  ===============
0.00              1.8138          1.8138
0.20              1.6363          1.8138
0.50              1.3361          1.2247
0.80              0.9174          0.5303
0.95              0.5198          0.2324
================  ==============  ===============

At :math:`\\gamma = 0.95` the formula is too small by more than a factor of two.
Treat it as a rough guide, and as a check that converges as looks are
accumulated; where the two disagree, the simulation is right.

The message to take away
------------------------

Look at the numbers in :func:`phase_noise_std`.  A coherence of 0.2 with a
single look gives a phase standard deviation of about 1.64 rad, against
1.81 rad for a phase that carries no information at all.  Coherence 0.2 is not
"slightly degraded"; it is essentially no measurement.  Averaging over looks is
the only way to recover usable phase from a low-coherence scene, and the
recovery is slow -- only as :math:`1/\\sqrt{L}`.

Conventions
-----------

* Coherence is dimensionless and lies in ``[0, 1]``.
* All phases are in radians and live in ``[-\\pi, \\pi]`` once wrapped.
* Arrays have shape ``(grid.ny, grid.nx)`` in the package-wide ENU raster
  layout; any shape works for the array-valued arguments here.
* A ``nan`` coherence produces a ``nan`` phase error, so masked pixels stay
  masked rather than silently acquiring a random phase.
* The returned noisy phase from :func:`add_phase_noise` is **unwrapped**, i.e.
  continuous.  Wrap it with
  :func:`parvaneh.synthetic.wrapping.wrap_phase` to obtain the
  interferogram an unwrapper would actually be handed.  Keeping the two apart
  is deliberate: the whole benchmark rests on being able to compare the wrapped
  interferogram against the continuous truth it came from.

References
----------

Bamler, R. & Hartl, P. (1998).  Synthetic aperture radar interferometry.
*Inverse Problems* 14, R1-R54.  doi:10.1088/0266-5611/14/4/001

Goodman, J. W. (1963).  Statistical analysis based on a certain multivariate
complex Gaussian distribution.  *Annals of Mathematical Statistics* 34,
152-177.  doi:10.1214/aoms/1177704250

Lee, J.-S., Hoppel, K. W., Mango, S. A. & Miller, A. R. (1994).  Intensity and
phase statistics of multilook polarimetric and interferometric SAR imagery.
*IEEE Transactions on Geoscience and Remote Sensing* 32, 1017-1028.
doi:10.1109/36.312890

Rodriguez, E. & Martin, J. M. (1992).  Theory and design of interferometric
synthetic aperture radars.  *IEE Proceedings F* 139, 147-159.
doi:10.1049/ip-f-2.1992.0018

Tough, R. J. A., Blacknell, D. & Quegan, S. (1995).  A statistical description
of the phenomenon of interferometric phase.  *Journal of Electromagnetic Waves
and Applications* 9, 1021-1040.  doi:10.1163/156939395X00596
"""

import numpy as np

from ._checks import positive_int as _positive_int
from ._checks import positive_float as _positive_float

__all__ = [
    "DEFAULT_LOOKS",
    "UNIFORM_PHASE_STD",
    "phase_noise",
    "add_phase_noise",
    "phase_noise_std",
    "single_look_phase_pdf",
]


#: The number of looks used when the caller does not say.  One look is the
#: honest default for a full-resolution interferogram, and it is the hardest
#: case, so a benchmark that passes by default has earned it.
DEFAULT_LOOKS = 1


#: Standard deviation, in radians, of a phase that is uniform on
#: ``(-pi, pi]``.  This is the largest standard deviation any wrapped phase can
#: have, and therefore the ceiling used by :func:`phase_noise_std`.
UNIFORM_PHASE_STD = float(np.pi / np.sqrt(3.0))


def _check_coherence_array(coherence):
    """Validate a coherence value or array and return it as ``float64``."""
    array = np.asarray(coherence, dtype=np.float64)
    if array.ndim > 2:
        raise ValueError(
            "coherence must be a scalar or at most a 2-D array, got shape "
            "{}".format(array.shape))
    finite = np.isfinite(array)
    if finite.any():
        low = float(array[finite].min())
        high = float(array[finite].max())
        if low < 0.0 or high > 1.0:
            raise ValueError(
                "coherence must lie in [0, 1], got range [{!r}, {!r}]".format(
                    low, high))
    return array


def phase_noise(coherence, looks=DEFAULT_LOOKS, rng=None):
    """Draw one realisation of the interferometric phase error.

    The phase error is obtained by simulating the complex multi-look
    interferogram described in the module documentation and taking its
    argument.  Every pixel is drawn independently.

    Parameters
    ----------
    coherence : float or array_like
        Coherence, or a map of coherence, in ``[0, 1]``.  A ``nan`` entry
        yields a ``nan`` phase error.
    looks : int, optional
        Number of independent looks averaged into the interferogram.  A
        positive integer.  Default ``1``, a full-resolution interferogram.
    rng : numpy.random.Generator, optional
        Random source.  A default generator is created if omitted, in which
        case the result differs between calls.  Pass
        ``np.random.default_rng(seed)`` for a reproducible realisation.

    Returns
    -------
    ndarray
        Phase error in radians, of the same shape as ``coherence``, with values
        in ``[-pi, pi]``.  Add it to a clean phase to obtain the observed phase.

    Raises
    ------
    ValueError
        If ``coherence`` is not scalar or 2-D, if it contains values outside
        ``[0, 1]``, or if ``looks`` is not a positive integer.

    Notes
    -----
    The distribution is the *exact* consequence of the physics, so no
    tuning is available -- and none is needed.  Three limits follow immediately
    and are checked in the test suite:

    * :math:`\\gamma = 1` gives exactly ``0`` for every pixel.  With perfect
      correlation the two reflectivities are equal, so the interferogram is
      real and positive.
    * :math:`\\gamma = 0` gives a phase that is uniform on ``(-pi, pi]``.
    * Large ``looks`` makes the phase error shrink as :math:`1/\\sqrt{L}`.

    Examples
    --------
    >>> rng = np.random.default_rng(12345)
    >>> float(phase_noise(1.0, rng=rng))
    0.0
    >>> psi = phase_noise(np.full((256, 256), 0.95), rng=rng)
    >>> bool(np.all(np.abs(psi) <= np.pi))
    True
    >>> round(float(psi.std()), 1)             # narrow, as high coherence demands
    0.5
    >>> psi_low = phase_noise(np.full((256, 256), 0.2), rng=rng)
    >>> bool(float(psi_low.std()) > 1.0)       # a low-coherence scene is noise
    True
    """
    gamma = _check_coherence_array(coherence)
    looks_value = _positive_int(looks, "looks")
    if rng is None:
        rng = np.random.default_rng()

    shape = gamma.shape
    # Accumulate the multi-look interferogram one look at a time so that only
    # two grid-sized complex temporaries exist at once, whatever `looks` is.
    accumulator = None
    for _ in range(looks_value):
        master = (rng.normal(size=shape) + 1j * rng.normal(size=shape))
        slave = (rng.normal(size=shape) + 1j * rng.normal(size=shape))
        master = master / np.sqrt(2.0)
        # Correlate the slave with the master: this is the physical model.
        correlated = gamma * master + np.sqrt(
            np.clip(1.0 - gamma ** 2, 0.0, 1.0)) * slave / np.sqrt(2.0)
        product = master * np.conj(correlated)
        accumulator = product if accumulator is None else accumulator + product

    psi = np.angle(accumulator)
    # A nan coherence must not turn into a random phase.
    invalid = ~np.isfinite(gamma)
    if np.any(invalid):
        psi = np.where(invalid, np.nan, psi)
    return np.asarray(psi, dtype=np.float64)


def add_phase_noise(phase, coherence, looks=DEFAULT_LOOKS, rng=None):
    """Add a physically correct phase error to a clean phase field.

    Parameters
    ----------
    phase : array_like
        Clean, **continuous** phase in radians.  This is normally the sum of
        the deformation phase and whatever other components the scenario
        includes.
    coherence : float or array_like
        Coherence, or coherence map, in ``[0, 1]``.  Broadcasting follows NumPy
        rules, so a scalar can be combined with a full phase field.
    looks : int, optional
        Number of looks.  Default ``1``.
    rng : numpy.random.Generator, optional
        Random source.

    Returns
    -------
    ndarray
        The noisy phase, still continuous.  Its shape is the broadcast shape of
        ``phase`` and ``coherence``.  It is **not** wrapped; pass it to
        :func:`parvaneh.synthetic.wrapping.wrap_phase` for the
        interferogram an unwrapper sees.

    Examples
    --------
    >>> add_phase_noise(np.zeros((3, 3)), 1.0, rng=np.random.default_rng(7))
    array([[0., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.]])

    The error that is added is exactly the drawing :func:`phase_noise` makes
    from the same random source, which is worth knowing when a test needs to
    separate the two:

    >>> truth = np.arange(9, dtype=float).reshape(3, 3)
    >>> noisy = add_phase_noise(truth, 0.9, rng=np.random.default_rng(7))
    >>> psi = phase_noise(np.full((3, 3), 0.9), rng=np.random.default_rng(7))
    >>> bool(np.array_equal(noisy, truth + psi))
    True
    """
    clean = np.asarray(phase, dtype=np.float64)
    gamma = _check_coherence_array(coherence)
    shape = np.broadcast(clean, gamma).shape
    psi = phase_noise(np.broadcast_to(gamma, shape), looks=looks, rng=rng)
    return np.asarray(clean, dtype=np.float64) + psi


def phase_noise_std(coherence, looks=DEFAULT_LOOKS):
    """Approximate standard deviation of the interferometric phase error.

    Returns

    .. math::

        \\sqrt{\\frac{1 - \\gamma^{2}}{2 L \\gamma^{2}}}

    clipped at :data:`UNIFORM_PHASE_STD`, the standard deviation of a phase
    that is uniform on ``(-pi, pi]``.

    Parameters
    ----------
    coherence : float or array_like
        Coherence in ``[0, 1]``.
    looks : int, optional
        Number of looks, a positive integer.  Default ``1``.

    Returns
    -------
    ndarray or float
        Standard deviation in radians.  A scalar input gives a scalar.

    Notes
    -----
    This is the standard many-look approximation of Rodriguez & Martin (1992)
    and Bamler & Hartl (1998).  It is provided because it makes the
    :math:`(\\gamma, L)` dependence explicit and because it gives the test suite
    an independent prediction to check the simulation against.  It is *not* used
    anywhere in the simulation itself, and for small ``looks`` it is not
    accurate: it follows from a central-limit argument that needs enough
    independent looks for the phase error to be nearly Gaussian.  Measured
    against the simulation, it is within a few per cent once :math:`L \\gtrsim
    16`, and at ``looks=1`` it underestimates the true scatter by more than a
    factor of two near :math:`\\gamma = 0.95`.  The module documentation lists
    the exact single-look values.  When this function and :func:`phase_noise`
    disagree, believe :func:`phase_noise`.

    The clip at :math:`\\pi/\\sqrt{3}` is not cosmetic.  The raw formula
    diverges as :math:`\\gamma \\to 0`, which is unphysical: a wrapped phase
    cannot scatter by more than a uniform phase on the circle does.  Clipping
    also makes :math:`\\gamma = 0` behave sensibly instead of returning
    infinity.

    Examples
    --------
    >>> round(float(phase_noise_std(0.95)), 4)
    0.2324
    >>> round(float(phase_noise_std(0.8)), 4)
    0.5303
    >>> round(float(phase_noise_std(0.0)), 4)         # clipped at pi/sqrt(3)
    1.8138
    >>> round(float(phase_noise_std(0.5, looks=4)), 4) # four looks halve it
    0.6124
    """
    gamma = _check_coherence_array(coherence)
    looks_value = _positive_int(looks, "looks")
    safe = np.where(gamma > 1e-12, gamma, 1e-12)
    variance = np.clip(1.0 - gamma ** 2, 0.0, None) / (
        2.0 * looks_value * safe ** 2)
    result = np.sqrt(variance)
    result = np.minimum(result, UNIFORM_PHASE_STD)
    invalid = ~np.isfinite(gamma)
    if np.any(invalid):
        result = np.where(invalid, np.nan, result)
    if result.ndim == 0:
        return float(result)
    return np.asarray(result, dtype=np.float64)


def single_look_phase_pdf(psi, coherence):
    """Exact probability density of the single-look (``L = 1``) phase error.

    Evaluates the closed-form density given in the module documentation, which
    describes the phase error of an interferogram formed from one pair of
    correlated circular complex Gaussian reflectivities.

    Parameters
    ----------
    psi : float or array_like
        Phase error in radians, in ``(-pi, pi]``.
    coherence : float
        Coherence in ``[0, 1)``.  ``1`` is rejected: the density degenerates to
        a Dirac delta at zero and cannot be represented as an array.

    Returns
    -------
    ndarray or float
        Probability density per radian, the same shape as ``psi``.

    Raises
    ------
    ValueError
        If ``coherence`` is outside ``[0, 1)`` or is not finite.

    Notes
    -----
    Use this to check a simulated histogram against theory.  It is the single
    most valuable validation in the module, because the density was derived by
    a completely different route from the simulation: the simulation draws
    Gaussian variates, while the density comes from integrating the
    four-dimensional complex Gaussian exactly.

    Examples
    --------
    >>> round(float(single_look_phase_pdf(0.0, 0.0)), 6)   # uniform
    0.159155
    >>> round(float(single_look_phase_pdf(0.0, 0.9)), 4)   # sharp peak
    1.0433
    >>> round(float(single_look_phase_pdf(np.pi, 0.9)), 4) # long tail
    0.0109
    >>> single_look_phase_pdf(0.0, 1.0)
    Traceback (most recent call last):
        ...
    ValueError: coherence must lie in [0, 1) for the single-look density, got 1.0
    """
    gamma = float(coherence)
    if not np.isfinite(gamma) or gamma < 0.0 or gamma >= 1.0:
        raise ValueError(
            "coherence must lie in [0, 1) for the single-look density, got "
            "{!r}".format(coherence))
    values = np.asarray(psi, dtype=np.float64)
    beta = gamma * np.cos(values)
    one_minus_beta_squared = np.clip(1.0 - beta ** 2, 1e-300, None)
    bracket = 1.0 + (
        beta * (np.pi - np.arccos(np.clip(beta, -1.0, 1.0)))
        / one_minus_beta_squared ** 0.5)
    density = (1.0 - gamma ** 2) / (2.0 * np.pi * one_minus_beta_squared)
    density = density * bracket
    if values.ndim == 0:
        return float(density)
    return np.asarray(density, dtype=np.float64)
