"""Wrapping and the exact integer ambiguity.

The single most important idea in this module
---------------------------------------------
An interferometric phase measurement is only ever known modulo :math:`2\\pi`.
Write the *true* (unwrapped) phase as :math:`\\Phi` and the *measured*
(wrapped) phase as :math:`\\phi`.  Then the relationship between them is

.. math::

    \\Phi = \\phi + 2\\pi k, \\qquad k \\in \\mathbb{Z}

and ``k`` is the **integer ambiguity**.  Unwrapping is nothing more, and
nothing less, than recovering ``k`` for every pixel.  Every phase-unwrapping
algorithm in the literature, whatever mathematical machinery it dresses itself
in, is solving for ``k`` under some assumption about the smoothness of
:math:`\\Phi`.

This has an enormous practical consequence for building a synthetic benchmark.
If we generate :math:`\\Phi` ourselves we know ``k`` **exactly**, for free,
without running any algorithm.  So we can score an unwrapper on the quantity it
actually cares about:

* how often did it get ``k`` right?
* when it was wrong, was it off by one cycle, or by a hundred?

That is far more informative than a phase RMSE, which mixes together
"one-cycle errors near a steep fringe" and "tiny sub-radian noise" into a
single number.

How to wrap without getting it wrong
------------------------------------
The naive formula

.. code-block:: python

    wrapped = (phase + np.pi) % (2 * np.pi) - np.pi

is not wrong, but it is easy to get subtly wrong: pick ``-`` instead of ``+``,
forget a parenthesis, or use ``np.pi`` where you meant ``2 * np.pi``, and the
result still *looks* like a wrapped phase.  The formulation used here is the
one that cannot go wrong:

.. math::

    \\phi = \\arg\\left(e^{i \\Phi}\\right)

i.e. ``np.angle(np.exp(1j * phase))``.  A complex exponential *is* a phase on
the unit circle, and ``np.angle`` is defined as its argument, so the identity
:math:`\\arg(e^{i\\Phi}) = \\Phi \\bmod 2\\pi` holds by construction.  The
package therefore uses the complex form everywhere and keeps the modulo form
only in :mod:`accelerated_unwrap.synthetic.legacy`, where it is preserved for
backward compatibility.

Representation convention
-------------------------
Angles are returned in the *closed* interval :math:`[-\\pi, \\pi]`, and both
endpoints are genuine outputs of the complex form::

    >>> import numpy as np
    >>> from accelerated_unwrap.synthetic.wrapping import wrap_phase
    >>> wrap_phase(-np.pi)
    -3.141592653589793
    >>> float(np.abs(wrap_phase(3.0 * np.pi)))
    3.1415926535897927
    >>> bool(np.abs(wrap_phase(3.0 * np.pi)) <= np.pi)
    True

:math:`-\\pi` and :math:`+\\pi` name the *same* physical angle, exactly one
cycle apart, so a wrapping is not made wrong by choosing either one.  Which
one you actually get near the branch cut depends on floating-point rounding
inside ``np.exp``: ``exp(1j * 3 * pi)`` lands one unit in the last place
*below* ``-1 + 0j``, so its argument is a hair less than :math:`\\pi` rather
than exactly :math:`\\pi`.  This module deliberately does **not** remap
:math:`-\\pi` onto :math:`+\\pi`, because spec section 20 requires the
identity :math:`\\arg(e^{i\\Phi}) = \\Phi \\bmod 2\\pi` to hold exactly, and
any remapping step breaks it at the branch cut.

The practical consequence: if you compare phase arrays element-by-element
against another library, compare :math:`\\exp(i\\phi)` rather than
:math:`\\phi`, or compare wrapped differences.  A half-open convention such as
:math:`(-\\pi, \\pi]` cannot be guaranteed by the complex form and should not
be asserted by callers.  The package test suite uses the identity above rather
than a literal comparison for exactly this reason.

Numerical care at the branch cut
--------------------------------
``k`` is obtained from ``(Phi - phi) / (2 * pi)``, which is mathematically an
integer but in floating point is ``-1e-16`` away from one.  Rounding is
therefore required, and the tie-break rule matters if you ever construct a case
that lands exactly on a half-integer, which happens when ``Phi`` is exactly an
odd multiple of ``pi``.  :func:`integer_ambiguity` uses **round half away from
zero**, implemented explicitly rather than through :func:`numpy.round` whose
behaviour changed between NumPy versions (banker's rounding).  A deterministic
tie-break is a reproducibility requirement, not a detail.
"""

import numpy as np

__all__ = [
    "TWO_PI",
    "wrap_phase",
    "integer_ambiguity",
    "unwrap_with_ambiguity",
    "ambiguity_error",
    "is_wrapped",
    "wrap_count",
]

TWO_PI = 2.0 * np.pi


def wrap_phase(phase):
    """Wrap an unwrapped phase into the interval :math:`[-\\pi, \\pi]`.

    Parameters
    ----------
    phase : array_like
        Phase in radians.  Any shape.  ``NaN`` entries stay ``NaN``, which
        matters because masked pixels are represented as ``NaN`` in this
        package (see :mod:`accelerated_unwrap.synthetic.masks`).

    Returns
    -------
    ndarray
        ``float64`` array of the same shape as ``phase``, in
        :math:`[-\\pi, \\pi]``.

    Notes
    -----
    Implemented as ``np.angle(np.exp(1j * phase))``.  The exponential is
    evaluated with :func:`numpy.exp`, which for large arguments loses
    precision: ``exp(1j * 1e17)`` has an argument that is not exactly
    ``1e17 mod 2*pi`` because ``1e17`` cannot itself be represented exactly.
    This is not a defect of the formula -- the *input* is already ambiguous at
    that magnitude.  In practice phases of interest are smaller than
    :math:`10^6` radians (about 160 000 fringes), where the loss is harmless.

    Examples
    --------
    Values already inside the interval are unchanged:

    >>> bool(np.allclose(wrap_phase(0.5), 0.5))
    True

    A whole extra cycle disappears:

    >>> round(float(wrap_phase(0.5 + TWO_PI)), 6)
    0.5

    And ``pi`` maps onto ``pi`` rather than ``-pi``:

    >>> round(float(wrap_phase(np.pi)), 6)
    3.141593
    """
    array = np.asarray(phase, dtype=np.float64)
    return np.angle(np.exp(1j * array))


def integer_ambiguity(phase_true, phase_wrapped):
    """Return the integer ``k`` such that ``phase_true = phase_wrapped + 2 pi k``.

    This is the ground truth that makes a synthetic dataset worth having: no
    unwrapping algorithm is run, so the answer is exact by construction rather
    than approximate.

    Parameters
    ----------
    phase_true : array_like
        The unwrapped phase, e.g. ``Phi_deformation`` plus every nuisance
        term, exactly as generated by this package.
    phase_wrapped : array_like
        The corresponding wrapped phase, normally ``wrap_phase(phase_true)``.
        It is accepted as an argument rather than recomputed so that the
        caller can pass a *noisy* wrapped phase and still obtain the ambiguity
        of the noise-free pair -- a distinction that matters when you want to
        know whether an unwrapper failed because of noise or because of
        aliasing.

    Returns
    -------
    ndarray
        ``int64`` array of the same shape as the inputs, holding ``k``.

    Raises
    ------
    ValueError
        If the two arrays have different shapes.

    Notes
    -----
    Mathematically ``(phase_true - phase_wrapped) / (2 * pi)`` is an integer.
    In floating point it is that integer plus a rounding error of order
    :math:`10^{-15}`, so the result is rounded.  **Half-integer values are
    rounded away from zero**, which makes the output bit-for-bit reproducible
    across NumPy versions; :func:`numpy.round` would round halves to the
    nearest even integer instead.

    The rounding error stays negligible as long as ``phase_true`` is not
    astronomically large: for ``|Phi| < 1e9`` radians the absolute error is far
    below ``0.5``, so no correct integer is ever mis-rounded.

    Examples
    --------
    >>> phi = np.array([0.3, 6.0, -7.5])
    >>> k = integer_ambiguity(phi + TWO_PI * np.array([2, -1, 3]), phi)
    >>> k.tolist()
    [2, -1, 3]

    Reconstructing the unwrapped phase is then exact:

    >>> bool(np.allclose(phi + TWO_PI * k, phi + TWO_PI * np.array([2, -1, 3])))
    True
    """
    true = np.asarray(phase_true, dtype=np.float64)
    wrapped = np.asarray(phase_wrapped, dtype=np.float64)
    if true.shape != wrapped.shape:
        raise ValueError(
            "phase_true and phase_wrapped must have the same shape, got "
            "{} and {}".format(true.shape, wrapped.shape))

    ratio = (true - wrapped) / TWO_PI
    # np.floor(x + 0.5) is "round half up" for positive numbers; applying it to
    # the magnitude and restoring the sign gives "round half away from zero",
    # which is symmetric and therefore free of a directional bias.
    magnitude = np.floor(np.abs(ratio) + 0.5)
    rounded = np.copysign(magnitude, ratio)
    return rounded.astype(np.int64)


def unwrap_with_ambiguity(phase_wrapped, ambiguity):
    """Reconstruct an unwrapped phase from a wrapped phase and its ambiguity.

    This is the forward direction of :func:`integer_ambiguity` and the
    numerical validation that the two are consistent.

    Parameters
    ----------
    phase_wrapped : array_like
        Wrapped phase in radians.
    ambiguity : array_like
        Integer ambiguity ``k``, as returned by :func:`integer_ambiguity`.

    Returns
    -------
    ndarray
        ``float64`` array ``phase_wrapped + 2 * pi * ambiguity``.

    Raises
    ------
    ValueError
        If the shapes differ, or if ``ambiguity`` is not integral.  The
        integrality check catches the common mistake of feeding a *float*
        ambiguity -- for instance the output of an unwrapper that reports
        cycles as floating point -- straight into this function, where
        silently truncating would hide a real error.

    Examples
    --------
    >>> phi = np.array([np.pi - 0.1, -np.pi + 0.1])
    >>> k = np.array([1, -2], dtype=np.int64)
    >>> [round(float(v), 4) for v in unwrap_with_ambiguity(phi, k)]
    [9.3248, -15.608]
    """
    wrapped = np.asarray(phase_wrapped, dtype=np.float64)
    k = np.asarray(ambiguity)
    if wrapped.shape != k.shape:
        raise ValueError(
            "phase_wrapped and ambiguity must have the same shape, got "
            "{} and {}".format(wrapped.shape, k.shape))
    if not np.all(np.isclose(k, np.round(k))):
        raise ValueError(
            "ambiguity must be integral; got fractional values such as "
            "{!r}".format(float(np.asarray(k, dtype=np.float64).ravel()[0])
                          if k.size else None))
    return wrapped + TWO_PI * k.astype(np.int64)


def ambiguity_error(ambiguity_estimated, ambiguity_true, mask=None):
    """Difference between an estimated and a true integer ambiguity.

    ``k_error = k_estimated - k_true``.  ``k_error == 0`` means the pixel was
    unwrapped correctly; ``k_error == +1`` means the estimate is one cycle too
    high, which is the characteristic failure mode near a dense set of
    fringes.

    Parameters
    ----------
    ambiguity_estimated, ambiguity_true : array_like
        Integer ambiguity arrays of the same shape.
    mask : array_like of bool, optional
        If given, only pixels where ``mask`` is true are considered.  Pixels
        outside the mask are set to ``0`` in the returned array so that the
        output remains a full-size image that can be plotted directly.

    Returns
    -------
    ndarray
        ``int64`` array ``k_error`` of the same shape as the inputs.

    Raises
    ------
    ValueError
        If the arrays have different shapes.
    """
    estimated = np.asarray(ambiguity_estimated, dtype=np.int64)
    true = np.asarray(ambiguity_true, dtype=np.int64)
    if estimated.shape != true.shape:
        raise ValueError(
            "ambiguity arrays must have the same shape, got {} and "
            "{}".format(estimated.shape, true.shape))
    error = estimated - true
    if mask is not None:
        valid = np.asarray(mask, dtype=bool)
        if valid.shape != error.shape:
            raise ValueError(
                "mask must have the same shape as the ambiguity arrays, got "
                "{} and {}".format(valid.shape, error.shape))
        error = np.where(valid, error, 0)
    return error


def is_wrapped(phase, atol=1e-9):
    """Return ``True`` if every finite value of ``phase`` lies in ``[-pi, pi]``.

    A cheap sanity check for a generated dataset: if it returns ``False`` you
    forgot to wrap something, and no unwrapping algorithm can possibly
    succeed on the result.

    Parameters
    ----------
    phase : array_like
        Candidate wrapped phase in radians.
    atol : float, optional
        Absolute tolerance in radians.  ``1e-9`` is generous enough to absorb
        the rounding error of ``np.angle`` while still rejecting a phase of
        ``pi + 1e-6``.

    Returns
    -------
    bool
        ``True`` if all finite values are within ``atol`` of the interval; a
        non-finite value (``NaN`` or ``inf``) is ignored, so a masked dataset
        still validates.

    Examples
    --------
    >>> is_wrapped(np.array([0.0, np.pi, -np.pi + 1e-12]))
    True
    >>> is_wrapped(np.array([0.0, 4.0]))
    False
    """
    array = np.asarray(phase, dtype=np.float64)
    finite = np.isfinite(array)
    if not np.any(finite):
        return True
    values = array[finite]
    return bool(np.all(values <= np.pi + atol) and np.all(values > -np.pi - atol))


def wrap_count(phase_true):
    """Count how many whole cycles were folded away, per pixel.

    This is an alternative, phase-space view of the same integer ambiguity and
    is useful when you want to reason about how many fringes the field
    contains without a reference phase to difference against.

    Parameters
    ----------
    phase_true : array_like
        Unwrapped phase in radians.

    Returns
    -------
    ndarray
        ``int64`` array ``k`` with ``phase_true - wrap_phase(phase_true)
        = 2 * pi * k``.

    Examples
    --------
    >>> wrap_count(np.array([0.0, TWO_PI, -TWO_PI])).tolist()
    [0, 1, -1]
    """
    array = np.asarray(phase_true, dtype=np.float64)
    return integer_ambiguity(array, wrap_phase(array))
