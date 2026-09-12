"""Tiny parameter validators shared by every module in this package.

Why a whole module for four functions?  Because the *error messages* are part
of the scientific interface.  If a user asks for a Mogi source at a depth of
``-500`` metres, a good package says so in words; a bad one lets ``NaN`` leak
through four layers of arithmetic and produce a blank plot.  Keeping the
validators in one place also guarantees that the same mistake is reported the
same way whichever model it was passed to.

All validators take the value first and the parameter *name* second, return a
plain Python ``float`` (or ``int``), and raise :exc:`ValueError` on failure.
They deliberately accept anything ``float()`` can convert, so callers do not
have to worry about whether the user passed an ``int``, a ``numpy`` scalar or
a string of digits.
"""

import numpy as np

__all__ = [
    "finite_float",
    "positive_float",
    "positive_int",
    "sign",
    "poisson_ratio",
]


def finite_float(value, name):
    """Return ``value`` as a finite ``float``, or raise :exc:`ValueError`."""
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("{} must be finite, got {!r}".format(name, value))
    return number


def positive_float(value, name):
    """Return ``value`` as a strictly positive finite ``float``."""
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError("{} must be a positive finite number, got {!r}"
                         .format(name, value))
    return number


def positive_int(value, name):
    """Return ``value`` as a strictly positive ``int``."""
    number = int(value)
    if number != value or number <= 0:
        raise ValueError("{} must be a positive integer, got {!r}"
                         .format(name, value))
    return number


def sign(value):
    """Return ``+1.0``, ``-1.0`` or raise :exc:`ValueError`.

    Used for the ``sign`` knobs that let callers match another package's phase
    convention.  Restricting the allowed values to exactly two catches the
    common mistake of passing ``True``/``False`` or ``"-1"``.
    """
    number = float(value)
    if number not in (-1.0, 1.0):
        raise ValueError("sign must be +1.0 or -1.0, got {!r}".format(value))
    return number


def poisson_ratio(value, name="poisson_ratio"):
    """Validate an elastic Poisson ratio.

    Thermodynamic stability of an isotropic elastic solid requires
    ``-1 < nu <= 0.5``.  The upper end is excluded here because the half-space
    formulae used in this package divide by ``1 - nu`` or ``1 - 2*nu``, which
    vanishes in the incompressible limit; ``0.5`` is therefore rejected with a
    message that says so rather than returning infinities.
    """
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("{} must be finite, got {!r}".format(name, value))
    if not -1.0 < number < 0.5:
        raise ValueError(
            "{} must lie strictly between -1 and 0.5, got {!r}; 0.25 is the "
            "usual crustal value".format(name, value))
    return number
