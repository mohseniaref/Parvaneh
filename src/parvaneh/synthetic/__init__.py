"""Physically meaningful synthetic InSAR datasets with exact ground truth.

This subpackage builds interferometric phase fields for which the *true*
unwrapped phase is known analytically, so that a phase-unwrapping algorithm can
be scored numerically instead of judged by eye.  It grew out of the single
``parvaneh.synthetic`` module, which is preserved verbatim as
:mod:`parvaneh.synthetic.legacy` so that existing notebooks,
benchmarks and scripts keep working.

Why bother
----------

A real interferogram has no ground truth: nobody knows the continuous phase
that produced it.  Benchmarks built on real data therefore compare algorithms
against *each other*, which cannot distinguish "all of them are right" from
"all of them are wrong in the same way".  A synthetic scene inverts this.  The
deformation field is produced by a closed-form elastic model, the phase follows
from that field by an exact formula, and the integer number of cycles lost in
wrapping is recorded as the data are created.  Every quantity an unwrapper is
supposed to recover is therefore available to arbitrary precision.

The construction is a pipeline, and each stage is kept separate so that it can
be inspected and tested on its own:

1. :mod:`~parvaneh.synthetic.grid` -- the sampling grid.
2. :mod:`~parvaneh.synthetic.mogi`,
   :mod:`~parvaneh.synthetic.okada`,
   :mod:`~parvaneh.synthetic.savage` -- ground displacement in ENU
   coordinates, with no notion of radar in them at all.
3. :mod:`~parvaneh.synthetic.geometry` -- projection onto the
   satellite line of sight, and conversion from displacement to phase.
4. :mod:`~parvaneh.synthetic.atmosphere`,
   :mod:`~parvaneh.synthetic.noise`,
   :mod:`~parvaneh.synthetic.coherence` -- the components that make
   the scene realistic.
5. :mod:`~parvaneh.synthetic.wrapping`,
   :mod:`~parvaneh.synthetic.residues` -- the wrapping that destroys
   information, and the residues that quantify the damage.
6. :mod:`~parvaneh.synthetic.masks`,
   :mod:`~parvaneh.synthetic.scenarios`,
   :mod:`~parvaneh.synthetic.dataset` -- assembling complete,
   reproducible, self-describing benchmarks.

Conventions that hold everywhere in this subpackage
---------------------------------------------------

* **Units are SI**: metres, radians, seconds, cubic metres.  An angle parameter
  whose name does not end in ``_deg`` is in radians, always.
* **Coordinates are East-North-Up**, with ``x`` increasing to the east and
  ``y`` increasing to the north.  Displacement arrays are ``(ny, nx)``.
* **Row 0 is the northernmost row**, so ``y`` decreases as the row index
  increases.  This matches ``matplotlib.imshow(origin="upper")`` and therefore
  lets diagnostic images be plotted without flips.  Column 0 is the
  westernmost column.
* **Lines of sight point from the ground toward the satellite**, and a
  positive line-of-sight displacement means the ground moved *toward* the
  satellite (range decreased).  Interferometric phase is
  :math:`\\Phi = +(4\\pi/\\lambda) \\, d_{\\mathrm{LOS}}`.
* **Masked pixels are ``NaN``**, never zero, in displacement and phase arrays.
  A zero would be indistinguishable from "no motion", which is a different
  thing.
* **Wrapping uses the complex form** ``np.angle(np.exp(1j * phase))`` and
  returns the interval :math:`[-\\pi, \\pi]`.

.. note::

   Two functions named ``wrap_phase`` exist for historical reasons.  The one
   exported here is the canonical complex-exponential version; the older
   modulo version lives in :mod:`parvaneh.synthetic.legacy`.  They
   agree everywhere except exactly at odd multiples of :math:`\\pi`, where the
   modulo form always returns ``-pi`` while the complex form returns ``-pi``
   for :math:`-\\pi` and a value one unit in the last place below ``+pi`` for
   :math:`3\\pi` and beyond.  Both are valid wrappings of the same angle: the
   endpoints differ by exactly one cycle.

References
----------
Mogi, K. (1958), *Bull. Earthquake Res. Inst. Univ. Tokyo* **36**, 99-134.
Okada, Y. (1985), *Bull. Seismol. Soc. Am.* **75**, 1135-1154.
Savage, J. C. and Burford, R. O. (1973), *J. Geophys. Res.* **78**, 832-845.
Ghiglia, D. C. and Pritt, M. D. (1998), *Two-Dimensional Phase Unwrapping*,
Wiley.
"""

from .wrapping import (
    TWO_PI,
    ambiguity_error,
    integer_ambiguity,
    is_wrapped,
    unwrap_with_ambiguity,
    wrap_count,
    wrap_phase,
)

from .geometry import (
    SPEED_OF_LIGHT,
    WAVELENGTHS,
    SarGeometry,
    displacement_to_phase,
    fringe_per_metre,
    ground_track_heading,
    heading_from_look_azimuth,
    look_azimuth_from_heading,
    los_unit_vector,
    phase_to_displacement,
    project_to_los,
    wavelength_for,
)

from .grid import (
    DEFAULT_NX,
    DEFAULT_NY,
    DEFAULT_SPACING,
    MAX_PIXELS,
    Grid,
    make_grid,
)

from .mogi import (
    MOGI_STYLES,
    mogi_displacement,
    mogi_displacement_multi,
    mogi_point_displacement,
    random_mogi_source,
)

from .savage import (
    SAVAGE_STYLES,
    random_savage_source,
    savage_displacement,
    savage_velocity,
    savage_velocity_profile,
)

from .residues import (
    aliasing_mask,
    boundary_circulation,
    negative_residues,
    phase_gradients,
    plaquette_valid_mask,
    positive_residues,
    residue_balance,
    residue_density,
    residue_map,
    wrapped_gradients,
)

from .coherence import (
    COHERENCE_LEVELS,
    NO_DATA,
    circular_no_data_mask,
    coherence_from_level,
    coherence_gradient,
    combine_coherence,
    decorrelation_stripe,
    fault_zone_coherence,
    gaussian_coherence_patch,
    random_coherence,
    uniform_coherence,
)

from .noise import (
    DEFAULT_LOOKS,
    UNIFORM_PHASE_STD,
    add_phase_noise,
    phase_noise,
    phase_noise_std,
    single_look_phase_pdf,
)

from .legacy import make_synthetic, rmse_aligned

__all__ = [
    # legacy module API
    "make_synthetic",
    "rmse_aligned",
    # grid
    "Grid",
    "make_grid",
    "DEFAULT_NX",
    "DEFAULT_NY",
    "DEFAULT_SPACING",
    "MAX_PIXELS",
    # geometry
    "SarGeometry",
    "SPEED_OF_LIGHT",
    "WAVELENGTHS",
    "wavelength_for",
    "los_unit_vector",
    "look_azimuth_from_heading",
    "heading_from_look_azimuth",
    "ground_track_heading",
    "project_to_los",
    "displacement_to_phase",
    "phase_to_displacement",
    "fringe_per_metre",
    # mogi
    "MOGI_STYLES",
    "mogi_point_displacement",
    "mogi_displacement",
    "mogi_displacement_multi",
    "random_mogi_source",
    # wrapping
    "TWO_PI",
    "wrap_phase",
    "integer_ambiguity",
    "unwrap_with_ambiguity",
    "ambiguity_error",
    "is_wrapped",
    "wrap_count",
    # residues
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
    # coherence
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
    # noise
    "DEFAULT_LOOKS",
    "UNIFORM_PHASE_STD",
    "phase_noise",
    "add_phase_noise",
    "phase_noise_std",
    "single_look_phase_pdf",
    # savage
    "SAVAGE_STYLES",
    "savage_velocity_profile",
    "savage_velocity",
    "savage_displacement",
    "random_savage_source",
]
