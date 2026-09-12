"""Numerical validation of the coherence-map generators (Phase D).

Coherence is the single number that decides how badly an interferogram's phase
is corrupted, so a synthetic benchmark has to be able to prescribe it exactly
and know that it did.  These tests check that each generator produces the shape
of map its docstring promises, that the maps stay inside ``[0, 1]``, that the
combining rule really is multiplicative, and that a random field is
reproducible for a fixed seed.
"""

import numpy as np
import pytest

from parvaneh.synthetic import (
    COHERENCE_LEVELS,
    NO_DATA,
    Grid,
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


GRID = Grid.centered(nx=41, ny=41, spacing=100.0)


# ---------------------------------------------------------------------------
# Named levels
# ---------------------------------------------------------------------------


def test_named_levels_are_ordered_and_inside_the_unit_interval():
    """The level names must be a usable scale, not an arbitrary lookup table."""
    values = [COHERENCE_LEVELS[name] for name in ("very_low", "low", "moderate",
                                                  "high", "very_high")]
    assert values == sorted(values)
    assert all(0.0 <= value <= 1.0 for value in values)


def test_coherence_from_level_accepts_names_and_numbers():
    """Both spellings of a coherence must reach the same value."""
    assert coherence_from_level("very_high") == COHERENCE_LEVELS["very_high"]
    assert coherence_from_level(0.0) == 0.0
    assert coherence_from_level(1.0) == 1.0
    assert coherence_from_level(0.37) == pytest.approx(0.37)
    # Names are matched loosely: case and separators must not matter.
    assert coherence_from_level(" Very High ") == COHERENCE_LEVELS["very_high"]
    assert coherence_from_level("very-high") == COHERENCE_LEVELS["very_high"]


def test_coherence_from_level_rejects_nonsense():
    """An unknown name or an out-of-range number is a caller error.

    A negative number must raise rather than being silently floored at zero:
    silently repairing a typo is how a benchmark ends up measuring nothing.
    """
    with pytest.raises(ValueError):
        coherence_from_level("extremely_good")
    with pytest.raises(ValueError):
        coherence_from_level(-0.5)
    with pytest.raises(ValueError):
        coherence_from_level(1.5)
    with pytest.raises(ValueError):
        coherence_from_level(np.nan)


def test_no_data_is_not_the_same_as_zero_coherence():
    """``nan`` means "not measured"; ``0.0`` means "measured, and useless"."""
    assert np.isnan(NO_DATA)
    assert NO_DATA != 0.0


# ---------------------------------------------------------------------------
# Uniform map
# ---------------------------------------------------------------------------


def test_uniform_coherence_is_flat_and_shaped_like_the_grid():
    """The clean-case baseline must be exactly constant."""
    grid = Grid(nx=7, ny=5, spacing=10.0)
    gamma = uniform_coherence(grid, 0.63)
    assert gamma.shape == (5, 7)
    assert gamma.dtype == np.float64
    assert np.all(gamma == 0.63)


def test_uniform_coherence_accepts_a_level_name():
    """The level names must work everywhere a coherence value does."""
    gamma = uniform_coherence(GRID, "moderate")
    assert np.all(gamma == COHERENCE_LEVELS["moderate"])


# ---------------------------------------------------------------------------
# Gradient
# ---------------------------------------------------------------------------


def test_coherence_gradient_spans_exactly_the_requested_range():
    """Whatever the grid, the map must reach ``low`` and ``high`` exactly."""
    gamma = coherence_gradient(GRID, low=0.2, high=0.95)
    assert float(gamma.min()) == pytest.approx(0.2)
    assert float(gamma.max()) == pytest.approx(0.95)


def test_coherence_gradient_increases_towards_the_azimuth_direction():
    """Azimuth 0 points north, so the northern rows must be the coherent ones.

    This is also a check on the row ordering: row 0 is the northernmost row.
    """
    gamma = coherence_gradient(GRID, low=0.2, high=0.95, azimuth_deg=0.0)
    assert float(gamma[0, 20]) > float(gamma[-1, 20])
    gamma_east = coherence_gradient(GRID, low=0.2, high=0.95, azimuth_deg=90.0)
    assert float(gamma_east[20, -1]) > float(gamma_east[20, 0])
    gamma_south = coherence_gradient(GRID, low=0.2, high=0.95, azimuth_deg=180.0)
    assert float(gamma_south[-1, 20]) > float(gamma_south[0, 20])


def test_coherence_gradient_is_antisymmetric_between_opposite_azimuths():
    """Reversing the direction must swap the two ends, nothing else."""
    forward = coherence_gradient(GRID, low=0.2, high=0.95, azimuth_deg=0.0)
    backward = coherence_gradient(GRID, low=0.2, high=0.95, azimuth_deg=180.0)
    assert np.allclose(forward + backward, 0.2 + 0.95)


def test_coherence_gradient_rejects_low_above_high():
    """The parameters have a direction; getting it backwards is an error."""
    with pytest.raises(ValueError):
        coherence_gradient(GRID, low=0.9, high=0.1)


# ---------------------------------------------------------------------------
# Gaussian patch
# ---------------------------------------------------------------------------


def test_gaussian_patch_reaches_the_requested_depth_and_background():
    """The centre must hit ``background - depth`` and the far field the background."""
    grid = Grid.centered(nx=21, ny=21, spacing=100.0)
    gamma = gaussian_coherence_patch(grid, 0.0, 0.0, 200.0, 0.6)
    assert float(gamma[10, 10]) == pytest.approx(0.4)
    assert float(gamma[0, 0]) == pytest.approx(1.0)


def test_gaussian_patch_is_radially_symmetric_and_increasing_outwards():
    """Coherence must recover monotonically with distance from the centre."""
    gamma = gaussian_coherence_patch(GRID, 0.0, 0.0, 300.0, 0.7)
    centre = gamma[GRID.ny // 2, GRID.nx // 2]
    assert gamma[0, 0] == pytest.approx(gamma[0, -1])
    assert gamma[0, 0] == pytest.approx(gamma[-1, 0])
    assert gamma[0, 0] > centre
    row = gamma[GRID.ny // 2, GRID.nx // 2:]
    assert np.all(np.diff(row) >= 0.0)


def test_gaussian_patch_is_clipped_rather_than_going_negative():
    """A patch deeper than the background gives a genuine dead core."""
    gamma = gaussian_coherence_patch(GRID, 0.0, 0.0, 500.0, 4.0)
    assert float(gamma.min()) == 0.0
    assert np.all(gamma >= 0.0)
    assert np.all(gamma <= 1.0)


def test_gaussian_patch_is_offset_by_its_centre():
    """The patch must land where it is asked to land, not at the origin."""
    gamma = gaussian_coherence_patch(GRID, 1000.0, -500.0, 200.0, 0.5)
    assert gamma[GRID.ny // 2 - 5, GRID.nx // 2 + 10] < gamma[0, 0]


# ---------------------------------------------------------------------------
# Stripe
# ---------------------------------------------------------------------------


def test_stripe_low_coherence_lies_inside_the_band_only():
    """The band is a hard-edged ribbon; everything outside is background."""
    grid = Grid(nx=11, ny=11, spacing=100.0, x_min=-500.0, y_max=500.0)
    gamma = decorrelation_stripe(grid, width=300.0, coherence=0.3)
    assert float(gamma[5, 5]) == pytest.approx(0.3)
    assert float(gamma[5, 0]) == pytest.approx(1.0)
    assert float(gamma[5, 8]) == pytest.approx(1.0)


def test_stripe_runs_along_its_azimuth():
    """A north-south band must vary across columns, not across rows."""
    grid = Grid(nx=21, ny=21, spacing=100.0, x_min=-1000.0, y_max=1000.0)
    gamma = decorrelation_stripe(grid, width=400.0, coherence=0.2)
    column = gamma[:, 10]
    assert np.all(column == column[0])
    assert np.all(gamma[:, 0] == 1.0)


def test_stripe_offset_shifts_the_band_east_for_a_north_south_band():
    """A positive offset moves the band in the ``azimuth + 90`` direction."""
    grid = Grid(nx=21, ny=21, spacing=100.0, x_min=-1000.0, y_max=1000.0)
    gamma = decorrelation_stripe(grid, width=200.0, coherence=0.2, offset=500.0)
    low_columns = np.where(np.all(gamma == 0.2, axis=0))[0]
    assert low_columns.size > 0
    assert float(grid.x[low_columns].mean()) == pytest.approx(500.0, abs=50.0)


def test_stripe_rotated_by_ninety_degrees_varies_along_rows_instead():
    """Rotating the long axis must rotate the ribbon with it."""
    grid = Grid(nx=21, ny=21, spacing=100.0, x_min=-1000.0, y_max=1000.0)
    gamma = decorrelation_stripe(grid, width=400.0, coherence=0.2,
                                 azimuth_deg=90.0)
    assert np.all(gamma[10, :] == gamma[10, 0])
    assert np.all(gamma[0, :] == 1.0)


# ---------------------------------------------------------------------------
# Fault-zone band
# ---------------------------------------------------------------------------


def test_fault_zone_follows_a_bent_trace():
    """A polyline trace must produce a bent band, which a stripe cannot."""
    trace_x = [-800.0, 0.0, 800.0]
    trace_y = [-800.0, 0.0, 800.0]
    gamma = fault_zone_coherence(GRID, trace_x, trace_y, width=300.0,
                                 coherence=0.2)
    assert float(gamma.min()) == pytest.approx(0.2)
    assert float(gamma.max()) == pytest.approx(1.0)
    # The trace is the diagonal y = x.  Its two ends are inside the grid and
    # the band must sit on top of them; the corners off the diagonal must not.
    assert float(gamma[20, 20]) == pytest.approx(0.2)   # the origin
    assert float(gamma[12, 28]) == pytest.approx(0.2)   # (800, 800)
    assert float(gamma[28, 12]) == pytest.approx(0.2)   # (-800, -800)
    assert float(gamma[0, 0]) == pytest.approx(1.0)
    assert float(gamma[0, -1]) == pytest.approx(1.0)


def test_fault_zone_band_has_the_requested_width():
    """Measure the band across a straight trace and compare with ``width``.

    A north-south trace through the origin must give a band whose width in the
    east direction is the width that was asked for.
    """
    grid = Grid(nx=81, ny=81, spacing=10.0, x_min=-400.0, y_max=400.0)
    gamma = fault_zone_coherence(grid, [0.0, 0.0], [-400.0, 400.0], width=200.0,
                                 coherence=0.0)
    inside = np.sum(gamma[40, :] == 0.0)
    assert inside * grid.spacing == pytest.approx(200.0, abs=grid.spacing)


def test_fault_zone_needs_a_real_polyline():
    """One vertex is not a trace."""
    with pytest.raises(ValueError):
        fault_zone_coherence(GRID, [0.0], [0.0], width=100.0)


# ---------------------------------------------------------------------------
# Random field
# ---------------------------------------------------------------------------


def test_random_coherence_is_reproducible_and_lies_in_range():
    """Same seed, same field; and the result is always a legal coherence."""
    first = random_coherence(GRID, rng=np.random.default_rng(20240101))
    second = random_coherence(GRID, rng=np.random.default_rng(20240101))
    different = random_coherence(GRID, rng=np.random.default_rng(20240102))
    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)
    assert np.all(first >= 0.0)
    assert np.all(first <= 1.0)


def test_random_coherence_mean_is_close_to_the_requested_mean():
    """The requested mean must actually be the mean, to within reason."""
    gamma = random_coherence(Grid(nx=128, ny=128, spacing=50.0), mean=0.75,
                             sigma=0.12, correlation_length=200.0,
                             rng=np.random.default_rng(7))
    assert float(gamma.mean()) == pytest.approx(0.75, abs=0.05)


def test_random_coherence_is_smoother_when_the_correlation_length_grows():
    """A longer correlation length must give a smoother field.

    "Smoother" is measured as the mean absolute difference between neighbouring
    pixels, which is a scale-free way to compare two fields on the same grid.
    """
    grid = Grid(nx=128, ny=128, spacing=50.0)
    rough = random_coherence(grid, correlation_length=100.0,
                             rng=np.random.default_rng(3))
    smooth = random_coherence(grid, correlation_length=1500.0,
                              rng=np.random.default_rng(3))
    roughness = lambda field: float(np.abs(np.diff(field, axis=1)).mean())
    assert roughness(smooth) < roughness(rough)


# ---------------------------------------------------------------------------
# No-data mask
# ---------------------------------------------------------------------------


def test_circular_no_data_mask_is_boolean_and_centred():
    """The mask must be a boolean array, ``True`` where there is no data."""
    grid = Grid.centered(nx=21, ny=21, spacing=100.0)
    mask = circular_no_data_mask(grid, 0.0, 0.0, 250.0)
    assert mask.dtype == bool
    assert bool(mask[10, 10])
    assert not bool(mask[0, 0])
    assert int(mask.sum()) == 21


def test_circular_no_data_mask_moves_with_its_centre():
    """Offsetting the centre must move the hole, not resize it."""
    grid = Grid.centered(nx=21, ny=21, spacing=100.0)
    centred = circular_no_data_mask(grid, 0.0, 0.0, 250.0)
    shifted = circular_no_data_mask(grid, 300.0, 300.0, 250.0)
    assert int(centred.sum()) == int(shifted.sum())
    assert not bool(shifted[10, 10])


# ---------------------------------------------------------------------------
# Combining
# ---------------------------------------------------------------------------


def test_combine_coherence_multiplies_its_inputs():
    """Independent decorrelation mechanisms compound multiplicatively."""
    a = np.full((4, 4), 0.6)
    b = np.full((4, 4), 0.6)
    combined = combine_coherence(a, b)
    assert np.allclose(combined, 0.36)


def test_combine_coherence_propagates_no_data():
    """A single no-data map makes the pixel no-data overall."""
    a = np.full((3, 3), 0.9)
    b = np.full((3, 3), 0.9)
    b[0, 0] = NO_DATA
    combined = combine_coherence(a, b)
    assert np.isnan(combined[0, 0])
    assert combined[1, 1] == pytest.approx(0.81)


def test_combine_coherence_handles_three_maps():
    """The product must generalise beyond the two-map example."""
    maps = [np.full((2, 2), 0.5), np.full((2, 2), 0.5), np.full((2, 2), 0.5)]
    assert np.allclose(combine_coherence(*maps), 0.125)


def test_combine_coherence_rejects_bad_input():
    """Too few maps, or mismatched shapes, must be reported not ignored."""
    with pytest.raises(ValueError):
        combine_coherence(np.ones((2, 2)))
    with pytest.raises(ValueError):
        combine_coherence(np.ones((2, 2)), np.ones((3, 3)))


# ---------------------------------------------------------------------------
# Argument validation shared by the generators
# ---------------------------------------------------------------------------


def test_generators_reject_something_that_is_not_a_grid():
    """A grid-like object without ``X`` and ``Y`` is not a grid."""
    with pytest.raises(ValueError):
        uniform_coherence(np.zeros((4, 4)), 0.5)
    with pytest.raises(ValueError):
        circular_no_data_mask(object(), 0.0, 0.0, 10.0)


def test_generators_produce_the_shape_of_the_grid():
    """Every generator must agree with the grid on shape, without exception."""
    for gamma in (
        uniform_coherence(GRID),
        coherence_gradient(GRID),
        gaussian_coherence_patch(GRID, 0.0, 0.0, 200.0, 0.5),
        decorrelation_stripe(GRID, width=200.0),
        fault_zone_coherence(GRID, [-100.0, 100.0], [0.0, 0.0], width=100.0),
        random_coherence(GRID, rng=np.random.default_rng(1)),
    ):
        assert gamma.shape == GRID.shape
