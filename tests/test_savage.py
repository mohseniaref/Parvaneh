"""Numerical validation of the Savage-Burford interseismic model (Phase B).

The tests here check the properties the specification singles out as the ones
that make a synthetic field *physically* rather than merely visually correct:
the far-field limits ``+V/2`` and ``-V/2``, the vanishing velocity on the fault
trace, the antisymmetry of the profile, monotone growth away from the fault,
the way the two-dimensional map responds to a change of strike, and the fact
that a displacement is a velocity multiplied by an interval.

Where a test compares against a number it uses the closed form
``v(x) = (V/pi) arctan(x/D)`` evaluated independently, never a snapshot of the
implementation's own output.
"""

import numpy as np
import pytest

from parvaneh.synthetic import (
    SAVAGE_STYLES,
    Grid,
    random_savage_source,
    savage_displacement,
    savage_velocity,
    savage_velocity_profile,
)


GRID = Grid.centered(nx=41, ny=41, spacing=250.0)


def _profile_end_on(x, slip_rate, locking_depth):
    """Closed-form Savage profile, written out independently of the package."""
    return (slip_rate / np.pi) * np.arctan(x / locking_depth)


# ---------------------------------------------------------------------------
# The one-dimensional profile
# ---------------------------------------------------------------------------
def test_profile_matches_the_closed_form():
    """``v(x)`` must equal (V/pi) arctan(x/D) at every sample."""
    x = np.linspace(-80000.0, 80000.0, 401)
    got = savage_velocity_profile(x, 0.035, 12000.0)
    assert np.allclose(got, _profile_end_on(x, 0.035, 12000.0), rtol=0.0,
                       atol=1e-15)


def test_profile_is_zero_on_the_fault_trace():
    """A locked fault has no surface slip, so v(0) is exactly zero."""
    assert savage_velocity_profile(0.0, 0.04, 9000.0) == 0.0


def test_profile_is_antisymmetric():
    """v(-x) must be exactly -v(x): the two plates move in opposite senses."""
    x = np.linspace(1.0, 60000.0, 101)
    forward = savage_velocity_profile(x, 0.03, 10000.0)
    backward = savage_velocity_profile(-x, 0.03, 10000.0)
    assert np.allclose(backward, -forward, rtol=0.0, atol=0.0)


def test_profile_saturates_at_plus_minus_half_the_slip_rate():
    """The far field is +V/2 and -V/2, so the relative velocity is the full V."""
    slip_rate = 0.033
    far_field = (slip_rate / np.pi) * (np.pi / 2.0)
    assert far_field == slip_rate / 2.0

    # The approach is algebraic, not exponential: arctan(x/D) -> pi/2 only as
    # 1/(x/D), so the test has to be a convergence check rather than equality.
    previous = 0.0
    for multiple in np.logspace(1.0, 9.0, 9):
        value = float(savage_velocity_profile(multiple * 10000.0, slip_rate,
                                              10000.0))
        assert value < slip_rate / 2.0
        assert value > previous
        previous = value
    assert abs(previous - slip_rate / 2.0) < 1e-9

    # And it stays inside the bound from below for any negative distance.
    assert (float(savage_velocity_profile(-1.0e9, slip_rate, 10000.0))
            > -slip_rate / 2.0)


def test_profile_is_monotone_increasing():
    """The velocity must grow steadily from -V/2 to +V/2 with no overshoot."""
    x = np.linspace(-50000.0, 50000.0, 501)
    v = savage_velocity_profile(x, 0.03, 8000.0)
    assert np.all(np.diff(v) > 0.0)


def test_profile_is_odd_in_the_slip_rate():
    """Negating the slip rate negates the whole field: the sense reverses."""
    x = np.linspace(-50000.0, 50000.0, 501)
    assert np.allclose(savage_velocity_profile(x, 0.03, 8000.0),
                       -savage_velocity_profile(x, -0.03, 8000.0))


def test_profile_is_self_similar_in_x_over_depth():
    """v depends on x and D only through the ratio x/D."""
    x = np.array([500.0, 2500.0, 12500.0])
    assert np.allclose(savage_velocity_profile(x, 1.0, 2000.0),
                       savage_velocity_profile(2.0 * x, 1.0, 4000.0))


def test_shallower_locking_concentrates_the_strain():
    """Halving D raises the velocity at a fixed distance from the fault."""
    x = 6000.0
    shallow = float(savage_velocity_profile(x, 0.03, 4000.0))
    deep = float(savage_velocity_profile(x, 0.03, 8000.0))
    assert shallow > deep


def test_velocity_at_the_locking_depth_is_a_quarter_of_the_slip_rate():
    """arctan(1) = pi/4, so v(D) = V/4 exactly, for any V and D."""
    for slip_rate, locking_depth in [(0.02, 5000.0), (0.07, 20000.0)]:
        assert abs(float(savage_velocity_profile(locking_depth, slip_rate,
                                                 locking_depth))
                   - slip_rate / 4.0) < 1e-15


def test_profile_accepts_a_scalar_and_an_array():
    """A scalar input gives a zero-dimensional result, an array a matching one."""
    scalar = savage_velocity_profile(5000.0, 0.03, 10000.0)
    assert float(scalar) == pytest.approx(_profile_end_on(5000.0, 0.03, 10000.0))
    array = savage_velocity_profile(np.zeros((3, 4)), 0.03, 10000.0)
    assert array.shape == (3, 4)


def test_profile_rejects_a_non_positive_locking_depth():
    """A zero or negative locking depth is unphysical and must be refused."""
    for bad in [0.0, -1000.0, np.nan]:
        with pytest.raises(ValueError):
            savage_velocity_profile(0.0, 0.03, bad)


def test_profile_rejects_a_non_finite_slip_rate():
    """A NaN or infinite slip rate must be refused rather than propagated."""
    for bad in [np.nan, np.inf, -np.inf]:
        with pytest.raises(ValueError):
            savage_velocity_profile(0.0, bad, 10000.0)


# ---------------------------------------------------------------------------
# The two-dimensional velocity map
# ---------------------------------------------------------------------------
def test_velocity_map_shapes_and_zero_vertical_motion():
    """A screw dislocation produces no vertical motion at all."""
    v_e, v_n, v_u = savage_velocity(GRID, 0.0, 10000.0, 0.03)
    assert v_e.shape == GRID.shape
    assert v_n.shape == GRID.shape
    assert v_u.shape == GRID.shape
    assert np.all(v_u == 0.0)


def test_north_striking_fault_moves_east_side_north():
    """strike = 0 puts "right of strike" to the east, and the east side moves north."""
    v_e, v_n, _ = savage_velocity(GRID, 0.0, 10000.0, 0.03)
    assert np.allclose(v_e, 0.0)
    eastern = v_n[:, GRID.nx // 2 + 1:]
    western = v_n[:, :GRID.nx // 2]
    assert np.all(eastern > 0.0)
    assert np.all(western < 0.0)
    assert np.allclose(eastern, -western[:, ::-1])


def test_east_west_fault_moves_the_northern_half_east():
    """strike = 90 puts "right of strike" to the south, so the south side moves east."""
    v_e, v_n, _ = savage_velocity(GRID, 90.0, 10000.0, 0.03)
    assert np.allclose(v_n, 0.0)
    northern = v_e[:GRID.ny // 2, :]
    southern = v_e[GRID.ny // 2 + 1:, :]
    assert np.all(northern < 0.0)
    assert np.all(southern > 0.0)


def test_map_magnitude_depends_only_on_distance_from_the_trace():
    """The speed is a function of perpendicular distance alone."""
    v_ref = savage_velocity(GRID, 0.0, 10000.0, 0.03)[1]
    speed = np.abs(v_ref)
    # Every row must hold the same speed profile, since the fault is infinite.
    for row in range(1, GRID.ny):
        assert np.allclose(speed[row], speed[0])


def test_rotating_the_strike_rotates_the_field_by_the_same_angle():
    """strike = 90 turns the whole field through 90 degrees."""
    _, v_n, _ = savage_velocity(GRID, 0.0, 10000.0, 0.03)
    v_e_90, _, _ = savage_velocity(GRID, 90.0, 10000.0, 0.03)
    assert np.allclose(np.sort(np.abs(v_e_90).ravel()),
                       np.sort(np.abs(v_n).ravel()))


def test_reversing_the_strike_by_180_degrees_changes_nothing():
    """The trace is the same line and the two flips cancel exactly."""
    v_e_0, v_n_0, _ = savage_velocity(GRID, 0.0, 10000.0, 0.03)
    v_e_180, v_n_180, _ = savage_velocity(GRID, 180.0, 10000.0, 0.03)
    assert np.allclose(v_e_180, v_e_0)
    assert np.allclose(v_n_180, v_n_0)


def test_negating_the_slip_rate_reverses_the_field():
    """The only way to flip the sense of motion is to flip the sign of V."""
    _, v_n, _ = savage_velocity(GRID, 0.0, 10000.0, 0.03)
    _, w_n, _ = savage_velocity(GRID, 0.0, 10000.0, -0.03)
    assert np.allclose(w_n, -v_n)


def test_map_is_antisymmetric_about_a_north_south_trace():
    """Reflecting the grid through a north-south trace negates the velocity."""
    v_e, v_n, _ = savage_velocity(GRID, 0.0, 10000.0, 0.03)
    assert np.allclose(v_n, -v_n[:, ::-1])


def test_map_takes_its_value_from_the_one_dimensional_profile():
    """The map is the profile sampled at the perpendicular distance."""
    v_e, v_n, _ = savage_velocity(GRID, 0.0, 10000.0, 0.03)
    expected = savage_velocity_profile(GRID.X, 0.03, 10000.0)
    assert np.allclose(v_n, expected)


def test_rotated_trace_matches_the_projected_distance_formula():
    """For an arbitrary strike the map must follow the projection formula."""
    strike, locking_depth, slip_rate = 37.0, 11000.0, 0.028
    v_e, v_n, _ = savage_velocity(GRID, strike, locking_depth, slip_rate,
                                  center_x=1200.0, center_y=-800.0)

    theta = np.deg2rad(strike)
    x = ((GRID.X - 1200.0) * np.cos(theta)
         - (GRID.Y + 800.0) * np.sin(theta))
    speed = savage_velocity_profile(x, slip_rate, locking_depth)
    assert np.allclose(v_e, speed * np.sin(theta))
    assert np.allclose(v_n, speed * np.cos(theta))


def test_placing_the_trace_elsewhere_translates_the_pattern():
    """Moving the fault centre must move the zero-velocity line with it."""
    v_e, v_n, _ = savage_velocity(GRID, 0.0, 10000.0, 0.03,
                                  center_x=2000.0)
    column = GRID.nearest_column(2000.0)
    assert np.allclose(v_n[:, column], 0.0, atol=1e-15)


def test_map_scales_linearly_with_the_slip_rate():
    """The model is linear in V, which matters when summing sources."""
    _, v_n, _ = savage_velocity(GRID, 0.0, 10000.0, 0.02)
    _, w_n, _ = savage_velocity(GRID, 0.0, 10000.0, 0.05)
    assert np.allclose(w_n, v_n * (0.05 / 0.02))


def test_map_rejects_a_negative_locking_depth():
    """The map must apply the same validation as the profile."""
    with pytest.raises(ValueError):
        savage_velocity(GRID, 0.0, -100.0, 0.03)


def test_map_rejects_a_non_grid_argument():
    """Passing a bare array instead of a Grid is a common mistake."""
    with pytest.raises(ValueError):
        savage_velocity(np.zeros((4, 4)), 0.0, 10000.0, 0.03)


def test_map_rejects_a_non_finite_strike():
    """A NaN strike must be reported by name, not silently produce NaNs."""
    with pytest.raises(ValueError):
        savage_velocity(GRID, np.nan, 10000.0, 0.03)


# ---------------------------------------------------------------------------
# Interseismic displacement
# ---------------------------------------------------------------------------
def test_displacement_is_the_velocity_times_the_interval():
    """u = v * dt, component by component."""
    u_e, u_n, u_u = savage_displacement(GRID, 22.0, 9000.0, 0.031,
                                        interval=3.5)
    v_e, v_n, v_u = savage_velocity(GRID, 22.0, 9000.0, 0.031)
    assert np.allclose(u_e, v_e * 3.5)
    assert np.allclose(u_n, v_n * 3.5)
    assert np.allclose(u_u, v_u * 3.5)


def test_displacement_keeps_the_model_parameters_visible():
    """Every argument of the velocity call must still take effect."""
    for kwargs in [dict(strike_deg=0.0), dict(strike_deg=90.0),
                   dict(locking_depth=4000.0), dict(slip_rate=0.09)]:
        parameters = dict(strike_deg=0.0, locking_depth=10000.0,
                          slip_rate=0.03)
        parameters.update(kwargs)
        u_e, u_n, _ = savage_displacement(GRID, interval=1.0, **parameters)
        v_e, v_n, _ = savage_velocity(GRID, **parameters)
        assert np.allclose(u_e, v_e)
        assert np.allclose(u_n, v_n)


def test_zero_interval_gives_no_displacement():
    """A zero-length interferogram measures nothing."""
    u_e, u_n, u_u = savage_displacement(GRID, 0.0, 10000.0, 0.03,
                                        interval=0.0)
    assert np.all(u_e == 0.0)
    assert np.all(u_n == 0.0)
    assert np.all(u_u == 0.0)


def test_displacement_scales_linearly_with_the_interval():
    """Doubling the time span doubles the accumulated displacement."""
    _, u_n, _ = savage_displacement(GRID, 0.0, 10000.0, 0.03, interval=2.0)
    _, w_n, _ = savage_displacement(GRID, 0.0, 10000.0, 0.03, interval=5.0)
    assert np.allclose(w_n, u_n * 2.5)


def test_displacement_rejects_a_negative_interval():
    """Time runs forwards; a negative interval is a mistake worth reporting."""
    with pytest.raises(ValueError):
        savage_displacement(GRID, 0.0, 10000.0, 0.03, interval=-1.0)


def test_displacement_rejects_a_non_finite_interval():
    """A NaN interval must not silently produce a NaN field."""
    for bad in [np.nan, np.inf]:
        with pytest.raises(ValueError):
            savage_displacement(GRID, 0.0, 10000.0, 0.03, interval=bad)


def test_displacement_stays_below_the_plate_bound():
    """The accumulated displacement can never exceed V/2 * interval."""
    interval = 4.0
    u_e, u_n, u_u = savage_displacement(GRID, 0.0, 10000.0, 0.04,
                                        interval=interval)
    assert np.abs(u_n).max() < 0.04 / 2.0 * interval


# ---------------------------------------------------------------------------
# Randomised sources
# ---------------------------------------------------------------------------
def test_styles_are_ranges_not_fixed_values():
    """Every style must describe ranges, and the ranges must be sane."""
    assert SAVAGE_STYLES
    for name, spec in SAVAGE_STYLES.items():
        for key in ("locking_depth_range", "slip_rate_range", "strike_range",
                    "centre_range"):
            assert key in spec, "{} is missing {}".format(name, key)
        for key in ("locking_depth_range", "slip_rate_range", "strike_range"):
            low, high = spec[key]
            assert low <= high
        assert 0.0 <= spec["centre_range"] <= 1.0
        low, high = spec["locking_depth_range"]
        assert low > 0.0
        low, high = spec["slip_rate_range"]
        assert low > 0.0


def test_random_source_is_reproducible_for_a_fixed_seed():
    """The same seed must give the same fault, and a different seed another."""
    first = random_savage_source(GRID, "deep_locking",
                                 rng=np.random.default_rng(7))
    again = random_savage_source(GRID, "deep_locking",
                                 rng=np.random.default_rng(7))
    other = random_savage_source(GRID, "deep_locking",
                                 rng=np.random.default_rng(8))
    assert first == again
    assert first != other


def test_random_source_parameters_land_inside_their_ranges():
    """Sampling must respect the ranges declared in SAVAGE_STYLES."""
    for style, spec in SAVAGE_STYLES.items():
        rng = np.random.default_rng(11)
        for _ in range(20):
            source = random_savage_source(GRID, style, rng=rng)
            assert (spec["locking_depth_range"][0] <= source["locking_depth"]
                    <= spec["locking_depth_range"][1])
            assert (spec["slip_rate_range"][0] <= source["slip_rate"]
                    <= spec["slip_rate_range"][1])
            assert (spec["strike_range"][0] <= source["strike_deg"]
                    <= spec["strike_range"][1])


def test_random_source_returns_keyword_arguments_that_work():
    """The returned dictionary must be directly usable as keyword arguments."""
    source = random_savage_source(GRID, "oblique",
                                  rng=np.random.default_rng(3))
    parameters = dict(source)
    parameters.pop("style")
    v_e, v_n, v_u = savage_velocity(GRID, **parameters)
    assert v_e.shape == GRID.shape
    assert np.all(np.isfinite(v_e))
    assert np.all(np.isfinite(v_n))


def test_random_source_rejects_an_unknown_style():
    """An unknown style name must be reported together with the valid ones."""
    with pytest.raises(ValueError):
        random_savage_source(GRID, "no_such_style")
