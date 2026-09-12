"""Numerical validation of the synthetic InSAR generator (Phase A).

Every test here checks a *known* mathematical property of a model, not the
appearance of a picture.  The list follows the validation requirements of the
generator specification: the wrapping identity, exact phase reconstruction from
the integer ambiguity, zero-deformation behaviour, the line-of-sight projection,
Mogi symmetry and decay, the residue/boundary-circulation identity, aliasing
detection, and reproducibility for a fixed seed.
"""

import numpy as np
import pytest

from accelerated_unwrap.synthetic import (
    Grid,
    SarGeometry,
    aliasing_mask,
    ambiguity_error,
    boundary_circulation,
    displacement_to_phase,
    fringe_per_metre,
    integer_ambiguity,
    los_unit_vector,
    make_grid,
    mogi_displacement,
    mogi_displacement_multi,
    mogi_point_displacement,
    phase_gradients,
    phase_to_displacement,
    positive_residues,
    negative_residues,
    project_to_los,
    random_mogi_source,
    residue_balance,
    residue_density,
    residue_map,
    unwrap_with_ambiguity,
    wavelength_for,
    wrap_phase,
    wrapped_gradients,
)


SENTINEL_1 = wavelength_for("sentinel-1")


# ---------------------------------------------------------------------------
# Wrapping and the integer ambiguity
# ---------------------------------------------------------------------------

def test_wrap_phase_matches_complex_definition():
    """The wrapping must be angle(exp(i phi)), on a genuinely wild field."""
    rng = np.random.default_rng(20240517)
    phi = rng.uniform(-40.0, 40.0, size=(37, 53))
    assert np.allclose(wrap_phase(phi), np.angle(np.exp(1j * phi)))
    # Idempotent, and always inside the half-open interval (-pi, pi].
    assert np.allclose(wrap_phase(wrap_phase(phi)), wrap_phase(phi))
    assert np.all(wrap_phase(phi) >= -np.pi)
    assert np.all(wrap_phase(phi) <= np.pi)


def test_wrap_phase_covers_both_endpoints_of_the_interval():
    """The reachable range is [-pi, pi]; both ends are attained.

    Why both, rather than a tidy half-open interval?  Because the defining
    identity of this package is ``wrap_phase(phi) == angle(exp(1j * phi))``
    evaluated exactly, and ``angle`` maps ``-pi`` to ``-pi``.  Remapping
    ``-pi`` onto ``+pi`` would break that identity by one full cycle; a
    benchmark that is scored against the *integer* ambiguity cannot afford it.
    """
    assert np.isclose(abs(wrap_phase(np.array([-np.pi, 3.0 * np.pi]))).max(),
                      np.pi, rtol=0, atol=1e-15)
    assert np.allclose(wrap_phase(np.array([-np.pi, 3.0 * np.pi])),
                       np.angle(np.exp(1j * np.array([-np.pi, 3.0 * np.pi]))))


def test_integer_ambiguity_reconstructs_the_true_phase_exactly():
    """This is the single most important identity in the package."""
    rng = np.random.default_rng(7)
    phi_true = rng.uniform(-60.0, 60.0, size=(41, 29))
    phi_wrapped = wrap_phase(phi_true)
    k = integer_ambiguity(phi_true, phi_wrapped)

    assert np.array_equal(k, np.round(k))
    assert np.allclose(phi_wrapped + 2.0 * np.pi * k, phi_true, atol=1e-9)
    assert np.allclose(unwrap_with_ambiguity(phi_wrapped, k), phi_true, atol=1e-9)


def test_integer_ambiguity_rounds_half_away_from_zero():
    """Banker's rounding would make the result depend on the NumPy version."""
    phi_true = np.array([np.pi, -np.pi, 3.0 * np.pi, -3.0 * np.pi])
    phi_wrapped = np.array([np.pi, np.pi, np.pi, np.pi])
    assert integer_ambiguity(phi_true, phi_wrapped).tolist() == [0, -1, 1, -2]


def test_ambiguity_error_reports_disagreement_in_cycles():
    truth = np.zeros((3, 3), dtype=int)
    truth[0, 0] = 2
    estimate = truth.copy()
    estimate[0, 0] = -1
    error = ambiguity_error(estimate, truth)
    assert np.array_equal(error, estimate - truth)
    assert int(np.abs(error).max()) == 3


def test_ambiguity_error_respects_the_mask():
    truth = np.zeros((2, 2), dtype=int)
    estimate = np.ones((2, 2), dtype=int)
    mask = np.array([[True, False], [True, False]])
    error = ambiguity_error(estimate, truth, mask=mask)
    assert np.array_equal(error, np.array([[1, 0], [1, 0]]))


# ---------------------------------------------------------------------------
# Zero deformation, line of sight, displacement to phase
# ---------------------------------------------------------------------------

def test_zero_deformation_produces_zero_phase():
    zeros = np.zeros((11, 13))
    phase = displacement_to_phase(project_to_los(zeros, zeros, zeros, 33.0, 100.0),
                                  SENTINEL_1)
    assert np.array_equal(phase, np.zeros((11, 13)))


def test_los_unit_vector_has_unit_norm():
    for incidence in (20.0, 34.0, 45.0, 60.0):
        for azimuth in (-180.0, -77.0, 0.0, 90.0, 257.4):
            vector = los_unit_vector(incidence, azimuth)
            assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-12)


def test_los_projection_matches_the_analytic_vertical_case():
    """For a purely vertical displacement, d_LOS = u_U cos(incidence)."""
    u_u = 0.01
    zeros = np.zeros((5, 5))
    d_los = project_to_los(zeros, zeros, np.full((5, 5), u_u), 30.0, 100.0)
    assert np.allclose(d_los, u_u * np.cos(np.radians(30.0)), atol=1e-14)


def test_los_projection_matches_the_analytic_east_west_case():
    """For a purely eastward displacement, d_LOS = u_E * l_E."""
    u_e = 0.02
    zeros = np.zeros((4, 4))
    for azimuth in (270.0, 90.0, 180.0, 0.0):
        l_e, _, l_u = los_unit_vector(34.0, azimuth)
        d_los = project_to_los(np.full((4, 4), u_e), zeros, zeros, 34.0,
                               look_azimuth_deg=azimuth)
        assert np.allclose(d_los, u_e * l_e, atol=1e-14)
        assert np.allclose(l_u, np.cos(np.radians(34.0)), atol=1e-12)


def test_los_sign_knob_negates_the_projection():
    u = np.random.default_rng(3).normal(scale=0.01, size=(6, 6))
    up = project_to_los(u, u, u, 39.0, 100.0)
    down = project_to_los(u, u, u, 39.0, 100.0, los_sign=-1.0)
    assert np.allclose(up, -down)


def test_displacement_to_phase_round_trip():
    rng = np.random.default_rng(11)
    d_los = rng.normal(scale=0.02, size=(9, 9))
    phase = displacement_to_phase(d_los, SENTINEL_1)
    assert np.allclose(phase_to_displacement(phase, SENTINEL_1), d_los, atol=1e-15)


def test_phase_is_proportional_to_one_over_wavelength():
    """A C-band sensor is far more sensitive than an L-band one."""
    d_los = 0.01
    c_band = displacement_to_phase(d_los, wavelength_for("sentinel1"))
    l_band = displacement_to_phase(d_los, wavelength_for("alos"))
    assert np.isclose(c_band / l_band,
                      wavelength_for("alos") / SENTINEL_1, rtol=1e-12)
    assert 4.2 < c_band / l_band < 4.3


def test_fringe_per_metre_is_inverse_of_half_wavelength():
    """A fringe is half a wavelength, so there are 2 / lambda fringes per metre."""
    assert np.isclose(fringe_per_metre(SENTINEL_1), 2.0 / SENTINEL_1, rtol=1e-12)


def test_sar_geometry_agrees_with_the_functional_interface():
    geometry = SarGeometry(SENTINEL_1, 33.0, look_azimuth_deg=257.4)
    rng = np.random.default_rng(5)
    u_e = rng.normal(scale=0.01, size=(7, 8))
    u_n = rng.normal(scale=0.01, size=(7, 8))
    u_u = rng.normal(scale=0.01, size=(7, 8))
    d_los = project_to_los(u_e, u_n, u_u, 33.0, look_azimuth_deg=257.4)
    assert np.allclose(geometry.project(u_e, u_n, u_u), d_los)
    assert np.allclose(geometry.phase(d_los),
                       displacement_to_phase(d_los, SENTINEL_1))
    assert np.allclose(geometry.phase_from_displacement(u_e, u_n, u_u),
                       displacement_to_phase(d_los, SENTINEL_1))


# ---------------------------------------------------------------------------
# Grid conventions
# ---------------------------------------------------------------------------

def test_grid_row_zero_is_the_northernmost_row():
    grid = make_grid(4, 3, 10.0, x_min=100.0, y_max=50.0)
    assert grid.y[0] == 50.0
    assert grid.y[-1] == 30.0
    assert grid.y[0] == grid.y.max()
    assert np.all(np.diff(grid.y) < 0)
    assert np.all(np.diff(grid.x) > 0)
    assert grid.x[0] == 100.0


def test_centered_grid_has_zero_mean_pixel_centre():
    grid = make_grid(64, 32, 25.0, centered=True)
    assert np.isclose(grid.X.mean(), 0.0, atol=1e-9)
    assert np.isclose(grid.Y.mean(), 0.0, atol=1e-9)
    assert grid.shape == (32, 64)


def test_grid_from_bounds_hits_the_requested_pixel_centres():
    """from_bounds takes first *and last* centre, so it catches off-by-ones."""
    grid = Grid.from_bounds(41, 21, -1000.0, 1000.0, 500.0, -500.0)
    assert (grid.nx, grid.ny) == (41, 21)
    assert np.isclose(grid.spacing, 50.0)
    assert np.isclose(grid.x[0], -1000.0) and np.isclose(grid.x[-1], 1000.0)
    assert np.isclose(grid.y[0], 500.0) and np.isclose(grid.y[-1], -500.0)


def test_grid_from_bounds_rejects_inconsistent_spacing():
    """Asking for 512 pixels across 512 spacings is an off-by-one error."""
    with pytest.raises(ValueError):
        Grid.from_bounds(41, 21, -1000.0, 1000.0, 500.0, -400.0)


def test_grid_rejects_non_positive_spacing():
    with pytest.raises(ValueError):
        make_grid(10, 10, 0.0)


# ---------------------------------------------------------------------------
# Mogi
# ---------------------------------------------------------------------------

def test_mogi_peak_uplift_has_the_analytic_value():
    """u_U at the source is (1 - nu) dV / (pi d^2)."""
    depth = 5000.0
    delta_volume = 1.0e6
    poisson = 0.25
    u_e, u_n, u_u = mogi_point_displacement(
        0.0, 0.0, 0.0, 0.0, depth, delta_volume, poisson)
    expected = (1.0 - poisson) * delta_volume / (np.pi * depth ** 2)
    assert np.isclose(float(u_u), expected, rtol=1e-12)
    assert np.isclose(float(u_e), 0.0, atol=1e-15)
    assert np.isclose(float(u_n), 0.0, atol=1e-15)


def test_mogi_vertical_displacement_is_radially_symmetric():
    depth, delta_volume = 4000.0, 5.0e5
    grid = make_grid(101, 101, 100.0, centered=True)
    _, _, u_u = mogi_displacement(grid, 0.0, 0.0, depth, delta_volume)
    centre = grid.nearest_row(0.0), grid.nearest_column(0.0)
    assert np.isclose(u_u[centre], u_u[:, :].max(), rtol=1e-12)
    # Left/right and up/down mirror symmetry about the source.
    assert np.allclose(u_u, u_u[:, ::-1], atol=1e-15)
    assert np.allclose(u_u, u_u[::-1, :], atol=1e-15)


def test_mogi_horizontal_displacement_is_odd_in_the_offset():
    """u_E changes sign when the observation crosses the source in x."""
    depth, delta_volume = 6000.0, 2.0e6
    grid = make_grid(81, 81, 200.0, centered=True)
    u_e, u_n, _ = mogi_displacement(grid, 0.0, 0.0, depth, delta_volume)
    assert np.allclose(u_e, -u_e[:, ::-1], atol=1e-14)
    assert np.allclose(u_n, -u_n[::-1, :], atol=1e-14)


def test_mogi_displacement_decays_with_distance():
    depth, delta_volume = 5000.0, 1.0e6
    offsets = np.array([1.0e4, 2.0e4, 4.0e4, 8.0e4])
    _, _, u_u = mogi_point_displacement(
        offsets, np.zeros_like(offsets), 0.0, 0.0, depth, delta_volume)
    assert np.all(np.diff(np.abs(u_u)) < 0.0)
    # u_U = (1 - nu) dV d / (pi R^3), so far from the source (R ~ r >> d) the
    # uplift falls off as 1 / r^3: doubling the range divides it by eight.
    ratio = np.abs(u_u[3]) / np.abs(u_u[2])
    assert np.isclose(ratio, 0.125, rtol=0.05)


def test_mogi_inflation_and_deflation_have_opposite_signs():
    depth, delta_volume = 5000.0, 1.0e6
    _, _, up = mogi_point_displacement(0.0, 0.0, 0.0, 0.0, depth, delta_volume)
    _, _, down = mogi_point_displacement(0.0, 0.0, 0.0, 0.0, depth, -delta_volume)
    assert np.isclose(float(up), -float(down), rtol=1e-14)


def test_mogi_multiple_sources_superpose_linearly():
    depths = [4000.0, 7000.0, 3000.0]
    volumes = [1.0e6, -4.0e5, 2.5e5]
    positions = [(-3000.0, 2000.0), (5000.0, -4000.0), (1000.0, 6000.0)]
    grid = make_grid(64, 48, 250.0, centered=True)

    sources = [dict(source_x=x, source_y=y, depth=d, delta_volume=v)
               for (x, y), d, v in zip(positions, depths, volumes)]
    total = mogi_displacement_multi(grid, sources)

    expected = [np.zeros(grid.shape) for _ in range(3)]
    for x, y, d, v in zip([p[0] for p in positions],
                          [p[1] for p in positions], depths, volumes):
        for index, component in enumerate(mogi_displacement(grid, x, y, d, v)):
            expected[index] += component

    for got, want in zip(total, expected):
        assert np.allclose(got, want, atol=1e-15)


def test_mogi_rejects_a_singular_or_unphysical_source():
    grid = make_grid(8, 8, 100.0)
    with pytest.raises(ValueError):
        mogi_displacement(grid, 0.0, 0.0, 0.0, 1.0e6)
    with pytest.raises(ValueError):
        mogi_displacement(grid, 0.0, 0.0, -100.0, 1.0e6)
    with pytest.raises(ValueError):
        mogi_displacement(grid, 0.0, 0.0, 5000.0, 1.0e6, poisson_ratio=0.5)


def test_random_mogi_source_is_reproducible_for_a_fixed_seed():
    grid = make_grid(32, 32, 200.0, centered=True)
    first = random_mogi_source(grid, "strong", rng=np.random.default_rng(1234))
    second = random_mogi_source(grid, "strong", rng=np.random.default_rng(1234))
    assert first == second
    third = random_mogi_source(grid, "strong", rng=np.random.default_rng(4321))
    assert third != first


# ---------------------------------------------------------------------------
# Residues
# ---------------------------------------------------------------------------

def test_residues_vanish_on_a_noise_free_ramp():
    y, x = np.mgrid[0:12, 0:9]
    phi = wrap_phase(0.31 * x + 0.17 * y)
    assert int(np.abs(residue_map(phi)).sum()) == 0
    assert np.isclose(boundary_circulation(phi), 0.0, atol=1e-12)


def test_residue_map_finds_a_single_charge_for_a_vortex():
    y, x = np.mgrid[-2:3, -2:3] * 1.0
    residues = residue_map(wrap_phase(np.arctan2(y, x)))
    assert int(residues.sum()) == -1
    assert int(np.abs(residues).sum()) == 1
    assert int(positive_residues(wrap_phase(np.arctan2(y, x))).sum()) == 0
    assert int(negative_residues(wrap_phase(np.arctan2(y, x))).sum()) == 1


def test_residues_are_computed_from_wrapped_gradients():
    """Recompute the circulation by hand from the wrapped gradients."""
    rng = np.random.default_rng(99)
    phi = wrap_phase(rng.uniform(-3.0, 3.0, size=(17, 23)))
    d_east, d_north = wrapped_gradients(phi)
    residues = residue_map(phi)

    # Edge A->B is d_east on the southern row, B->C is d_north on the eastern
    # column, C->D is d_east reversed on the northern row, D->A is d_north
    # reversed on the western column.
    circulation = (d_east[1:, :] - d_east[:-1, :]
                   + d_north[:, 1:] - d_north[:, :-1])
    manual = np.copysign(np.floor(np.abs(circulation / (2.0 * np.pi)) + 0.5),
                         circulation / (2.0 * np.pi)).astype(np.int8)
    assert np.array_equal(residues, manual)


def test_residue_sum_equals_the_boundary_circulation():
    """The divergence theorem on the plaquette lattice, checked numerically."""
    for seed in (0, 3, 7, 12345):
        rng = np.random.default_rng(seed)
        phi = wrap_phase(rng.normal(scale=2.0, size=(17, 23)))
        total = int(residue_map(phi).sum())
        boundary = int(round(boundary_circulation(phi) / (2.0 * np.pi)))
        assert total == boundary


def test_residue_balance_reports_a_dipole():
    y, x = np.mgrid[-2:3, -2:3] * 1.0
    phi = wrap_phase(np.arctan2(y, x) - np.arctan2(y, x - 2.0))
    charges = residue_map(phi)
    balance = residue_balance(charges)
    assert balance["positive"] == 1
    assert balance["negative"] == 1
    assert balance["total"] == 0
    assert balance["absolute"] == 2
    assert np.isclose(residue_density(charges), 2.0 / 16.0)


def test_residues_are_suppressed_by_the_mask():
    y, x = np.mgrid[-2:3, -2:3] * 1.0
    phi = wrap_phase(np.arctan2(y, x))
    mask = np.ones(phi.shape, dtype=bool)
    mask[2, 2] = False  # covers the vortex core and three of its blocks
    masked = residue_map(phi, mask=mask)
    assert int(np.abs(masked).sum()) == 0
    assert int(np.abs(residue_map(phi)).sum()) == 1


def test_residue_map_shape_and_dtype():
    residues = residue_map(np.zeros((6, 9)))
    assert residues.shape == (5, 8)
    assert residues.dtype == np.int8


# ---------------------------------------------------------------------------
# Aliasing
# ---------------------------------------------------------------------------

def test_aliasing_mask_flags_a_steep_gradient_and_not_a_gentle_one():
    y, x = np.mgrid[0:8, 0:8]
    assert not aliasing_mask(0.2 * x + 0.1 * y).any()
    assert aliasing_mask(3.5 * x).any()


def test_aliasing_is_invisible_in_the_wrapped_phase():
    """The reason aliasing_mask needs the continuous phase at all."""
    y, x = np.mgrid[0:6, 0:6] * 1.0
    phi = 4.1 * x + 0.2 * y
    d_east_true = phase_gradients(phi)[0]
    d_east_wrapped = wrapped_gradients(wrap_phase(phi))[0]
    assert np.all(np.abs(d_east_true) > np.pi)
    assert np.all(np.abs(d_east_wrapped) <= np.pi)
    assert not np.allclose(np.sign(d_east_true), np.sign(d_east_wrapped))
    assert aliasing_mask(phi).all()


def test_aliasing_mask_shrinks_as_the_threshold_grows():
    phi = 3.5 * np.mgrid[0:9, 0:9][1]
    coarse = aliasing_mask(phi, threshold=np.pi)
    fine = aliasing_mask(phi, threshold=3.6)
    assert coarse.sum() >= fine.sum()
    assert not fine.any()


def test_aliasing_mask_validates_its_input():
    with pytest.raises(ValueError):
        aliasing_mask(np.zeros(5))
    with pytest.raises(ValueError):
        aliasing_mask(np.zeros((4, 4)), threshold=0.0)
    with pytest.raises(ValueError):
        phase_gradients(np.zeros((4, 4, 2)))
