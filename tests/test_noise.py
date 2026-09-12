"""Numerical validation of the InSAR phase-noise model (Phase D).

The point of these tests is that the noise in the generator is *physics*, not an
arbitrary Gaussian.  The single-look phase of an interferogram built from two
correlated circular complex Gaussian reflectivities has a known closed-form
probability density, and the simulator draws from it by construction.  If the
simulator and the density ever stop agreeing, one of them is wrong, and these
tests say so.

Three independent checks are made:

* the simulator's histogram matches ``single_look_phase_pdf`` bin by bin;
* the simulator's standard deviation matches the density's exact standard
  deviation, which is computed here by quadrature, not assumed;
* the standard many-look approximation ``phase_noise_std`` converges to the
  simulation as looks are accumulated, and -- importantly -- is *not* trusted
  at ``looks=1``, where it is known to be poor.
"""

import numpy as np
import pytest

from parvaneh.synthetic import (
    DEFAULT_LOOKS,
    UNIFORM_PHASE_STD,
    add_phase_noise,
    phase_noise,
    phase_noise_std,
    single_look_phase_pdf,
)


#: Exact single-look standard deviations, in radians, obtained by integrating
#: ``single_look_phase_pdf`` on a log-spaced grid over ``(-pi, pi)``.  These are
#: hard-coded so that the test is a genuine check and not a restatement of the
#: implementation.
EXACT_SINGLE_LOOK_STD = (
    (0.00, 1.813799),
    (0.20, 1.636345),
    (0.50, 1.336138),
    (0.80, 0.917359),
    (0.95, 0.519849),
)


def _integrate_density(coherence, exponent=0):
    """Integrate ``psi**exponent * pdf(psi)`` over ``(-pi, pi)`` on a log grid.

    The density is symmetric in ``psi``, so only the positive half is sampled
    and the result is doubled.  A log-spaced grid is essential: for high
    coherence the density is a narrow peak at zero that a uniform grid would
    straddle, and ``scipy.integrate.quad`` misses it even with a breakpoint.
    """
    u = np.linspace(-40.0, np.log(np.pi), 500001)
    psi = np.exp(u)
    step = np.diff(u)
    mid = 0.5 * (psi[1:] + psi[:-1])
    density = np.asarray(single_look_phase_pdf(mid, coherence))
    return 2.0 * float(np.sum(density * mid ** exponent * mid * step))


# ---------------------------------------------------------------------------
# The single-look probability density
# ---------------------------------------------------------------------------


def test_single_look_density_is_a_normalised_probability_density():
    """It must be non-negative everywhere and integrate to exactly one.

    An earlier form of this density in the literature uses a cube in the
    denominator; that version goes negative at high coherence and integrates to
    about 5.5, which is how the mistake here was caught.
    """
    psi = np.linspace(-np.pi, np.pi, 200001)
    for coherence in (0.0, 0.2, 0.5, 0.8, 0.95):
        density = np.asarray(single_look_phase_pdf(psi, coherence))
        assert density.min() >= 0.0
        assert np.trapz(density, psi) == pytest.approx(1.0, abs=1e-6)


def test_single_look_density_is_uniform_when_coherence_is_zero():
    """With no correlation there is no phase information, so phase is uniform."""
    psi = np.linspace(-np.pi, np.pi, 101)
    density = np.asarray(single_look_phase_pdf(psi, 0.0))
    assert np.allclose(density, 1.0 / (2.0 * np.pi))
    assert UNIFORM_PHASE_STD == pytest.approx(np.pi / np.sqrt(3.0))


def test_single_look_density_peaks_at_zero_and_falls_away():
    """The most likely phase error is zero, and the tails are much smaller."""
    for coherence in (0.2, 0.5, 0.8, 0.95):
        at_zero = float(single_look_phase_pdf(0.0, coherence))
        at_edge = float(single_look_phase_pdf(np.pi, coherence))
        assert at_zero > at_edge
        assert at_edge >= 0.0


def test_single_look_density_rejects_full_coherence():
    """Coherence one is a Dirac delta and cannot be returned as an array."""
    with pytest.raises(ValueError):
        single_look_phase_pdf(0.0, 1.0)
    with pytest.raises(ValueError):
        single_look_phase_pdf(0.0, 1.5)
    with pytest.raises(ValueError):
        single_look_phase_pdf(0.0, -0.1)


def test_single_look_density_standard_deviation_matches_exact_values():
    """Integrate the density and compare with values computed independently."""
    for coherence, expected in EXACT_SINGLE_LOOK_STD:
        variance = _integrate_density(coherence, exponent=2)
        assert np.sqrt(variance) == pytest.approx(expected, abs=2e-6)


# ---------------------------------------------------------------------------
# The simulator
# ---------------------------------------------------------------------------


def test_simulated_phase_is_inside_the_principal_interval():
    """Whatever the coherence, the returned phase is a wrapped phase."""
    rng = np.random.default_rng(11)
    for coherence in (0.0, 0.3, 0.9):
        psi = phase_noise(np.full((64, 64), coherence), rng=rng)
        assert np.all(psi >= -np.pi)
        assert np.all(psi <= np.pi)


def test_simulator_histogram_matches_the_single_look_density():
    """This is the central validation of the module.

    Several million single-look samples are binned and compared with the
    closed-form density, which was derived along a completely different route:
    the simulation draws complex Gaussians, the density integrates the
    four-dimensional complex Gaussian exactly.
    """
    rng = np.random.default_rng(20240101)
    edges = np.linspace(-np.pi, np.pi, 41)
    width = float(edges[1] - edges[0])
    mid = 0.5 * (edges[1:] + edges[:-1])
    for coherence in (0.0, 0.2, 0.5, 0.8):
        samples = phase_noise(np.full(400000, coherence), rng=rng)
        histogram, _ = np.histogram(samples, bins=edges, density=True)
        expected = np.asarray(single_look_phase_pdf(mid, coherence))
        total_variation = float(np.abs(histogram - expected).sum() * width)
        assert total_variation < 0.02


def test_simulator_standard_deviation_matches_the_exact_density():
    """The scatter of the simulation equals the density's exact scatter."""
    rng = np.random.default_rng(4242)
    for coherence, expected in EXACT_SINGLE_LOOK_STD:
        samples = phase_noise(np.full(400000, coherence), rng=rng)
        assert float(samples.std()) == pytest.approx(expected, abs=0.02)


def test_perfect_coherence_returns_exactly_zero_phase_error():
    """No decorrelation means no phase noise, exactly and not approximately."""
    rng = np.random.default_rng(5)
    assert float(phase_noise(1.0, rng=rng)) == 0.0
    assert not np.any(phase_noise(np.ones((32, 32)), rng=rng))


def test_no_data_pixels_stay_no_data():
    """A coherence of ``nan`` means "not measured" and must propagate."""
    coherence = np.full((16, 16), 0.8)
    coherence[:, 4] = np.nan
    psi = phase_noise(coherence, rng=np.random.default_rng(3))
    assert np.all(np.isnan(psi[:, 4]))
    assert np.all(np.isfinite(psi[:, 0]))
    assert np.allclose(np.isnan(psi), np.isnan(coherence))


def test_simulator_is_reproducible_for_a_fixed_seed():
    """Two generators with the same seed must agree bit for bit."""
    coherence = np.full((48, 48), 0.7)
    first = phase_noise(coherence, looks=4, rng=np.random.default_rng(99))
    second = phase_noise(coherence, looks=4, rng=np.random.default_rng(99))
    assert np.array_equal(first, second)
    third = phase_noise(coherence, looks=4, rng=np.random.default_rng(100))
    assert not np.array_equal(first, third)


def test_more_looks_reduce_the_phase_scatter():
    """Averaging looks is the standard cure for low coherence."""
    rng = np.random.default_rng(17)
    coherence = np.full(400000, 0.6)
    previous = None
    for looks in (1, 4, 16, 64):
        sigma = float(phase_noise(coherence, looks=looks, rng=rng).std())
        if previous is not None:
            assert sigma < previous
        previous = sigma


def test_looks_must_be_a_positive_integer():
    """A non-integer or non-positive look count is a caller error."""
    with pytest.raises(ValueError):
        phase_noise(0.8, looks=0)
    with pytest.raises(ValueError):
        phase_noise(0.8, looks=2.5)


def test_coherence_outside_the_unit_interval_is_rejected():
    """Coherence is a correlation coefficient and cannot leave ``[0, 1]``."""
    with pytest.raises(ValueError):
        phase_noise(1.5)
    with pytest.raises(ValueError):
        phase_noise(-0.2)
    with pytest.raises(ValueError):
        phase_noise(np.full((4, 4), 1.2))


# ---------------------------------------------------------------------------
# Adding noise to a phase field
# ---------------------------------------------------------------------------


def test_add_phase_noise_is_exactly_phase_noise_added_to_the_truth():
    """The two functions must stay strictly in step for the same seed."""
    truth = np.arange(9, dtype=float).reshape(3, 3)
    noisy = add_phase_noise(truth, 0.9, rng=np.random.default_rng(7))
    psi = phase_noise(np.full((3, 3), 0.9), rng=np.random.default_rng(7))
    assert np.array_equal(noisy, truth + psi)


def test_add_phase_noise_leaves_the_phase_unchanged_at_full_coherence():
    """Perfect coherence must be a no-op, otherwise the clean case is untestable."""
    truth = np.linspace(-3.0, 3.0, 24).reshape(4, 6)
    noisy = add_phase_noise(truth, 1.0, rng=np.random.default_rng(1))
    assert np.array_equal(noisy, truth)


def test_add_phase_noise_preserves_a_perfectly_coherent_region():
    """Coherence may vary in space; the good patch must survive untouched."""
    truth = np.zeros((20, 20))
    coherence = np.full((20, 20), 0.4)
    coherence[:5, :5] = 1.0
    noisy = add_phase_noise(truth, coherence, rng=np.random.default_rng(2))
    assert np.array_equal(noisy[:5, :5], truth[:5, :5])
    assert float(noisy[10:, 10:].std()) > 0.0


def test_add_phase_noise_requires_matching_shapes():
    """A coherence map that does not match the phase field is a caller error."""
    with pytest.raises(ValueError):
        add_phase_noise(np.zeros((8, 8)), np.full((4, 4), 0.5))


# ---------------------------------------------------------------------------
# The many-look approximation
# ---------------------------------------------------------------------------


def test_phase_noise_std_is_clipped_at_the_uniform_value():
    """The raw formula diverges at zero coherence; the clip keeps it physical."""
    assert float(phase_noise_std(0.0)) == pytest.approx(UNIFORM_PHASE_STD)
    assert float(phase_noise_std(0.0)) < np.inf
    assert float(phase_noise_std(0.2)) == pytest.approx(UNIFORM_PHASE_STD)


def test_phase_noise_std_decreases_with_coherence_and_with_looks():
    """Both knobs must push the prediction in the expected direction."""
    values = [float(phase_noise_std(g)) for g in (0.2, 0.4, 0.6, 0.8, 0.95)]
    assert values == sorted(values, reverse=True)
    assert float(phase_noise_std(0.5, looks=4)) < float(phase_noise_std(0.5))
    assert float(phase_noise_std(0.95, looks=4)) < float(phase_noise_std(0.95))


def test_phase_noise_std_approaches_the_simulation_for_many_looks():
    """The formula is a large-looks asymptotic, so it must converge there.

    This is the honest statement of what the formula is good for.  At one look
    it is poor (see the next test), so a benchmark must never use it to decide
    that the simulation is wrong.
    """
    rng = np.random.default_rng(31)
    for coherence in (0.6, 0.8, 0.95):
        samples = phase_noise(np.full(400000, coherence), looks=64, rng=rng)
        predicted = float(phase_noise_std(coherence, looks=64))
        assert float(samples.std()) == pytest.approx(predicted, rel=0.05)


def test_phase_noise_std_is_poor_for_a_single_look():
    """Documenting a trap: the formula underestimates single-look scatter.

    At coherence 0.95 it is too small by more than a factor of two.  Asserting
    the size of the discrepancy pins it down, so that the module documentation
    cannot quietly drift back to claiming that the formula is accurate here.
    """
    rng = np.random.default_rng(53)
    for coherence in (0.6, 0.8, 0.95):
        samples = phase_noise(np.full(400000, coherence), looks=1, rng=rng)
        ratio = float(samples.std()) / float(phase_noise_std(coherence))
        assert ratio > 1.2
    samples = phase_noise(np.full(400000, 0.95), looks=1, rng=rng)
    assert float(samples.std()) / float(phase_noise_std(0.95)) > 2.0


def test_default_looks_is_one():
    """The default must be the honest full-resolution case."""
    assert DEFAULT_LOOKS == 1
