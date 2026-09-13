"""Tests for reliability-sorting unwrapping in one, two and three dimensions.

The method is the one published by Herraez and co-workers in 2002: rate every
pixel, list every pair of neighbours, and grow one tree per region from the
most reliable pairs downwards.  The tests below pin the rating formula, the
bookkeeping of the tree, the exactness on clean data, the phase congruence on
noisy data, and the one claim this family exists for: a third axis lets the
tree route around a noisy pixel.
"""

import numpy as np
import pytest

from parvaneh import (available_backends, make_synthetic, pixel_reliability,
                      reliability_unwrap, rmse_aligned, wrap_phase)


HAS_NUMBA = bool(available_backends().get("numba", False))
NEEDS_NUMBA = pytest.mark.skipif(not HAS_NUMBA, reason="numba is not installed")


def ramp_volume(rows=24, cols=31, steps=6):
    """A smooth noise-free volume: a ramp in every axis plus a bump."""
    y, x = np.mgrid[0:rows, 0:cols].astype(float)
    bump = 10.0 * np.exp(-(((x - (cols - 1) / 2.0) / 7.0) ** 2
                           + ((y - (rows - 1) / 2.0) / 7.0) ** 2))
    drift = 0.15 * np.arange(steps)
    return ((0.12 * x + 0.05 * y)[:, :, None] + bump[:, :, None]
            + drift[None, None, :])


def steep_volume(sigma=0.45, seed=1, rows=48, cols=48, steps=8):
    """A volume with steep fringes and noise, plus the truth it came from.

    This is the experiment of the three-dimensional notebook: the Gaussian
    bump is steep enough to wrap, and the noise is strong enough to put a few
    whole-fringe mistakes into a slice that is unwrapped on its own.
    """
    y, x = np.mgrid[0:rows, 0:cols].astype(float)
    bump = 10.0 * np.exp(-(((x - (cols - 1) / 2.0) / 7.0) ** 2
                           + ((y - (rows - 1) / 2.0) / 7.0) ** 2))
    drift = 0.15 * np.arange(steps)
    truth = ((0.12 * x + 0.05 * y)[:, :, None] + bump[:, :, None]
             + drift[None, None, :])
    rng = np.random.default_rng(seed)
    return truth, wrap_phase(truth + rng.normal(scale=sigma, size=truth.shape))


def wrong_fraction(estimate, truth):
    """Fraction of pixels that ended a whole number of fringes away.

    A single constant offset is removed first, because every unwrapped field
    is only defined up to such an offset.
    """
    error = estimate - estimate.mean() - (truth - truth.mean())
    return float((np.abs(error) > np.pi).mean())


# --------------------------------------------------------------------------
# the per-pixel rating
# --------------------------------------------------------------------------

def test_rating_follows_the_definition_on_a_hand_built_profile():
    """On a line the rating is ``1 / (1 + sum of neighbouring steps)``."""
    phase = np.array([0.0, 0.3, 0.7, -3.1, -2.7, -2.3])
    steps = np.abs(wrap_phase(np.diff(phase)))
    expected = np.empty(phase.size)
    expected[0] = 1.0 / (1.0 + steps[0])
    expected[-1] = 1.0 / (1.0 + steps[-1])
    expected[1:-1] = 1.0 / (1.0 + steps[:-1] + steps[1:])
    assert np.allclose(pixel_reliability(phase), expected)


def test_rating_separates_a_smooth_surface_from_a_noisy_one():
    """The worst pixel of a smooth ramp beats the best pixel of a noisy one."""
    y, x = np.mgrid[0:40, 0:50].astype(float)
    surface = wrap_phase(0.1 * x + 0.05 * y)
    rng = np.random.default_rng(0)
    noisy = wrap_phase(surface + rng.normal(scale=1.0, size=surface.shape))

    smooth_rating = pixel_reliability(surface)
    noisy_rating = pixel_reliability(noisy)

    # An interior pixel of this ramp sees two steps of 0.1 and two of 0.05.
    assert np.allclose(smooth_rating[1:-1, 1:-1], 1.0 / 1.3)
    assert noisy_rating.max() < smooth_rating.min()
    assert noisy_rating.mean() < 0.5 * smooth_rating.mean()
    assert np.isfinite(noisy_rating).all()


def test_rating_scales_with_the_confidence_weights():
    """The weights multiply the step sizes, so halving them raises the rating."""
    _, wrapped, _ = make_synthetic((24, 31), noise=0.01)
    unit = pixel_reliability(wrapped, weight=np.ones(wrapped.shape))
    half = pixel_reliability(wrapped, weight=0.5 * np.ones(wrapped.shape))
    assert np.all(half > unit)


def test_rating_hides_pixels_without_a_usable_value():
    _, wrapped, weight = make_synthetic((17, 23), noise=0.01)
    weight = np.array(weight, dtype=float)
    weight[6, 7] = 0.0
    mask = np.ones(wrapped.shape, dtype=bool)
    mask[4, 5] = False

    rating = pixel_reliability(wrapped, weight=weight, mask=mask)

    assert np.isnan(rating[4, 5])
    assert np.isnan(rating[6, 7])
    assert np.isfinite(rating).sum() == wrapped.size - 2


def test_rating_rejects_bad_input():
    phase = np.zeros((6, 7))
    with pytest.raises(ValueError, match="match phase.shape"):
        pixel_reliability(phase, weight=np.ones((6, 6)))
    with pytest.raises(ValueError, match="match phase.shape"):
        pixel_reliability(phase, mask=np.ones((6, 6), dtype=bool))
    with pytest.raises(ValueError, match="nonnegative"):
        pixel_reliability(phase, weight=-np.ones((6, 7)))
    with pytest.raises(ValueError, match="finite"):
        pixel_reliability(phase, weight=np.full((6, 7), np.nan))
    with pytest.raises(ValueError, match="finite"):
        pixel_reliability(np.array([0.0, np.inf]))


# --------------------------------------------------------------------------
# unwrapping
# --------------------------------------------------------------------------

def test_one_dimensional_recovery():
    truth = 0.3 * np.arange(50)
    result = reliability_unwrap(wrap_phase(truth))
    assert np.allclose(result - result[0], truth - truth[0], atol=1e-12)


def test_three_dimensional_recovery():
    """A clean volume is recovered exactly, in all three axes at once."""
    truth = ramp_volume()
    result = reliability_unwrap(wrap_phase(truth))
    assert result.shape == truth.shape
    assert rmse_aligned(result, truth) < 1e-12


def test_three_dimensional_result_is_phase_congruent():
    """Whatever the noise, the answer keeps the wrapped values of the input."""
    truth, wrapped = steep_volume(sigma=0.3, rows=24, cols=31, steps=5)
    result = reliability_unwrap(wrapped, backend="python")

    assert np.isfinite(result).all()
    assert np.abs(wrap_phase(result) - wrapped).max() < 1e-9
    assert wrong_fraction(result, truth) < 0.02


def test_repeated_runs_give_identical_numbers():
    _, wrapped, _ = make_synthetic((31, 39), noise=0.05)
    assert np.array_equal(reliability_unwrap(wrapped),
                          reliability_unwrap(wrapped))


def test_the_used_pairs_form_a_spanning_tree():
    """One region needs ``pixels - 1`` pairs, whatever the residues do."""
    _, wrapped, _ = make_synthetic((31, 39), noise=0.05)
    rows, cols = wrapped.shape
    result, info = reliability_unwrap(wrapped, return_info=True)

    assert info.backend == "python"
    assert info.pixels == wrapped.size
    assert info.edges == rows * (cols - 1) + (rows - 1) * cols
    assert info.merges == info.pixels - 1
    assert info.components == 1
    assert info.discarded == info.edges - info.merges
    assert np.isfinite(result).all()


def test_a_masked_wall_separates_two_independent_regions():
    y, x = np.mgrid[0:24, 0:31].astype(float)
    truth = 0.3 * x + 0.1 * y
    mask = np.ones(truth.shape, dtype=bool)
    mask[:, 10] = False

    result, info = reliability_unwrap(wrap_phase(truth), mask=mask,
                                      return_info=True)

    assert info.components == 2
    assert info.merges == info.pixels - 2
    assert np.isnan(result[:, 10]).all()
    assert np.isfinite(result[:, :10]).all()
    assert np.isfinite(result[:, 11:]).all()
    # Each region has its own offset, so compare it with its own corner.
    for columns in (slice(0, 10), slice(11, truth.shape[1])):
        start = columns.start
        region = result[:, columns] - result[0, start]
        wanted = truth[:, columns] - truth[0, start]
        assert np.allclose(region, wanted, atol=1e-12)


def test_third_dimension_rescues_slices_that_fail_on_their_own():
    """The point of the 3-D family, measured on one deterministic volume.

    Unwrapping each slice separately makes whole-fringe mistakes, because a
    noisy pixel can look like the most reliable route across the steep bump.
    Unwrapping the volume at once lets the tree step sideways in time, where
    the phase barely changes, and every slice comes out right.
    """
    truth, wrapped = steep_volume(sigma=0.45, seed=1)
    joint = reliability_unwrap(wrapped, backend="python")
    per_slice = np.stack([reliability_unwrap(wrapped[:, :, k],
                                             backend="python")
                          for k in range(wrapped.shape[2])], axis=2)

    assert wrong_fraction(per_slice, truth) > 0.1
    assert wrong_fraction(joint, truth) < 0.01
    assert np.abs(wrap_phase(joint) - wrapped).max() < 1e-9


@NEEDS_NUMBA
@pytest.mark.parametrize("shape", [(31, 43), (12, 9, 4)])
def test_python_and_numba_agree_bit_for_bit(shape):
    """Both backends take the same merge decisions in the same order."""
    truth = np.zeros(shape)
    for coefficient, index in zip((0.2, 0.13, 0.4), np.indices(shape)):
        truth += coefficient * index
    rng = np.random.default_rng(4)
    wrapped = wrap_phase(truth + rng.normal(scale=0.3, size=shape))

    reference = reliability_unwrap(wrapped, backend="python")
    compiled = reliability_unwrap(wrapped, backend="numba")

    assert np.array_equal(reference, compiled)


def test_backend_name_is_checked():
    with pytest.raises(ValueError, match="'python' or 'numba'"):
        reliability_unwrap(np.zeros((5, 6)), backend="cupy")


def test_bad_shapes_and_nonfinite_phase_are_rejected():
    for bad in (np.array(5.0), np.zeros((1, 5)), np.zeros((5, 3, 1)),
                np.zeros((4, 4, 0))):
        with pytest.raises(ValueError, match="at least 2 samples"):
            reliability_unwrap(bad)
    phase = np.zeros((5, 6))
    phase[2, 3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        reliability_unwrap(phase)


def test_matches_the_scikit_image_baseline_on_a_smooth_surface():
    """scikit-image is the reference N-D baseline; clean data must agree."""
    skip = pytest.importorskip("skimage.restoration")
    y, x = np.mgrid[0:32, 0:41].astype(float)
    truth = 0.4 * x + 0.3 * y + 5.0 * np.exp(-(((x - 20) / 8.0) ** 2
                                              + ((y - 16) / 8.0) ** 2))
    wrapped = wrap_phase(truth)

    ours = reliability_unwrap(wrapped)
    theirs = skip.unwrap_phase(wrapped)

    assert np.abs((ours - ours.mean()) - (theirs - theirs.mean())).max() < 1e-6
