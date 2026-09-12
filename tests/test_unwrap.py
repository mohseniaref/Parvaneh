import numpy as np
import pytest
from parvaneh import (available_backends, discontinuity_map, flynn_unwrap,
                      goldstein_unwrap, make_synthetic, mask_cut_unwrap,
                      max_gradient_quality, phase_residues,
                      quality_guided_unwrap, read_raw_raster, rmse_aligned,
                      surface_difference, unwrap, unwrap_lp,
                      wrapped_gradients, write_raw_raster)


CPU_BACKENDS = [name for name, usable in available_backends().items()
                if usable and name != "cupy"]


@pytest.mark.parametrize("backend", CPU_BACKENDS)
def test_rectangular_synthetic(backend):
    truth, wrapped, weight = make_synthetic((48, 71), noise=0.001)
    result, info = unwrap(wrapped, weight, backend=backend, max_iter=80, return_info=True)
    assert result.shape == truth.shape
    assert np.isfinite(result).all()
    assert info.relative_residual < 1e-5
    assert rmse_aligned(result, truth) < 0.02


def test_unweighted_converges_in_one_iteration():
    truth, wrapped, _ = make_synthetic((40, 52), noise=0)
    result, info = unwrap(wrapped, backend="numpy", return_info=True)
    assert info.iterations == 1
    assert rmse_aligned(result, truth) < 1e-10


def test_parallel_dct_workers_preserve_result():
    _, wrapped, weight = make_synthetic((32, 43), noise=0.01)
    serial = unwrap(wrapped, weight, workers=1)
    parallel = unwrap(wrapped, weight, workers=-1)
    assert np.allclose(serial, parallel, atol=1e-11)


def test_bad_weight_rejected():
    with pytest.raises(ValueError):
        unwrap(np.zeros((4, 4)), -np.ones((4, 4)))


def test_nonfinite_phase_rejected():
    """A no-data pixel has no value to unwrap, so it is refused, not guessed."""
    nan_phase = np.zeros((5, 6))
    nan_phase[2, 3] = np.nan
    for bad in (nan_phase, np.full((5, 6), np.inf)):
        with pytest.raises(ValueError, match="finite"):
            unwrap(bad)


def test_nonfinite_phase_rejected_even_with_weight():
    """Zero weight does not rescue a NaN pixel: 0 * NaN is still NaN."""
    truth, wrapped, _ = make_synthetic((16, 21), noise=0.01)
    wrapped = np.array(wrapped)
    wrapped[7, 9] = np.nan
    valid = np.isfinite(wrapped)
    with pytest.raises(ValueError, match="finite"):
        unwrap(wrapped, valid.astype(float))


def test_masked_least_squares_ignores_a_hole():
    """The documented raster recipe: finite placeholder plus zero weight."""
    truth, wrapped, _ = make_synthetic((28, 34), noise=0.005)
    wrapped = np.array(wrapped)
    hole = np.zeros(wrapped.shape, dtype=bool)
    hole[10:14, 12:20] = True
    wrapped[hole] = np.nan
    valid = np.isfinite(wrapped)

    result = unwrap(np.where(valid, wrapped, 0.0), valid.astype(float))
    assert np.isfinite(result).all()
    error = np.abs((result - result.mean()) - (truth - truth.mean()))
    assert error[valid].max() < 0.05


def test_max_iter_must_be_positive():
    with pytest.raises(ValueError, match="max_iter"):
        unwrap(np.zeros((4, 4)), max_iter=0)


def test_exhausted_iterations_report_nonconvergence():
    """Hitting the iteration cap still returns a result and a finite residual."""
    _, wrapped, weight = make_synthetic((48, 64), noise=0.05)
    result, info = unwrap(wrapped, weight, max_iter=1, return_info=True)
    assert info.converged is False
    assert info.iterations == 1
    assert np.isfinite(info.relative_residual)
    assert np.isfinite(result).all()


def test_generic_raw_raster_round_trip(tmp_path):
    original = np.linspace(-np.pi, np.pi, 35).reshape(5, 7)
    path = write_raw_raster(tmp_path / "phase.f32", original, dtype="<f4")
    restored = read_raw_raster(path, original.shape, dtype="<f4")
    assert restored.shape == original.shape
    assert np.allclose(restored, original, atol=2e-7)


def test_generic_raw_raster_scale_and_size_validation(tmp_path):
    codes = np.arange(12, dtype=np.uint8).reshape(3, 4)
    path = write_raw_raster(tmp_path / "codes.u8", codes, dtype=np.uint8)
    restored = read_raw_raster(path, (3, 4), dtype=np.uint8,
                               scale=2 * np.pi / 256, offset=-np.pi)
    assert restored[0, 0] == pytest.approx(-np.pi)
    with pytest.raises(ValueError, match="expected"):
        read_raw_raster(path, (4, 4), dtype=np.uint8)


def test_residue_charge_on_synthetic_vortex():
    y, x = np.mgrid[-1:1:41j, -1:1:41j]
    wrapped = np.arctan2(y - 0.02, x - 0.03)
    residues = phase_residues(wrapped)
    assert np.count_nonzero(residues) == 1
    assert abs(int(residues.sum())) == 1


def test_quality_guided_unwrap_is_phase_congruent():
    truth, wrapped, weight = make_synthetic((43, 57), noise=0)
    result = quality_guided_unwrap(wrapped, quality=weight)
    cycles = (result - wrapped) / (2 * np.pi)
    assert np.max(np.abs(cycles - np.rint(cycles))) < 1e-12
    assert rmse_aligned(result, truth) < 1e-10
    rdx, rdy = wrapped_gradients(result)
    wdx, wdy = wrapped_gradients(wrapped)
    assert np.allclose(rdx, wdx)
    assert np.allclose(rdy, wdy)


def test_numba_quality_path_matches_python_reference():
    _, wrapped, _ = make_synthetic((43, 57), noise=0.01)
    python = quality_guided_unwrap(wrapped, "min_gradient", backend="python")
    compiled = quality_guided_unwrap(wrapped, "min_gradient", backend="numba")
    assert np.allclose(compiled, python, atol=1e-6)


def test_lp_two_matches_least_squares():
    _, wrapped, _ = make_synthetic((35, 47), noise=0.01)
    assert np.allclose(unwrap_lp(wrapped, p=2), unwrap(wrapped), atol=1e-11)


def test_lp_irls_is_finite_and_reports_iterations():
    _, wrapped, _ = make_synthetic((35, 47), noise=0.01)
    result, info = unwrap_lp(wrapped, p=1.2, outer_iter=4, return_info=True)
    assert np.isfinite(result).all()
    assert 1 <= info.outer_iterations <= 4
    assert info.inner_iterations >= info.outer_iterations
    assert np.isfinite(info.objective)


def test_lp_rejects_nonconvex_request():
    with pytest.raises(ValueError):
        unwrap_lp(np.zeros((4, 4)), p=0.5)


def test_goldstein_smooth_synthetic_and_cut_shape():
    truth, wrapped, _ = make_synthetic((31, 39), noise=0)
    result, cuts = goldstein_unwrap(wrapped, return_cuts=True)
    assert cuts.shape == wrapped.shape and cuts.dtype == bool
    assert rmse_aligned(result, truth) < 1e-5


def test_mask_cut_smooth_synthetic():
    truth, wrapped, _ = make_synthetic((25, 31), noise=0)
    result, cuts = mask_cut_unwrap(wrapped, return_cuts=True)
    assert cuts.shape == wrapped.shape
    assert rmse_aligned(result, truth) < 1e-5


@pytest.mark.parametrize("algorithm", [goldstein_unwrap, mask_cut_unwrap])
def test_branch_cut_algorithms_unwrap_valid_pixels_touching_a_mask(algorithm):
    """A valid pixel beside the mask is data, not a border.

    Both algorithms exclude the invalid region grown by one pixel from residue
    counting and cut growth, so that a quad straddling the edge of the data is
    not mistaken for a residue.  That guard once reached the integration step
    as well, which left a one-pixel frame of valid pixels at zero and made them
    wrong by several whole cycles.  A cropped mask puts the frame at the image
    edge and an interior hole puts it in the middle of the scene.
    """
    truth, wrapped, _ = make_synthetic((32, 44), noise=0)
    cropped = np.zeros(truth.shape, dtype=bool)
    cropped[2:-2, 2:-2] = True
    holed = np.ones(truth.shape, dtype=bool)
    holed[10:18, 14:30] = False
    rows, cols = truth.shape

    for valid in (cropped, holed):
        result, cuts = algorithm(wrapped, mask=valid, return_cuts=True)
        assert cuts.shape == wrapped.shape
        usable = valid & ~cuts
        difference = result - truth
        difference -= np.median(difference[usable])
        # 1e-4 rad leaves room for the float32 accumulation of the integrator;
        # the defect this pins was wrong by more than 1 rad.
        assert np.max(np.abs(difference[usable])) < 1e-4

        padded = np.pad(~valid, 1, constant_values=True)
        touching = np.zeros(truth.shape, dtype=bool)
        for row_offset in range(3):
            for col_offset in range(3):
                touching |= padded[row_offset:row_offset + rows,
                                   col_offset:col_offset + cols]
        on_the_edge = valid & touching
        assert on_the_edge.any() and np.any(on_the_edge & usable)
        assert np.max(np.abs(difference[on_the_edge & usable])) < 1e-4
        assert not np.any(result[usable] == 0)


def test_surface_diagnostics_remove_offset_and_find_jump():
    surface = np.zeros((5, 6))
    surface[:, 3:] = 2 * np.pi
    difference, metrics = surface_difference(surface + 7, surface,
                                             return_metrics=True)
    assert np.max(np.abs(difference)) < 1e-12
    assert metrics.mean_offset == pytest.approx(7)
    jumps, fraction = discontinuity_map(surface, return_fraction=True)
    assert jumps[:, 2].sum() == 4
    assert fraction == pytest.approx(4 / 20)


def test_flynn_result_is_phase_congruent():
    _, wrapped, _ = make_synthetic((17, 23), noise=0)
    result, iterations = flynn_unwrap(wrapped, return_iterations=True)
    cycles = (result - wrapped) / (2 * np.pi)
    cycles -= cycles.flat[0]
    assert np.max(np.abs(cycles - np.rint(cycles))) < 2e-6
    assert iterations > 0


def test_flynn_with_mask_and_quality_matches_reference_cycles():
    """Flynn with a mask is the case that takes the non-uniform cost path.

    An unmasked scene gives every edge the same cost, so the masks below are
    what force the algorithm to make real choices.  The number of whole cycles
    added to each pixel is pinned exactly: it is the observable output of the
    search, and it is identical to the reference implementation.
    """
    _, wrapped, _ = make_synthetic((11, 13), noise=0.05, seed=3)
    mask = np.zeros(wrapped.shape, dtype=bool)
    mask[3:6, 2:5] = True
    mask[8, 9] = True
    quality = max_gradient_quality(wrapped, window=1)

    result, iterations = flynn_unwrap(wrapped, quality=quality, mask=mask,
                                      return_iterations=True)

    assert result.dtype == np.float64
    assert np.isfinite(result).all()
    cycles = (result - wrapped) / (2 * np.pi)
    cycles -= cycles.flat[0]
    assert np.max(np.abs(cycles - np.rint(cycles))) < 2e-6
    cycles = np.rint(cycles).astype(int)

    assert iterations == 5
    assert np.count_nonzero(cycles) == 7
    assert cycles.tolist() == [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
