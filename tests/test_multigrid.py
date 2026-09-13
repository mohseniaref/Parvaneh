"""Tests for the multigrid unwrapping in :mod:`parvaneh.multigrid`.

The solver is a V-cycle over a hierarchy of coarser and coarser versions of the
same weighted least-squares problem, described in :mod:`parvaneh.multigrid`.
The tests below pin down four separate claims.

* The answer is the same problem solved twice.  On the finest grid the
  equations are exactly the ones :func:`parvaneh.core.unwrap` writes down, so
  the two solvers must return the same field, and a ramp or a plane must come
  back exactly, up to the constant that no unwrapping method can know.
* The answer is an unwrapping.  Where the wrapped input carries no residue the
  result equals the input plus a global constant modulo ``2 pi``, and every
  pixel step differs from the wrapped step by a whole number of turns.  Where
  the input does carry residues that congruence is expected to fail, so it is
  only asserted on noise-free data.
* The hierarchy is what earns its keep.  Relaxing the finest grid alone still
  converges, but it needs an order of magnitude more sweeps, and the number of
  V-cycles needed at a fixed tolerance hardly grows with the image.
* The parts are sound and the guards behave.  The transfer operators are
  adjoint to each other, coarse confidences keep a uniform confidence uniform,
  the residual falls monotonically, and bad arguments, bad arrays and an
  empty problem are refused rather than guessed at.
"""

import doctest

import numpy as np
import pytest
from scipy import ndimage

from parvaneh import MultigridInfo, multigrid_unwrap, rmse_aligned, unwrap
from parvaneh import wrap_phase
from parvaneh.multigrid import _coarsen_confidence, _prolong, _restrict


def bump(rows, cols, scale=6.0):
    """A smooth dome, which is what a real surface looks like locally."""
    yy, xx = np.mgrid[0:rows, 0:cols]
    return scale * np.exp(-((yy - rows / 2.0) ** 2 + (xx - cols / 2.0) ** 2)
                          / (0.45 * rows * cols / 4.0))


def scene(rows, cols, sigma=0.0, scale=6.0, seed=7):
    """A wrapped dome with optional circular noise, and its truth."""
    truth = bump(rows, cols, scale=scale)
    rng = np.random.default_rng(seed)
    noise = sigma * rng.standard_normal((rows, cols))
    return truth, wrap_phase(wrap_phase(truth) + noise)


def aligned(a, b):
    """Largest difference between two fields once a global constant is gone."""
    difference = np.asarray(a) - np.asarray(b)
    return float(np.abs(difference - difference.mean()).max())


def turns(step):
    """Distance of a step field from the nearest whole number of turns."""
    quarters = np.asarray(step) / (2.0 * np.pi)
    return float(np.abs(quarters - np.rint(quarters)).max())


def planes():
    """A plane on every branch: even, odd, one-dimensional and a volume."""
    even = np.mgrid[0:16, 0:24]
    yield "plane 16x24", 0.3 * even[0] - 0.21 * even[1] + 1.7
    odd = np.mgrid[0:33, 0:17]
    yield "plane 33x17", 0.3 * odd[0] - 0.21 * odd[1] + 1.7
    square = np.mgrid[0:8, 0:8]
    yield "plane 8x8", 0.4 * square[0] - 0.3 * square[1]
    yield "line of 41", np.linspace(0.0, 12.0, 41)
    volume = np.mgrid[0:9, 0:11, 0:4]
    yield "volume 9x11x4", 0.25 * volume[0] - 0.2 * volume[2]


@pytest.mark.parametrize("name,plane", list(planes()))
def test_a_plane_comes_back_as_a_plane(name, plane):
    """A field with no wraps to resolve is copied, minus its constant."""
    result = multigrid_unwrap(wrap_phase(plane), tol=1e-13, max_cycles=200)
    assert aligned(result, plane) < 1e-9
    assert abs(result.mean()) < 1e-9


def test_the_result_keeps_the_input_modulo_one_global_constant():
    """With no residue the solver only moves the input by a global constant.

    The constant itself is unknowable, so the claim is made modulo ``2 pi``:
    the wrapped difference between the result and the input is the same
    everywhere, and every pixel step is the wrapped step plus whole turns.
    """
    for rows, cols in [(16, 16), (23, 31), (8, 8)]:
        _, phase = scene(rows, cols, seed=rows)
        result = multigrid_unwrap(phase, tol=1e-13, max_cycles=200)
        assert np.ptp(wrap_phase(result - phase)) < 1e-8
        for axis in (0, 1):
            step = np.diff(result, axis=axis) - np.diff(phase, axis=axis)
            assert turns(step) < 1e-8


@pytest.mark.parametrize("shape", [(23,), (2, 2), (7, 5), (17, 3), (23, 31),
                                   (9, 10, 4)])
@pytest.mark.parametrize("sigma", [0.0, 0.8])
def test_the_result_agrees_with_the_least_squares_solver(shape, sigma):
    """Both solvers state the same equations, so both must give one answer."""
    rows = shape[0]
    cols = shape[1] if len(shape) > 1 else shape[0]
    truth = bump(rows, cols)
    rng = np.random.default_rng(11)
    phase = wrap_phase(wrap_phase(truth)
                       + sigma * rng.standard_normal((rows, cols)))
    if len(shape) == 1:
        phase = phase[:, 0]
    elif len(shape) == 3:
        phase = np.stack([phase] * shape[2])
    reference = unwrap(phase, backend="numpy")
    result = multigrid_unwrap(phase, tol=1e-13, max_cycles=200)
    assert aligned(result, reference) < 1e-8


def test_a_weighted_problem_agrees_with_the_weighted_solver():
    """Varying coefficients are the interesting case, and the same system."""
    rng = np.random.default_rng(23)
    for rows, cols in [(24, 24), (40, 40)]:
        _, phase = scene(rows, cols, sigma=0.6, seed=rows)
        weight = rng.uniform(0.05, 1.0, phase.shape)
        reference = unwrap(phase, weight, max_iter=5000, tol=1e-14)
        result = multigrid_unwrap(phase, weight=weight, tol=1e-13,
                                  max_cycles=300)
        assert aligned(result, reference) < 1e-8


def test_a_zero_weight_is_a_mask():
    """A weight of zero removes the samples, just as a false mask entry does."""
    _, phase = scene(20, 22, sigma=0.5, seed=5)
    mask = np.ones(phase.shape, dtype=bool)
    mask[:, 9:11] = False
    by_weight = multigrid_unwrap(phase, weight=mask.astype(float))
    by_mask = multigrid_unwrap(phase, mask=mask)
    assert np.array_equal(by_weight, by_mask)


def test_a_mask_leaves_one_free_constant_per_region():
    """Each island of valid samples carries its own unknowable constant."""
    for cut in [(slice(None), slice(11, 13)), (slice(6, 9), slice(None))]:
        _, phase = scene(24, 26, sigma=0.6, seed=3)
        mask = np.ones(phase.shape, dtype=bool)
        mask[cut] = False
        reference = unwrap(phase, mask.astype(float), max_iter=5000,
                           tol=1e-14)
        result = multigrid_unwrap(phase, mask=mask, tol=1e-13, max_cycles=300)
        labels, count = ndimage.label(mask)
        assert count == 2
        for label in range(1, count + 1):
            island = labels == label
            difference = (result - reference)[island]
            assert np.abs(difference - difference.mean()).max() < 1e-8


def test_masked_samples_are_inert():
    """A removed pixel carries no information, in either direction.

    Every edge that touches a masked sample has zero weight, so the sample
    never appears in a residual: what the solver leaves there is arbitrary,
    and the rest of the answer cannot depend on it.  Replacing the phase of
    the removed samples with large garbage therefore changes nothing at all,
    which is the cheapest way to state that the mask really is a mask.
    """
    _, phase = scene(24, 24, sigma=0.6, seed=9)
    mask = np.ones(phase.shape, dtype=bool)
    mask[:, 11:13] = False
    rng = np.random.default_rng(5)
    wild = phase.copy()
    wild[~mask] = 100.0 * rng.standard_normal(int((~mask).sum()))
    plain = multigrid_unwrap(phase, mask=mask, tol=1e-12, max_cycles=300)
    junk = multigrid_unwrap(wild, mask=mask, tol=1e-12, max_cycles=300)
    assert np.isfinite(plain).all()
    assert np.array_equal(plain, junk)
    assert np.abs(plain[mask]).max() > 1.0


def test_the_hierarchy_beats_relaxing_the_finest_grid_alone():
    """Coarse grids are the whole point; without them the work explodes.

    Both runs reach the same tolerance, and the number of Gauss--Seidel sweeps
    is the honest work measure, because a sweep costs the same wherever it is
    spent and the coarse grids are only a few samples across.
    """
    _, phase = scene(64, 64, sigma=0.4, seed=13)
    _, whole = multigrid_unwrap(phase, tol=1e-7, max_cycles=400,
                                return_info=True)
    _, alone = multigrid_unwrap(phase, tol=1e-7, max_cycles=400, levels=1,
                                return_info=True)
    assert whole.converged and alone.converged
    assert whole.levels > 1 and alone.levels == 1
    assert whole.sweeps * 4 < alone.sweeps


def test_the_cycle_count_barely_grows_with_the_image():
    """A V-cycle is size independent, which is what multigrid promises."""
    counts = []
    for size in (32, 64):
        _, phase = scene(size, size, sigma=0.4, seed=size)
        _, info = multigrid_unwrap(phase, tol=1e-8, max_cycles=200,
                                   return_info=True)
        assert info.converged
        counts.append(info.cycles)
    assert counts[-1] <= 2 * counts[0]


def test_the_v_cycle_contracts_the_residual():
    """Each cycle leaves a smaller residual than the one before it."""
    rng = np.random.default_rng(17)
    for weight in (None, rng.uniform(0.05, 1.0, (32, 32))):
        _, phase = scene(32, 32, sigma=0.4, seed=19)
        _, info = multigrid_unwrap(phase, weight=weight, tol=1e-9,
                                   max_cycles=200, return_info=True)
        assert info.converged
        history = info.residuals
        assert all(later < earlier
                   for earlier, later in zip(history, history[1:]))
        assert history[-1] < 0.5 * history[0]


def test_a_constant_field_needs_no_cycle():
    """A free constant gives a zero right-hand side, not an endless chase."""
    result, info = multigrid_unwrap(np.full((8, 12), 0.7), return_info=True)
    assert info.cycles == 0
    assert info.converged and info.relative_residual == 0.0
    assert np.abs(result).max() == 0.0
    assert isinstance(info, MultigridInfo)


def test_the_transfer_operators_are_adjoint_up_to_the_scaling():
    """``R`` must be the transpose of ``P`` divided by ``2`` per axis.

    This is what makes the coarse-grid correction a projection of the fine
    residual rather than an ad hoc average, and it is verified as the inner
    product identity ``2^d <R f, c> = <f, P c>``.
    """
    rng = np.random.default_rng(29)
    for shape in [(16, 16), (17, 23), (9, 10, 4)]:
        fine = rng.standard_normal(shape)
        coarse = rng.standard_normal(_restrict(np.ones(shape)).shape)
        restricted = _restrict(fine)
        prolonged = _prolong(coarse, shape)
        left = (2.0 ** len(shape)) * float(np.sum(restricted * coarse))
        right = float(np.sum(fine * prolonged))
        assert abs(left - right) < 1e-11 * max(1.0, abs(right))


def test_coarse_confidence_keeps_a_uniform_confidence_uniform():
    """The bug that made the cycle rate drift with the image size.

    Restricting a constant field with a normalised filter returns something
    smaller than the constant, so the coarse confidence has to be divided by
    the restriction of ones.  Without that division a uniform problem acquires
    weights that fall with the level and the cycle rate crawls up with size.
    """
    uniform = np.ones((16, 24))
    coarse = _coarsen_confidence(uniform)
    assert np.abs(coarse - 1.0).max() < 1e-12
    assert np.abs(_coarsen_confidence(coarse) - 1.0).max() < 1e-12

    rng = np.random.default_rng(37)
    nodes = rng.uniform(0.1, 1.0, (20, 26))
    smoothed = _coarsen_confidence(nodes)
    assert smoothed.min() >= nodes.min() - 1e-12
    assert smoothed.max() <= nodes.max() + 1e-12


def test_the_hierarchy_stops_before_an_axis_gets_too_short():
    """Every level needs an edge to compute, so coarsening has to stop."""
    _, phase = scene(16, 16)
    _, info = multigrid_unwrap(phase, return_info=True)
    assert info.levels == 4
    assert info.coarsest == (2, 2)

    _, phase = scene(8, 12)
    _, info = multigrid_unwrap(phase, return_info=True)
    assert info.levels == 3
    assert info.coarsest == (2, 3)


def test_the_work_is_bookkept():
    """The reported cycle, sweep and residual counts describe the run."""
    _, phase = scene(32, 32, sigma=0.3, seed=41)
    _, info = multigrid_unwrap(phase, tol=1e-9, max_cycles=200,
                               pre_smooth=3, post_smooth=2, coarse_sweeps=20,
                               return_info=True)
    per_cycle = (info.levels - 1) * 5 + 20
    assert info.cycles == len(info.residuals) == info.sweeps // per_cycle
    assert info.sweeps == info.cycles * per_cycle
    assert info.converged == (info.relative_residual <= 1e-9)
    assert info.relative_residual == info.residuals[-1]
    assert info.seconds > 0.0
    assert info.backend == "numpy"


def test_the_same_input_gives_the_same_answer():
    """The iteration is deterministic, so a rerun is not a new experiment."""
    _, phase = scene(24, 20, sigma=0.5, seed=43)
    first = multigrid_unwrap(phase, tol=1e-10)
    second = multigrid_unwrap(phase, tol=1e-10)
    assert np.array_equal(first, second)


@pytest.mark.parametrize("kwargs", [{"levels": 0}, {"max_cycles": 0},
                                   {"tol": 0.0}, {"pre_smooth": -1},
                                   {"pre_smooth": 0, "post_smooth": 0},
                                   {"coarse_sweeps": 0},
                                   {"backend": "cupy"}])
def test_bad_arguments_are_refused(kwargs):
    """A parameter that cannot mean anything is an error, not a default."""
    with pytest.raises(ValueError):
        multigrid_unwrap(np.zeros((8, 8)), **kwargs)


def test_bad_arrays_are_refused():
    """An arity, a value or a shape that cannot be unwrapped is an error."""
    with pytest.raises(ValueError):
        multigrid_unwrap(np.float64(0.5))
    with pytest.raises(ValueError):
        multigrid_unwrap(np.zeros((1, 5)))
    with pytest.raises(ValueError):
        multigrid_unwrap(np.array([[0.0, np.nan], [0.1, 0.2]]))
    with pytest.raises(ValueError):
        multigrid_unwrap(np.array([[0.0, np.inf], [0.1, 0.2]]))
    with pytest.raises(ValueError):
        multigrid_unwrap(np.zeros((8, 8)), weight=np.ones((8, 7)))
    with pytest.raises(ValueError):
        multigrid_unwrap(np.zeros((8, 8)), weight=-np.ones((8, 8)))
    with pytest.raises(ValueError):
        multigrid_unwrap(np.zeros((8, 8)), weight=np.full((8, 8), np.nan))
    with pytest.raises(ValueError):
        multigrid_unwrap(np.zeros((8, 8)), mask=np.ones((8, 7), dtype=bool))
    with pytest.raises(ValueError):
        multigrid_unwrap(np.zeros((8, 8)), weight=np.zeros((8, 8)))


def test_an_empty_problem_is_reported_as_an_error():
    """No usable sample anywhere means there is nothing to solve."""
    with pytest.raises(ValueError):
        multigrid_unwrap(np.zeros((8, 8)), mask=np.zeros((8, 8), dtype=bool))


def test_the_masked_island_experiment_is_reproducible():
    """The published comparison must not depend on the seed only by luck.

    The multigrid solution is a weighted least squares one, so on the island
    problem it matches the least-squares solver rather than the network-flow
    solver; this test records the measurement the documentation quotes.
    """
    _, phase = scene(24, 24, sigma=0.6, seed=47)
    mask = np.ones(phase.shape, dtype=bool)
    mask[:, 11:13] = False
    truth = bump(24, 24)
    reference = unwrap(phase, mask.astype(float), max_iter=5000, tol=1e-14)
    result = multigrid_unwrap(phase, mask=mask, tol=1e-13, max_cycles=300)
    labels, count = ndimage.label(mask)
    assert count == 2
    errors = []
    for label in range(1, count + 1):
        island = labels == label
        errors.append(rmse_aligned(result[island], truth[island]))
        assert aligned(result[island], reference[island]) < 1e-8
    assert max(errors) < 1.0


def test_the_documented_example_in_the_module_is_true():
    """The docstrings of :mod:`parvaneh.multigrid` are part of its contract."""
    from parvaneh import multigrid

    assert doctest.testmod(multigrid, verbose=False).failed == 0
