"""Tests for minimum-cost-flow unwrapping on Costantini's dual network.

The network is described in :mod:`parvaneh.network_flow`: one node per 2x2
cell, one arc per pixel edge, one ground node for everything outside the data.
A feasible flow is exactly a curl-free corrected field, and the flow on an arc
*is* the integer ambiguity of that pixel edge.  The tests below pin down three
separate claims.

* The answer is a real unwrapping.  It keeps the wrapped values of the input,
  its steps differ from the wrapped steps by whole turns, and every interior
  loop closes exactly, even where the wrapped input carries residues.
* The flow is optimal.  An independent solve with HiGHS computes the continuous
  optimum of the same problem, which is also the integer optimum because the
  constraint matrix of a network is totally unimodular; the two agree exactly.
  The L1 and L2 modes are cross-checked the same way, each beating the other on
  its own objective.  Because the objective is symmetric in the flow, the
  optimum alone does not pin the orientation of the arcs, so the cost is also
  recomputed from the returned result: that recomputation reads the flow as
  whole turns of pixel steps, and it only agrees with the solver when every arc
  is oriented consistently with the grid.
* The guards behave.  A mask or a zero weight removes pixels from the network,
  a partly valid image keeps independent offsets per region, and a flow that
  cannot finish in the allowed number of steps reports that instead of quietly
  returning a worse answer.
"""

import doctest

import numpy as np
import pytest
from scipy.optimize import linprog

from parvaneh import (NetworkFlowInfo, network_flow_unwrap, phase_residues,
                      rmse_aligned, wrap_phase)
from parvaneh.network_flow import COST_SCALE, _build


def smooth_surface(rows=24, cols=31, slope=(0.12, 0.05)):
    """A noise-free surface whose steps stay well inside one fringe."""
    y, x = np.mgrid[0:rows, 0:cols].astype(float)
    return slope[0] * x + slope[1] * y


def noisy_field(sigma=4.0, seed=5, rows=12, cols=10):
    """A wrapped speckle field: the truth, and the residues it hides."""
    truth = smooth_surface(rows, cols)
    rng = np.random.default_rng(seed)
    return truth, wrap_phase(truth + rng.normal(scale=sigma, size=truth.shape))


def cell_curl(field):
    """Circulation of a field around every 2x2 cell.

    This is the quantity a residue counts: it is what the cell constraints of
    the network force to zero.
    """
    down = field[1:, :] - field[:-1, :]
    right = field[:, 1:] - field[:, :-1]
    return right[:-1, :] + down[:, 1:] - right[1:, :] - down[:, :-1]


def interior_cells(valid):
    """Cells whose four corners are all usable, so a constraint exists."""
    return (valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, 1:] & valid[1:, :-1])


def node_arc_incidence(tail, head, ground, ncells):
    """Node-arc incidence matrix of the network the module built.

    A ground end contributes nothing: leaving the data and coming back is what
    the ground node absorbs, which is exactly why border steps are free.
    """
    matrix = np.zeros((ncells, tail.size))
    for edge in range(tail.size):
        if tail[edge] != ground:
            matrix[tail[edge], edge] += 1.0
        if head[edge] != ground:
            matrix[head[edge], edge] -= 1.0
    return matrix


def lp_optimum(phase, confidence, quadratic=False, levels=6):
    """Optimum of the same network, computed by HiGHS instead.

    The arcs and their costs are taken from the module so that this isolates
    the *minimisation*: the network construction is covered by the public-API
    tests.  Linear costs need one non-negative variable per arc, and a convex
    quadratic cost is expanded into successive increments whose marginal costs
    1, 3, 5, ... are its exact optimality conditions, so a linear program still
    models the integer problem exactly.
    """
    supply, tail, head, unit, _, _, _, cells, ground = _build(phase, confidence)
    ncells = int(cells.sum())
    width = tail.size
    if width == 0 or ncells == 0:
        return 0
    unit = unit.astype(float)
    steps = levels if quadratic else 1
    matrix = node_arc_incidence(tail, head, ground, ncells)
    blocks = matrix @ np.tile(np.eye(width), (1, steps))
    marginal = np.concatenate([unit * (2 * level + 1)
                               for level in range(steps)])
    optimum = linprog(np.concatenate([marginal, marginal]),
                      A_eq=np.hstack([blocks, -blocks]),
                      b_eq=-supply[:ncells].astype(float),
                      bounds=(0, None), method="highs")
    assert optimum.status == 0, optimum.message
    return int(round(optimum.fun))


def confidence_to_cost(confidence):
    """The integer cost of one pixel edge, straight from the definition.

    An edge is as expensive as its least reliable end, the same rule the
    least-squares backends use, rescaled so the flow stays exact in integers.
    A pixel with zero confidence has cost zero, which switches its edges off.
    """
    scaled = np.rint(COST_SCALE * np.clip(confidence, 0.0, 1.0))
    scaled = np.where(confidence > 0.0, np.maximum(scaled, 1), 0)
    return np.minimum(scaled[:, :-1], scaled[:, 1:]), np.minimum(scaled[:-1, :],
                                                                 scaled[1:, :])


def turns_of(phase, result):
    """Whole turns added to each right and down step of the wrapped input."""
    right = (np.diff(result, axis=1) - wrap_phase(np.diff(phase, axis=1)))
    down = (np.diff(result, axis=0) - wrap_phase(np.diff(phase, axis=0)))
    return right / (2.0 * np.pi), down / (2.0 * np.pi)


def objective(phase, result, confidence, quadratic=False):
    """The cost of a candidate solution, recomputed from the public result."""
    right_cost, down_cost = confidence_to_cost(confidence)
    right, down = turns_of(phase, result)
    finite_right = np.isfinite(result[:, :-1]) & np.isfinite(result[:, 1:])
    finite_down = np.isfinite(result[:-1, :]) & np.isfinite(result[1:, :])
    right = np.where(finite_right, right, 0.0)
    down = np.where(finite_down, down, 0.0)
    if quadratic:
        return float((right_cost * right ** 2).sum()
                     + (down_cost * down ** 2).sum())
    return float((right_cost * np.abs(right)).sum()
                 + (down_cost * np.abs(down)).sum())


# --------------------------------------------------------------------------
# the answer is a valid unwrapping
# --------------------------------------------------------------------------

def test_a_clean_ramp_needs_no_flow_at_all():
    """With no residues the wrapped steps are already curl free."""
    truth = smooth_surface(24, 31)
    result, info = network_flow_unwrap(wrap_phase(truth), return_info=True)

    assert rmse_aligned(result, truth) < 1e-10
    assert info.total_cost == 0
    assert info.augmentations == 0
    assert info.residues == 0
    assert info.max_jump == 0


def test_the_result_keeps_the_wrapped_values_of_the_input():
    """Any unwrapping has to differ from its input by whole turns."""
    truth, phase = noisy_field(sigma=4.0)
    result = network_flow_unwrap(phase)

    assert isinstance(result, np.ndarray)
    assert result.shape == phase.shape
    assert np.abs(wrap_phase(result - phase)).max() < 1e-9

    right, down = turns_of(phase, result)
    turns = np.concatenate([right.ravel(), down.ravel()])
    assert np.abs(turns - np.rint(turns)).max() < 1e-9


def test_every_interior_loop_closes_even_where_the_input_has_residues():
    """Conservation of flow is the curl-free condition, cell by cell."""
    truth, phase = noisy_field(sigma=4.0)
    result = network_flow_unwrap(phase)

    assert np.abs(phase_residues(phase)).sum() > 0
    inside = interior_cells(np.ones(phase.shape, dtype=bool))
    assert np.abs(cell_curl(result)[inside]).max() < 1e-8


@pytest.mark.parametrize("sigma, seed", [(0.4, 7), (1.0, 11), (1.0, 13)])
def test_a_noisy_scene_unwraps_to_the_right_surface(sigma, seed):
    """The result tracks the truth down to the noise floor.

    The noise itself cannot be recovered, so the result differs from the truth
    by about one standard deviation per pixel; what the unwrapping has to get
    right is every whole turn.  A constant offset is allowed, because the
    reference surface is only defined up to one.
    """
    truth, phase = noisy_field(sigma=sigma, seed=seed, rows=32, cols=32)
    result = network_flow_unwrap(phase)

    assert rmse_aligned(result, truth) < 1.5 * sigma


def test_the_info_record_describes_the_documented_network():
    truth = smooth_surface(9, 8)
    result, info = network_flow_unwrap(wrap_phase(truth), return_info=True)

    assert isinstance(info, NetworkFlowInfo)
    assert (info.backend, info.cost) == ("python", "linear")
    assert info.pixels == 72
    # One node per 2x2 cell, plus the ground node.
    assert info.nodes == (9 - 1) * (8 - 1) + 1
    # One arc per pixel edge; none is dropped when everything is valid.
    assert info.edges == (9 - 1) * 8 + 9 * (8 - 1)
    assert info.ground_imbalance == 0
    assert np.isfinite(result).all()


# --------------------------------------------------------------------------
# the flow is optimal
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sigma, seed", [(0.4, 7), (3.0, 8), (5.0, 9)])
def test_the_flow_cost_is_the_optimum_a_linear_program_finds(sigma, seed):
    """HiGHS and the successive-shortest-path solver agree exactly.

    The two searches share nothing but the network, so an exact match on four
    different instances is strong evidence that the flow really is minimal.
    """
    truth, phase = noisy_field(sigma=sigma, seed=seed)
    confidence = np.ones(phase.shape)
    result, info = network_flow_unwrap(phase, return_info=True)

    assert info.total_cost == int(objective(phase, result, confidence))
    assert info.total_cost == lp_optimum(phase, confidence)


def test_a_masked_instance_is_optimal_too():
    """Cells touching the mask are gone, so the flow has more room."""
    truth, phase = noisy_field(sigma=4.0, seed=11, rows=12, cols=10)
    confidence = np.ones(phase.shape)
    confidence[4:7, 4:8] = 0.0
    result, info = network_flow_unwrap(phase, confidence, return_info=True)

    assert info.total_cost == lp_optimum(phase, confidence)
    assert np.array_equal(np.isfinite(result), confidence > 0.0)


def test_a_quadratic_instance_is_optimal_against_its_expanded_program():
    truth, phase = noisy_field(sigma=4.0, seed=12)
    confidence = np.ones(phase.shape)
    result, info = network_flow_unwrap(phase, cost="quadratic",
                                       return_info=True)

    assert info.cost == "quadratic"
    assert lp_optimum(phase, confidence, quadratic=True) == int(
        objective(phase, result, confidence, quadratic=True))


def test_each_cost_mode_beats_the_other_on_its_own_objective():
    """L1 counts turns, L2 squares them, and nothing else is optimal for both.

    The recorded ``total_cost`` is the L1 measure in either mode, so the two
    objectives are recomputed from the results before they are compared.
    """
    truth, phase = noisy_field(sigma=4.0, seed=13, rows=16, cols=16)
    confidence = np.ones(phase.shape)
    linear, l_info = network_flow_unwrap(phase, cost="linear",
                                         return_info=True)
    quadratic, q_info = network_flow_unwrap(phase, cost="quadratic",
                                            return_info=True)

    l1 = objective(phase, linear, confidence)
    l1_other = objective(phase, quadratic, confidence)
    l2 = objective(phase, quadratic, confidence, quadratic=True)
    l2_other = objective(phase, linear, confidence, quadratic=True)

    assert l_info.total_cost == int(l1)
    assert l1 <= l1_other
    assert l2 <= l2_other
    # Both are unwrappings of the same input, so both keep its wrapped values.
    for result in (linear, quadratic):
        assert np.abs(wrap_phase(result - phase)).max() < 1e-9


def test_weighting_the_edges_changes_which_solution_is_cheapest():
    """A weight is an instruction: turns avoid the pixels that are cheap to trust."""
    truth, phase = noisy_field(sigma=4.0, seed=14, rows=16, cols=16)
    confidence = np.ones(phase.shape)
    confidence[:, 7:9] = 0.2

    unweighted = network_flow_unwrap(phase)
    weighted = network_flow_unwrap(phase, confidence)

    uniform = np.ones(phase.shape)
    # The weighted run minimises the weighted cost, so it can only be better
    # there, and the uniform run can only be better on the uniform cost.
    assert objective(phase, weighted, confidence) < objective(
        phase, unweighted, confidence)
    assert objective(phase, unweighted, uniform) <= objective(
        phase, weighted, uniform)


def test_a_uniform_weight_reproduces_the_unweighted_answer():
    truth, phase = noisy_field(sigma=4.0, seed=15)
    plain, plain_info = network_flow_unwrap(phase, return_info=True)
    scaled, scaled_info = network_flow_unwrap(phase, np.ones(phase.shape),
                                              return_info=True)

    assert np.allclose(plain, scaled)
    assert plain_info.total_cost == scaled_info.total_cost


# --------------------------------------------------------------------------
# masks, zero weights, and the ground node
# --------------------------------------------------------------------------

def test_a_masked_hole_comes_back_as_nan():
    truth, phase = noisy_field(sigma=3.0, seed=16, rows=12, cols=10)
    mask = np.ones(phase.shape, dtype=bool)
    mask[4:7, 4:8] = False
    result, info = network_flow_unwrap(phase, mask=mask, return_info=True)

    assert np.array_equal(np.isfinite(result), mask)
    assert info.pixels == int(mask.sum())
    assert info.components == 1
    # Cells touching the mask carry no constraint, so only the cells with four
    # usable corners are expected to close.
    assert np.abs(cell_curl(np.where(np.isfinite(result), result, 0.0))
                  [interior_cells(mask)]).max() < 1e-8


def test_a_zero_weight_removes_a_pixel_just_like_a_mask():
    truth, phase = noisy_field(sigma=3.0, seed=17, rows=12, cols=10)
    weight = np.ones(phase.shape)
    weight[:, 5] = 0.0

    by_weight = network_flow_unwrap(phase, weight)
    by_mask = network_flow_unwrap(phase, weight > 0.0)

    assert np.isnan(by_weight[:, 5]).all()
    assert np.array_equal(np.isfinite(by_weight), np.isfinite(by_mask))
    assert np.allclose(by_weight, by_mask, equal_nan=True)


def test_regions_that_a_zero_weight_separates_keep_their_own_offset():
    """Each connected region is anchored at its own first pixel."""
    truth = smooth_surface(24, 21)
    phase = wrap_phase(truth)
    weight = np.ones(phase.shape)
    weight[:, 10] = 0.0

    result, info = network_flow_unwrap(phase, weight, return_info=True)
    left = result[:, :10] - truth[:, :10]
    right = result[:, 11:] - truth[:, 11:]

    assert info.components == 2
    # One constant offset per region, exactly: no drift within either side.
    assert np.ptp(left) < 1e-9
    assert np.ptp(right) < 1e-9
    # The two offsets differ by a whole number of turns, because nothing in the
    # network relates the two halves of the image.
    assert abs(wrap_phase(left.mean() - right.mean())) < 1e-9


def test_a_fully_masked_image_has_nothing_to_solve():
    phase = np.zeros((4, 5))
    result, info = network_flow_unwrap(phase, np.zeros(phase.shape),
                                       return_info=True)

    assert np.isnan(result).all()
    assert info.pixels == 0
    assert info.nodes == 1  # the ground node alone
    assert info.edges == 0
    assert info.components == 0


def test_a_single_usable_pixel_needs_no_edges():
    phase = np.zeros((3, 3))
    mask = np.zeros(phase.shape, dtype=bool)
    mask[1, 1] = True
    result, info = network_flow_unwrap(phase, mask=mask, return_info=True)

    assert info.pixels == 1
    assert info.nodes == 1
    assert info.edges == 0
    assert info.components == 1
    assert result[1, 1] == 0.0
    assert np.isnan(result[0, 0])


# --------------------------------------------------------------------------
# arguments, guards, and the documented example
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"phase": np.zeros(6)}, "finite 2-D array"),
        ({"phase": np.zeros((1, 6))}, "finite 2-D array"),
        ({"phase": np.full((4, 4), np.nan)}, "finite 2-D array"),
        ({"phase": np.full((4, 4), np.inf)}, "finite 2-D array"),
        ({"cost": "cubic"}, "cost must be one of"),
        ({"backend": "numba"}, "backend must be 'python'"),
        ({"max_iter": -1}, "nonnegative"),
        ({"weight": np.ones((3, 3))}, "weight must be finite"),
        ({"weight": -np.ones((4, 4))}, "nonnegative"),
        ({"weight": np.full((4, 4), np.inf)}, "weight must be finite"),
        ({"mask": np.ones((3, 3), dtype=bool)}, "mask must match"),
    ],
)
def test_malformed_input_is_refused_with_a_message(kwargs, message):
    phase = kwargs.pop("phase", np.zeros((4, 4)))
    with pytest.raises(ValueError, match=message):
        network_flow_unwrap(phase, **kwargs)


def test_an_unfinished_flow_is_reported_instead_of_returned():
    """``max_iter`` is a work limit, so hitting it is an error, not an answer.

    A truncated flow would look like a result while being neither optimal nor
    even feasible, so the solver refuses rather than mislead.
    """
    truth, phase = noisy_field(sigma=8.0, seed=18, rows=16, cols=16)
    _, info = network_flow_unwrap(phase, return_info=True)
    assert info.augmentations >= 2

    with pytest.raises(RuntimeError):
        network_flow_unwrap(phase, max_iter=1)


def test_zero_residue_data_ignores_a_zero_max_iter():
    """No excess anywhere means no augmentation is ever attempted."""
    truth = smooth_surface(8, 8)
    result, info = network_flow_unwrap(wrap_phase(truth), max_iter=0,
                                       return_info=True)
    assert info.augmentations == 0
    assert rmse_aligned(result, truth) < 1e-10


def test_the_documented_example_in_the_module_is_true():
    """The docstring of :mod:`parvaneh.network_flow` is part of its contract."""
    from parvaneh import network_flow

    assert doctest.testmod(network_flow, verbose=False).failed == 0
