import numpy as np
import pytest

from parvaneh import wrap_phase
from parvaneh.flow_nd import grid_cycles, grid_edge_index, grid_edges, jumps_from_field
from parvaneh.graph_flow import residues_of_cycles
from parvaneh.stat_costs import (DeformationCost, StatCostParams, _star_moves,
                                 stat_cost_unwrap)


def test_deformation_cost_has_the_quadratic_shelf_and_falloff_b19():
    cost = DeformationCost([40.0], [0.0], shelf=[True], level=[10.0],
                           defomax=1.2, falloff=2.0)
    z = np.array([0.0, 0.3, 0.5, 1.0, 1.2, 1.5, 2.0])
    got = cost.cost_at(np.zeros(z.size, dtype=int), z)
    expected = np.where(z > 1.2, 40 * (z - 1.2) ** 2 / 2 + 10,
                        np.minimum(40 * z ** 2, 10))
    assert np.allclose(got, expected)
    assert np.all(np.diff(got) >= -1e-12)


def test_vertex_star_moves_leave_every_grid_cycle_unchanged_b16():
    nodes, tail, head = grid_edges((4, 5))
    cycles = grid_cycles(nodes.shape, grid_edge_index(nodes.shape))
    moves, signs = _star_moves(nodes.size, tail, head)
    matrix = np.zeros((len(cycles), tail.size), dtype=int)
    for row, (edges, orientation) in enumerate(cycles):
        matrix[row, edges] = orientation
    for edges, direction in zip(moves, signs):
        delta = np.zeros(tail.size, dtype=int)
        np.add.at(delta, edges, direction)
        assert np.array_equal(matrix @ delta, np.zeros(len(cycles), dtype=int))


def test_capped_deformation_run_is_finite_feasible_and_nonincreasing_b16_b19():
    y, x = np.mgrid[:10, :11]
    phase = wrap_phase(6 * np.exp(-((x - 5) ** 2 + (y - 5) ** 2) / 12))
    result, info = stat_cost_unwrap(
        phase, np.full(phase.shape, 0.8),
        params=StatCostParams(kperpdpsi=3, kpardpsi=3),
        costmode="defo", shelf=True, return_info=True)
    assert np.isfinite(result).all()
    assert info.objective <= info.initial_objective + 1e-10
    nodes, tail, head = grid_edges(phase.shape)
    cycles = grid_cycles(phase.shape, grid_edge_index(phase.shape))
    jumps = np.concatenate([x.ravel() for x in jumps_from_field(phase, result)])
    gradients = np.concatenate([wrap_phase(np.diff(phase, axis=a)).ravel()
                                for a in range(2)])
    assert np.array_equal(np.array([np.dot(s, jumps[e]) for e, s in cycles]),
                          -residues_of_cycles(gradients, cycles))


def test_bad_coherence_and_parameters_are_rejected():
    phase = np.zeros((8, 8))
    with pytest.raises(ValueError):
        stat_cost_unwrap(phase, np.full((8, 8), 1.1))
    with pytest.raises(ValueError):
        StatCostParams(kperpdpsi=2)
