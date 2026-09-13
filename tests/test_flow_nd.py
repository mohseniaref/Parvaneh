import numpy as np
import pytest

from parvaneh.flow_nd import (corrected_gradients, flow_nd_unwrap,
                              grid_cycles, grid_edge_index, grid_edges,
                              jumps_from_field)
from parvaneh import wrap_phase


def aligned_error(value, truth):
    delta = value - truth
    return np.max(np.abs(delta - np.nanmean(delta)))


def test_grid_enumeration_and_three_dimensional_cycle_count():
    for n in (2, 3, 4):
        nodes, tail, head = grid_edges((n, n, n))
        cycles = grid_cycles(nodes.shape, grid_edge_index(nodes.shape))
        assert len(cycles) == 3 * n * (n - 1) ** 2
        assert tail.size == 3 * n * n * (n - 1)
        assert np.all(head > tail)


def test_two_dimensional_flow_recovers_a_wrapped_plane_and_closes_b12():
    y, x = np.mgrid[:7, :8]
    truth = 1.1 * x + 0.8 * y
    phase = wrap_phase(truth)
    result, info = flow_nd_unwrap(phase, method="flow", return_info=True)
    assert aligned_error(result, truth) < 1e-10
    assert info.backend == "grid-network" and info.proven
    jumps = jumps_from_field(phase, result)
    corrected = corrected_gradients(phase, jumps)
    assert np.max(np.abs(np.diff(corrected[0], axis=1)
                         - np.diff(corrected[1], axis=0))) < 1e-10


def test_volume_auto_uses_exact_fallback_and_flow_is_refused():
    z, y, x = np.mgrid[:2, :3, :3]
    phase = wrap_phase(0.4 * x + 0.3 * y + 0.2 * z)
    result, info = flow_nd_unwrap(phase, method="auto", return_info=True)
    assert aligned_error(result, 0.4 * x + 0.3 * y + 0.2 * z) < 1e-10
    assert info.backend == "grid-branch-bound"
    with pytest.raises(ValueError, match="flow network"):
        flow_nd_unwrap(phase, method="flow")


@pytest.mark.parametrize("kwargs", [{"method": "bad"}, {"cost": "bad"},
                                    {"align": "bad"}, {"slice_axis": 9}])
def test_invalid_options_are_rejected(kwargs):
    with pytest.raises(ValueError):
        flow_nd_unwrap(np.zeros((3, 3)), **kwargs)
