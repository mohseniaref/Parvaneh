import numpy as np
import pytest

from parvaneh import rmse_aligned, wrap_phase
from parvaneh.graph_cut import puma_unwrap


def scene():
    y, x = np.mgrid[:14, :15]
    truth = 7.0 * np.exp(-((x - 7) ** 2 + (y - 6) ** 2) / 20.0)
    return truth, wrap_phase(truth)


def test_puma_is_not_the_global_noop_b20():
    truth, phase = scene()
    result, info = puma_unwrap(phase, p=1, max_jump=2, return_info=True)
    assert info.accepted > 0
    assert info.objective < info.initial_objective
    assert rmse_aligned(result, truth) < rmse_aligned(phase, truth)
    assert np.allclose(wrap_phase(result), phase)


def test_reported_objective_is_the_returned_fields_energy():
    _, phase = scene()
    result, info = puma_unwrap(phase, p=2, return_info=True)
    energy = sum(np.sum(np.abs(np.diff(result, axis=a)) ** 2)
                 for a in range(2))
    assert info.objective == pytest.approx(energy)
    assert not info.proven


def test_mask_and_validation():
    _, phase = scene()
    mask = np.ones(phase.shape, bool)
    mask[4:7, 5:8] = False
    result = puma_unwrap(phase, mask=mask)
    assert np.array_equal(np.isfinite(result), mask)
    for kwargs in ({"p": .5}, {"max_jump": 0}, {"align": "x"},
                   {"max_iter": 0}):
        with pytest.raises(ValueError):
            puma_unwrap(phase, **kwargs)
