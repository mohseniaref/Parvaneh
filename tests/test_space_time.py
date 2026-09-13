import numpy as np
import pytest

from parvaneh import wrap_phase
from parvaneh.space_time import (SpaceTimeParams, _smooth_series,
                                 space_time_priors, space_time_unwrap)


def test_series_smoothing_operates_on_epochs_before_design_regression():
    design = np.array([[0, 1, 0, 0], [0, -1, 1, 0],
                       [0, 0, -1, 1]], dtype=float)
    day = np.array([0., 2., 10., 30.])
    steps = np.array([[1.], [2.], [-1.]])
    series = np.zeros((4, 1))
    series[1:] = np.linalg.lstsq(design[:, 1:], steps, rcond=None)[0]
    expected_epochs = np.vstack([
        np.exp(-.5 * ((day[i] - day) / 5.) ** 2)
        @ series / np.exp(-.5 * ((day[i] - day) / 5.) ** 2).sum()
        for i in range(4)])
    assert np.allclose(_smooth_series(steps, design, day, 5.),
                       design @ expected_epochs)


def test_priors_and_unwrap_report_consistent_clean_stack():
    t, y, x = np.mgrid[:4, :5, :6]
    truth = 0.2 * t * x + 0.1 * y
    phase = wrap_phase(truth)
    priors = space_time_priors(phase, np.arange(4.),
                               params=SpaceTimeParams(time_win=2.))
    result, info = space_time_unwrap(
        phase, np.arange(4.), params=SpaceTimeParams(time_win=2.),
        return_info=True)
    assert priors.steps.shape == priors.smooth.shape
    assert info.observations == 4 and info.proven
    assert np.allclose(wrap_phase(result), phase)


@pytest.mark.parametrize("kwargs", [
    {"day": [0, 0, 2]}, {"day": [0, 1]},
    {"day": [0, 1, 2], "gate": "bad"},
])
def test_invalid_temporal_arguments(kwargs):
    with pytest.raises(ValueError):
        space_time_priors(np.zeros((3, 4, 4)), **kwargs)


def test_params_validate_ranges():
    with pytest.raises(ValueError):
        SpaceTimeParams(time_win=0)
