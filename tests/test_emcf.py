import numpy as np
import pytest

from parvaneh import wrap_phase
from parvaneh.emcf import (emcf_cycles, emcf_links, emcf_unwrap,
                           emcf_unwrap_interferograms)


def test_hop_three_links_and_triangles_close():
    links = emcf_links(5)
    assert links.shape == (9, 2)
    assert len(emcf_cycles(5)) == 5
    epoch = np.arange(5.0)
    values = epoch[links[:, 1]] - epoch[links[:, 0]]
    for edges, signs in emcf_cycles(5):
        assert np.dot(signs, values[edges]) == 0


def test_stack_and_preformed_interferograms_are_equivalent():
    t, y, x = np.mgrid[:4, :5, :6]
    stack = wrap_phase(0.3 * t * x + 0.15 * y)
    links = emcf_links(4)
    ifg = wrap_phase(stack[links[:, 1]] - stack[links[:, 0]])
    direct, direct_info = emcf_unwrap(stack, links, return_info=True)
    formed, formed_info = emcf_unwrap_interferograms(
        ifg, links, return_info=True)
    assert np.allclose(direct, formed)
    assert np.allclose(wrap_phase(direct), ifg)
    assert direct_info.network == formed_info.network == "hop3"
    assert direct_info.proven and formed_info.proven


def test_mask_complex_input_and_info():
    stack = np.ones((4, 4, 5), dtype=complex)
    mask = np.ones((4, 5), dtype=bool)
    mask[1, 2] = False
    result, info = emcf_unwrap(stack, mask=mask, return_info=True)
    assert np.isnan(result[:, 1, 2]).all()
    assert info.pixels == mask.sum()
    assert info.interferograms == emcf_links(4).shape[0]


@pytest.mark.parametrize("links", [
    [[0, 1], [0, 1]], [[1, 0]], [[0, 1.5]], [],
])
def test_invalid_links_are_rejected(links):
    with pytest.raises(ValueError):
        emcf_unwrap_interferograms(np.zeros((max(1, len(links)), 3, 3)), links)
