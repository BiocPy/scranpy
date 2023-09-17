import numpy as np
from copy import deepcopy
from mattress import tatamize
from scranpy import CenterSizeFactorsOptions, center_size_factors

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def test_center_size_factors():
    sf = np.random.rand(100)
    out = center_size_factors(sf)
    assert np.allclose(out, sf / sf.mean())
    assert np.allclose(out.mean(), 1)

    # Works in place.
    sf2 = deepcopy(sf)
    center_size_factors(sf2, CenterSizeFactorsOptions(in_place=True))
    assert np.allclose(sf2, out)
    assert not np.allclose(sf2, sf)

    # Works with a little bit of blocking.
    block_levels = ["A", "B", "C"]
    block = []
    for i in range(len(sf)):
        block.append(block_levels[i % len(block_levels)])

    bout = center_size_factors(sf, CenterSizeFactorsOptions(block=block))
    assert not np.allclose(bout, out)

    Amean = bout[[b == "A" for b in block]].mean()
    Bmean = bout[[b == "B" for b in block]].mean()
    Cmean = bout[[b == "C" for b in block]].mean()
    assert np.allclose(min(Amean, Bmean, Cmean), 1)

    # Works with sanitization; all-zero size factors are replaced with 1's.
    sf = np.zeros(10)
    out = center_size_factors(sf, CenterSizeFactorsOptions(allow_zeros=True))
    assert np.allclose(out, np.ones(10))
