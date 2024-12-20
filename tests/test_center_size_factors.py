from copy import deepcopy
import numpy
import scranpy

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def test_center_size_factors():
    sf = numpy.random.rand(100)
    out = scranpy.center_size_factors(sf)
    assert numpy.allclose(out, sf / sf.mean())
    assert numpy.allclose(out.mean(), 1)

    # Works in place.
    sf2 = deepcopy(sf)
    scranpy.center_size_factors(sf2, in_place=True)
    assert (sf2 == out).all()
    assert not numpy.allclose(sf2, sf)

    # Works with a little bit of blocking.
    block_levels = ["A", "B", "C"]
    block = []
    for i in range(len(sf)):
        block.append(block_levels[i % len(block_levels)])

    bout = scranpy.center_size_factors(sf, block=block)
    assert not numpy.allclose(bout, out)

    Amean = bout[[b == "A" for b in block]].mean()
    Bmean = bout[[b == "B" for b in block]].mean()
    Cmean = bout[[b == "C" for b in block]].mean()
    assert numpy.allclose(min(Amean, Bmean, Cmean), 1)

    bout = scranpy.center_size_factors(sf, block=block, mode="per-block")
    assert not numpy.allclose(bout, out)

    assert numpy.allclose(bout[[b == "A" for b in block]].mean(), 1)
    assert numpy.allclose(bout[[b == "B" for b in block]].mean(), 1)
    assert numpy.allclose(bout[[b == "C" for b in block]].mean(), 1)
