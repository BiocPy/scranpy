import numpy as np
from scranpy import ChooseHvgsOptions, choose_hvgs

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


def test_choose_hvgs():
    ex = np.random.rand(10000)
    out = choose_hvgs(ex, ChooseHvgsOptions(number=2000))

    # Not exactly equal due to potential ties.
    assert out.sum() >= 2000
    assert out.sum() < 10000

    assert ex[out].min() > ex[np.logical_not(out)].max()
