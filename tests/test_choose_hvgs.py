import numpy as np
from scranpy.feature_selection import ChooseHvgArgs, choose_hvgs

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


def test_choose_hvgs():
    ex = np.random.rand(10000)
    out = choose_hvgs(ex, ChooseHvgArgs(number=2000))

    # Not exactly equal due to potential ties.
    assert out.sum() >= 2000
    assert out.sum() < 10000

    assert ex[out].min() > ex[np.logical_not(out)].max()
