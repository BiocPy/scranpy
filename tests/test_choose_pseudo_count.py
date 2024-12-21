import scranpy
import numpy

def test_choose_pseudo_count():
    sf = 2**(numpy.random.randn(100) * 2)
    out = scranpy.choose_pseudo_count(sf)
    assert out > 0
    assert out < scranpy.choose_pseudo_count(sf, quantile = 0.01)
    assert out < scranpy.choose_pseudo_count(sf, max_bias = 0.5)
