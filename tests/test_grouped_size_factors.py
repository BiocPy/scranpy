import scranpy
import numpy


def test_grouped_size_factors_simple():
    x = numpy.random.rand(100, 50)
    x[1:10,:10] = 1000
    x[1:10,10:20] = 2000
    x[1:10,20:30] = 3000
    x[1:10,30:40] = 4000
    x[1:10,40:] = 5000
    clusters = [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10 + [4] * 10

    out = scranpy.grouped_size_factors(x, options=scranpy.GroupedSizeFactorsOptions(groups=clusters))
    assert len(out) == 50
    assert (out > 0).all()
    
    out = scranpy.grouped_size_factors(x)
    assert len(out) == 50
    assert (out > 0).all()

    # Same results with some initial factors.
    out2 = scranpy.grouped_size_factors(x, options=scranpy.GroupedSizeFactorsOptions(initial_factors=x.sum(axis=0)))
    assert (out == out2).all()
