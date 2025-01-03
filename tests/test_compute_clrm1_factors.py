import numpy
import scranpy


def test_compute_clrm1_factors():
    x = numpy.random.rand(100, 1000)
    ref = numpy.expm1(numpy.log1p(x).mean(axis=0))
    obs = scranpy.compute_clrm1_factors(x)
    assert numpy.allclose(ref, obs)

    x[0,:] = 0
    x[99,:] = 0
    ref = numpy.expm1(numpy.log1p(x[1:99,:]).mean(axis=0))
    obs = scranpy.compute_clrm1_factors(x)
    assert numpy.allclose(ref, obs)
