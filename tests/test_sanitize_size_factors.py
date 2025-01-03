import numpy
import scranpy


def test_sanitize_size_factors():
    f = numpy.random.rand(100)
    f[0] = 0
    f[1] = numpy.nan
    f[2] = numpy.inf
    f[3] = -1

    cf = scranpy.sanitize_size_factors(f)
    assert numpy.isfinite(cf).all()
    assert (cf > 0).all()
    assert (cf[4:] == f[4:]).all()

    scranpy.sanitize_size_factors(f, in_place=True)
    assert numpy.isfinite(f).all()
    assert (f > 0).all()
