from scranpy import (
    hypergeometric_test,
    HypergeometricTestOptions,
)
import numpy


def test_hypergeometric_test_inputs():
    de_in_set = (numpy.random.rand(100) * 20).astype(numpy.int32)
    num_de = de_in_set + (numpy.random.rand(100) * 20).astype(numpy.int32)
    set_size = de_in_set + (numpy.random.rand(100) * 50).astype(numpy.int32)
    total_size = ((numpy.random.rand(100) + 1) * 50).astype(numpy.int32)

    # All vectors.
    p = hypergeometric_test(de_in_set, set_size, num_de, total_size)
    assert len(p) == 100
    assert (p >= 0).all()
    assert (p <= 1).all()

    # All scalars.
    ps = hypergeometric_test(de_in_set[0], set_size[0], num_de[0], total_size[0])
    assert p[0] == ps[0]

    # Partial scalars, vectors.
    pp = hypergeometric_test(de_in_set, set_size, num_de[0], total_size[0])
    assert len(p) == 100
    assert pp[0] == ps
    assert (pp != p).any()


def test_hypergeometric_test_options():
    de_in_set = (numpy.random.rand(100) * 20).astype(numpy.int32)
    set_size = de_in_set + (numpy.random.rand(100) * 20).astype(numpy.int32)

    lp = hypergeometric_test(
        de_in_set, set_size, 30, 1000, options=HypergeometricTestOptions(log=True)
    )
    assert len(lp) == 100
    assert (lp <= 0).all()

    up = hypergeometric_test(
        de_in_set,
        set_size,
        30,
        1000,
        options=HypergeometricTestOptions(upper_tail=True),
    )
    assert len(up) == 100
    assert (up >= 0).all()
    assert (up <= 1).all()

    ref = hypergeometric_test(de_in_set, set_size, 40, 500)
    par = hypergeometric_test(
        de_in_set, set_size, 40, 500, options=HypergeometricTestOptions(num_threads=3)
    )
    assert (ref == par).all()
