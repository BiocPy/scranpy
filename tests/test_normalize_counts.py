import numpy
import delayedarray
import mattress
import scranpy

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def test_normalize_counts():
    x = numpy.random.rand(1000, 100)
    sf = numpy.random.rand(100)
    ptr = mattress.initialize(x)

    y = scranpy.normalize_counts(x, sf)
    ptr2 = scranpy.normalize_counts(ptr, sf)
    assert numpy.isclose(y[0,:], ptr2.row(0)).all()
    assert numpy.isclose(y[:,99], ptr2.column(99)).all()

    y = scranpy.normalize_counts(x, sf, log=False)
    ptr2 = scranpy.normalize_counts(ptr, sf, log=False)
    assert numpy.isclose(y[0,:], ptr2.row(0)).all()
    assert numpy.isclose(y[:,99], ptr2.column(99)).all()

    y = scranpy.normalize_counts(x, sf, log_base=10)
    ptr2 = scranpy.normalize_counts(ptr, sf, log_base=10)
    assert numpy.isclose(y[0,:], ptr2.row(0)).all()
    assert numpy.isclose(y[:,99], ptr2.column(99)).all()

    y = scranpy.normalize_counts(x, sf, pseudo_count=3)
    ptr2 = scranpy.normalize_counts(ptr, sf, pseudo_count=3)
    assert numpy.isclose(y[0,:], ptr2.row(0)).all()
    assert numpy.isclose(y[:,99], ptr2.column(99)).all()

    y = scranpy.normalize_counts(x, sf, pseudo_count=3, preserve_sparsity=True)
    ptr2 = scranpy.normalize_counts(ptr, sf, pseudo_count=3, preserve_sparsity=True)
    assert numpy.isclose(y[0,:], ptr2.row(0)).all()
    assert numpy.isclose(y[:,99], ptr2.column(99)).all()
