import numpy
import scranpy


def test_fit_variance_trend():
    x = numpy.random.rand(1000)
    y = 2**numpy.random.randn(1000)
    fitted, resids = scranpy.fit_variance_trend(x, y)
    assert numpy.allclose(y - fitted, resids)

    # Works with parallellization.
    fitted2, resids2 = scranpy.fit_variance_trend(x, y, num_threads=2)
    assert (fitted2 == fitted).all()
    assert (resids2 == resids).all()

    # Responds to the various options.
    fitted2, resids2 = scranpy.fit_variance_trend(x, y, use_min_width=True, min_width=0.5)
    assert (fitted != fitted2).any()

    fitted2, resids2 = scranpy.fit_variance_trend(x, y, transform=False)
    assert (fitted != fitted2).any()

    fitted2, resids2 = scranpy.fit_variance_trend(x, y, span=0.5)
    assert (fitted != fitted2).any()
