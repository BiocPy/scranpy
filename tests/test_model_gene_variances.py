import scranpy
import numpy

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


def test_model_gene_variances_basic():
    x = numpy.random.rand(1000, 100) * 10
    out = scranpy.model_gene_variances(x)
    assert numpy.allclose(out.mean, x.mean(axis=1))
    assert numpy.allclose(out.variance, x.var(axis=1, ddof=1))
    assert numpy.allclose(out.variance - out.fitted, out.residual)
    assert out.per_block is None

    fitted, resid = scranpy.fit_variance_trend(out.mean, out.variance)
    assert (out.fitted == fitted).all()
    assert (out.residual == resid).all()

    # Responds to trend-fitting options. 
    out2 = scranpy.model_gene_variances(x, span=0.5)
    assert (out.variance == out2.variance).all()
    assert (out.mean == out2.mean).all()
    assert (out.fitted != out2.fitted).any()

    fitted2, resid2 = scranpy.fit_variance_trend(out.mean, out.variance, span=0.5)
    assert (out2.fitted == fitted2).all()
    assert (out2.residual == resid2).all()

    # Generates a DF correctly.
    df = out.to_biocframe()
    assert df.shape == (1000, 4)
    assert (df.get_column("mean") == out.mean).all()


def test_model_gene_variances_blocked():
    x = numpy.random.rand(1000, 100) * 10
    block = (numpy.random.rand(x.shape[1]) * 3).astype(numpy.int32)
    out = scranpy.model_gene_variances(x, block=block, block_weight_policy="equal")

    for b in range(3):
        sub = x[:,block == b]
        current = out.per_block[b]
        assert numpy.allclose(current.mean, sub.mean(axis=1))
        assert numpy.allclose(current.variance, sub.var(axis=1, ddof=1))
        assert numpy.allclose(current.variance - current.fitted, current.residual)

    assert numpy.allclose(out.mean, (out.per_block[0].mean + out.per_block[1].mean + out.per_block[2].mean)/3)
    assert numpy.allclose(out.variance, (out.per_block[0].variance + out.per_block[1].variance + out.per_block[2].variance)/3)
    assert numpy.allclose(out.fitted, (out.per_block[0].fitted + out.per_block[1].fitted + out.per_block[2].fitted)/3)
    assert numpy.allclose(out.residual, (out.per_block[0].residual + out.per_block[1].residual + out.per_block[2].residual)/3)

    # Generates a DF correctly.
    df = out.to_biocframe()
    assert df.shape == (1000, 4)
    assert (df.get_column("residual") == out.residual).all()

    df = out.to_biocframe(include_per_block=True)
    assert df.shape == (1000, 5)
    assert df.get_column("per_block").get_column_names().as_list() == ["0", "1", "2"]
    assert (df.get_column("per_block").get_column("2").get_column("variance") == out.per_block[2].variance).all()
