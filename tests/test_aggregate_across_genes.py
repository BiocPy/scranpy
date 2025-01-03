import scranpy
import numpy
import biocutils
import pytest


def test_aggregate_across_genes_unweighted():
    x = numpy.random.rand(1000, 100)

    sets = [
        (numpy.random.rand(20) * x.shape[0]).astype(numpy.int32),
        (numpy.random.rand(10) * x.shape[0]).astype(numpy.int32),
        (numpy.random.rand(500) * x.shape[0]).astype(numpy.int32)
    ]

    agg = scranpy.aggregate_across_genes(x, sets)
    for i, ss in enumerate(sets):
        assert numpy.allclose(agg[i], x[ss,:].sum(axis=0))

    agg = scranpy.aggregate_across_genes(x, sets, average=True)
    for i, ss in enumerate(sets):
        assert numpy.allclose(agg[i], x[ss,:].mean(axis=0))

    # Works with names.
    names = ["foo", "bar", "whee"]
    agg = scranpy.aggregate_across_genes(x, biocutils.NamedList(sets, names))
    assert agg.get_names().as_list() == names


def test_aggregate_across_genes_weighted():
    x = numpy.random.rand(1000, 100)

    sets = [
        (
            (numpy.random.rand(20) * x.shape[0]).astype(numpy.int32),
            numpy.random.randn(20)
        ),
        (
            (numpy.random.rand(10) * x.shape[0]).astype(numpy.int32),
            numpy.random.randn(10)
        ),
        (
            (numpy.random.rand(500) * x.shape[0]).astype(numpy.int32),
            numpy.random.randn(500)
        )
    ]

    agg = scranpy.aggregate_across_genes(x, sets)
    for i, ss in enumerate(sets):
        assert numpy.allclose(agg[i], (x[ss[0],:].T * ss[1]).sum(axis=1))

    agg = scranpy.aggregate_across_genes(x, sets, average=True)
    for i, ss in enumerate(sets):
        assert numpy.allclose(agg[i], (x[ss[0],:].T * ss[1]).sum(axis=1) / ss[1].sum())

    with pytest.raises(Exception, match = "equal length"):
        scranpy.aggregate_across_genes(x, [([0], [1,2,3])])
