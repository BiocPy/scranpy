import numpy
import scranpy
import knncolle
import pytest


def test_subsample_by_neighbors():
    numpy.random.seed(10000)
    x = numpy.asfortranarray(numpy.random.randn(10, 1000))

    keep = scranpy.subsample_by_neighbors(x, num_neighbors=5, min_remaining=2)
    assert len(keep) < x.shape[1]
    assert keep.min() >= 0
    assert keep.max() < x.shape[1]

    keep2 = scranpy.subsample_by_neighbors(x, num_neighbors=10, min_remaining=2)
    assert len(keep2) < len(keep)
    assert keep.min() >= 0
    assert keep.max() < x.shape[1]

    # Same results when given some nearest neighbor results.
    idx = knncolle.build_index(knncolle.AnnoyParameters(), x.T)
    res = knncolle.find_knn(idx, num_neighbors=10)
    keep_alt = scranpy.subsample_by_neighbors(res, num_neighbors=10, min_remaining=2)
    assert (keep2 == keep_alt).all()

    with pytest.raises(Exception, match="should not be greater"):
        scranpy.subsample_by_neighbors(x, num_neighbors=10, min_remaining=1000)
