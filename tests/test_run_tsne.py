import numpy
import knncolle
import scranpy
import warnings
import pytest


def test_run_tsne():
    x = numpy.random.rand(10, 500)

    embed = scranpy.run_tsne(x)
    assert embed.shape == (2, 500)

    again = scranpy.run_tsne(x)
    assert (embed == again).all() # check that it's reproducible.

    alt = scranpy.run_tsne(x, perplexity=20)
    assert alt.shape == (2, 500)
    assert (alt != embed).any() # check that perplexity has an effect.

    idx = knncolle.build_index(knncolle.AnnoyParameters(), x.T)
    res = knncolle.find_knn(idx, num_neighbors=scranpy.tsne_perplexity_to_neighbors(30))
    nnin = scranpy.run_tsne(res)
    assert (nnin == embed).all()

    res = knncolle.find_knn(idx, num_neighbors=30)
    with pytest.warns(match="not consistent with 'num_neighbors'"):
        scranpy.run_tsne(res)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scranpy.run_tsne(res, num_neighbors=30) # no warning should be emitted here.
