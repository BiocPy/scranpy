import numpy
import knncolle
import scranpy
import warnings
import pytest


def test_run_umap():
    x = numpy.random.rand(10, 500)

    embed = scranpy.run_umap(x)
    assert embed.shape == (2, 500)

    again = scranpy.run_umap(x)
    assert (embed == again).all() # check that it's reproducible.

    alt = scranpy.run_umap(x, num_neighbors=20)
    assert alt.shape == (2, 500)
    assert (alt != embed).any() # check that perplexity has an effect.

    idx = knncolle.build_index(knncolle.AnnoyParameters(), x.T)
    res = knncolle.find_knn(idx, num_neighbors=15)
    nnin = scranpy.run_umap(res)
    assert (nnin == embed).all()
