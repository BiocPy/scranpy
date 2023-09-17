import numpy as np
from scranpy import (
    combine_embeddings,
    CombineEmbeddingsOptions,
)

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def test_combine_embeddings_simple(mock_data):
    x = np.random.rand(200, 10)
    y = x * 50
    res = combine_embeddings([x, y])

    assert res.shape[0] == 200
    assert res.shape[1] == 20
    assert np.allclose(res[:, :10], x)
    assert np.allclose(res[:, 10:], x)

    # Works for different embeddings.
    x1 = np.random.rand(200, 10)
    x2 = np.random.rand(200, 5)
    x3 = np.random.rand(200, 20)
    res = combine_embeddings([x1, x2, x3])

    assert res.shape[0] == 200
    assert res.shape[1] == 35


def test_combine_embeddings_weighted(mock_data):
    x = np.random.rand(200, 10)
    y = x * 50
    res = combine_embeddings([x, y], CombineEmbeddingsOptions(weights=[5, 3]))

    assert res.shape[0] == 200
    assert res.shape[1] == 20
    assert np.allclose(res[:, :10] / 5, x)
    assert np.allclose(res[:, 10:] / 3, x)

    # Zero-weight embeddings are thrown away.
    x1 = np.random.rand(200, 10)
    x2 = np.random.rand(200, 5)
    x3 = np.random.rand(200, 20)

    res = combine_embeddings(
        [x1, x2, x3], CombineEmbeddingsOptions(weights=[1.5, 0, 0.5])
    )
    assert res.shape[0] == 200
    assert res.shape[1] == 30

    res = combine_embeddings([x1, x2, x3], CombineEmbeddingsOptions(weights=[0, 0, 1]))
    assert np.allclose(res, x3)

    empty = combine_embeddings(
        [x1, x2, x3], CombineEmbeddingsOptions(weights=[0, 0, 0])
    )
    assert empty.shape[0] == 200
    assert empty.shape[1] == 0
