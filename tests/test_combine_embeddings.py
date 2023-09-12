import numpy as np
from scranpy.dimensionality_reduction import combine_embeddings

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def test_combine_embeddings(mock_data):
    x = np.random.rand(200, 10)
    y = x * 50
    res = combine_embeddings([x, y])

    assert res.shape[0] == 200
    assert res.shape[1] == 20
    assert np.allclose(res[:,:10], x)
    assert np.allclose(res[:,10:], x)

    # Works for different embeddings.
    x1 = np.random.rand(200, 10)
    x2 = np.random.rand(200, 5) 
    x3 = np.random.rand(200, 20)
    res = combine_embeddings([x1, x2, x3])

    assert res.shape[0] == 200
    assert res.shape[1] == 35

