import numpy as np
from scranpy.dimensionality_reduction import run_pca

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def test_run_pca(mock_data):
    x = mock_data.x * 20
    res = run_pca(x, rank=10)

    mres = run_pca(x, rank=10, block=mock_data.block, block_method="project")
    assert np.allclose(mres.principal_components, res.principal_components) is False

    mres2 = run_pca(
        x, rank=10, block=mock_data.block, block_method="none", block_weights=False
    )
    assert np.allclose(mres2.principal_components, res.principal_components)

    rres = run_pca(x, rank=10, block=mock_data.block, block_method="regress")
    assert np.allclose(mres.principal_components, res.principal_components) is False
    assert np.allclose(rres.principal_components, res.principal_components) is False
