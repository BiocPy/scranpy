import numpy as np
from scranpy import RunPcaOptions, run_pca

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def test_run_pca(mock_data):
    x = mock_data.x * 20
    res = run_pca(x, RunPcaOptions(rank=10))

    mres = run_pca(
        x, RunPcaOptions(rank=10, block=mock_data.block, block_method="project")
    )
    assert np.allclose(mres.principal_components, res.principal_components) is False

    mres2 = run_pca(
        x,
        RunPcaOptions(
            rank=10, block=mock_data.block, block_method="none", block_weights=False
        ),
    )
    assert np.allclose(mres2.principal_components, res.principal_components)

    rres = run_pca(
        x, RunPcaOptions(rank=10, block=mock_data.block, block_method="regress")
    )
    assert np.allclose(mres.principal_components, res.principal_components) is False
    assert np.allclose(rres.principal_components, res.principal_components) is False

    # Subsetting behaves as expected.
    sub_ref = run_pca(x[20:50, :], RunPcaOptions(rank=10))
    sub_res = run_pca(x, RunPcaOptions(rank=10, subset=range(20, 50)))
    assert (sub_ref.principal_components == sub_res.principal_components).all()

    # Same results with multiple threads.
    resp = run_pca(x, RunPcaOptions(rank=10, num_threads=3))
    assert (res.principal_components == resp.principal_components).all()

    mresp = run_pca(
        x,
        RunPcaOptions(
            block=mock_data.block, block_method="project", rank=10, num_threads=3
        ),
    )
    assert (mres.principal_components == mresp.principal_components).all()

    rresp = run_pca(
        x,
        RunPcaOptions(
            block=mock_data.block, block_method="regress", rank=10, num_threads=3
        ),
    )
    assert (rres.principal_components == rresp.principal_components).all()
