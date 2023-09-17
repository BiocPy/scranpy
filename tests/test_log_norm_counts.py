import numpy as np
import delayedarray as da
from mattress import tatamize
from scranpy import (
    LogNormCountsOptions,
    log_norm_counts,
    CenterSizeFactorsOptions,
)

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def test_log_norm_counts(mock_data):
    x = mock_data.x
    y = tatamize(x)
    result = log_norm_counts(y)

    # Comparison to a reference.
    sf = x.sum(0)
    csf = sf / sf.mean()
    ref = np.log2(x[0, :] / csf + 1)
    assert np.allclose(result.row(0), ref)

    # Works without centering.
    result_uncentered = log_norm_counts(y, LogNormCountsOptions(center=False))
    assert np.allclose(result_uncentered.row(0), np.log2(x[0, :] / sf + 1))

    # Works with blocking.
    result_blocked = log_norm_counts(
        y,
        LogNormCountsOptions(
            center_size_factors_options=CenterSizeFactorsOptions(block=mock_data.block)
        ),
    )
    first_blocked = result_blocked.row(0)
    assert np.allclose(first_blocked, ref) is False

    # Same results after parallelization.
    result_parallel = log_norm_counts(y, LogNormCountsOptions(num_threads=3))
    assert (result.row(0) == result_parallel.row(0)).all()
    last = result.nrow() - 1
    assert (result.row(last) == result_parallel.row(last)).all()


def test_log_norm_counts_matrix(mock_data):
    x = mock_data.x

    ref = log_norm_counts(x, LogNormCountsOptions())
    assert isinstance(ref, da.DelayedArray)

    sf = x.sum(axis=0)
    out = log_norm_counts(x, LogNormCountsOptions(size_factors=sf))
    assert isinstance(out, da.DelayedArray)
    assert np.allclose(out[:, 0], ref[:, 0])

    sf = x.sum(axis=0)
    out = log_norm_counts(x, LogNormCountsOptions(size_factors=sf, center=False))
    assert isinstance(out, da.DelayedArray)
    assert np.allclose(out[:, 0], np.log(x[:, 0] / sf[0] + 1) / np.log(2))

    out = log_norm_counts(x, LogNormCountsOptions(size_factors=sf, delayed=False))
    assert isinstance(out, np.ndarray)
    assert np.allclose(out[:, 0], ref[:, 0])


def test_log_norm_counts_size_factors(mock_data):
    x = mock_data.x

    ref, sf = log_norm_counts(x, LogNormCountsOptions(with_size_factors=True))
    assert isinstance(ref, da.DelayedArray)
    assert isinstance(sf, np.ndarray)
    assert sf.shape[0] == x.shape[1]
