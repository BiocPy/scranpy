import numpy as np
from mattress import tatamize
from scranpy.normalization import log_norm_counts

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def test_log_norm_counts():
    x = np.random.rand(1000, 100)
    y = tatamize(x)
    result = log_norm_counts(y)

    # Comparison to a reference.
    sf = x.sum(0)
    csf = sf / sf.mean()
    ref = np.log2(x[0, :] / csf + 1)
    assert np.allclose(result.row(0), ref)

    # Works without centering.
    result_uncentered = log_norm_counts(y, center=False)
    assert np.allclose(result_uncentered.row(0), np.log2(x[0, :] / sf + 1))

    # Gives a different result with blocks.
    # TODO: factor out random block generation into a separate function!)
    block_levels = ["A", "B", "C"]
    block = []
    for i in range(x.shape[1]):
        block.append(block_levels[i % len(block_levels)])

    result_blocked = log_norm_counts(y, block=block)
    first_blocked = result_blocked.row(0)
    assert np.allclose(first_blocked, ref) is False
