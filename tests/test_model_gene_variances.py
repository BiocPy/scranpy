import numpy as np
from scranpy.feature_selection import model_gene_variances

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


def test_numpy_no_blocks():
    # No blocks
    x = np.random.rand(1000, 100) * 20
    res = model_gene_variances(x)

    assert res is not None
    assert res.column("means") is not None
    assert res.column("variances") is not None
    assert res.column("fitted") is not None
    assert res.column("residuals") is not None


def test_numpy_blocks():
    x = np.random.rand(1000, 100) * 20
    # + blocks
    block_levels = ["A", "B", "C"]
    block = []
    for i in range(x.shape[1]):
        block.append(block_levels[i % len(block_levels)])

    resblock = model_gene_variances(x, block=block)

    assert resblock is not None
    assert resblock.column("means") is not None
    assert resblock.column("variances") is not None
    assert resblock.column("fitted") is not None
    assert resblock.column("residuals") is not None
    assert resblock.column("per_block") is not None

    assert resblock.column("per_block").shape[1] == len(block_levels)
