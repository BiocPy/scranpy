import numpy as np
from scranpy.feature_selection import model_gene_variances

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


def test_numpy_no_blocks(mock_data):
    # No blocks
    x = mock_data.x
    res = model_gene_variances(x)

    assert res is not None
    assert res.column("means") is not None
    assert res.column("variances") is not None
    assert res.column("fitted") is not None
    assert res.column("residuals") is not None


def test_numpy_blocks(mock_data):
    x = mock_data.x

    resblock = model_gene_variances(x, block=mock_data.block)

    assert resblock is not None
    assert resblock.column("means") is not None
    assert resblock.column("variances") is not None
    assert resblock.column("fitted") is not None
    assert resblock.column("residuals") is not None
    assert resblock.column("per_block") is not None

    assert resblock.column("per_block").shape[1] == 3
