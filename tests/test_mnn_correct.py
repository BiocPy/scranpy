import numpy as np
from scranpy import MnnCorrectOptions, mnn_correct
import pytest as pt

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def test_mnn_correct():
    x = np.random.rand(1000, 10)

    batch = []
    levels = ["A", "B", "C"]
    for i in range(x.shape[0]):
        batch.append(levels[i % 3])

    res = mnn_correct(x, batch)
    assert np.allclose(res.corrected, x) is False
    assert sorted(res.merge_order) == levels
    assert len(res.num_pairs) == 2

    # Set an alternative ordering.
    altorder = ["C", "B", "A"]
    res2 = mnn_correct(x, batch, MnnCorrectOptions(order=altorder))
    assert np.allclose(res2.corrected, x) is False
    assert res2.merge_order == altorder
    assert len(res.num_pairs) == 2

    # Check errors.
    with pt.raises(ValueError, match="same values"):
        mnn_correct(x, batch, MnnCorrectOptions(order=["a", "b", "c"]))
    with pt.raises(ValueError, match="duplicate"):
        mnn_correct(x, batch, MnnCorrectOptions(order=["A", "A", "C"]))
    with pt.raises(ValueError, match="number of batches"):
        mnn_correct(x, batch, MnnCorrectOptions(order=["A", "B"]))
    with pt.raises(RuntimeError, match="reference policy"):
        mnn_correct(x, batch, MnnCorrectOptions(reference_policy="WHEE"))
