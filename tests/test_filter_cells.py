import numpy as np
from mattress import tatamize
from scranpy.quality_control import filter_cells

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def test_filter_cells(mock_data):
    x = mock_data.x
    y = tatamize(x)
    filtered = filter_cells(y, np.asarray([2, 4, 6, 8]), discard=False)

    assert filtered.ncol() == 4
    assert (filtered.column(0) == y.column(2)).all()
    assert (filtered.column(3) == y.column(8)).all()

    filtered = filter_cells(y, np.asarray([0, 1, 2, 3, 4]))
    assert filtered.ncol() == y.ncol() - 5
    assert (filtered.column(0) == y.column(5)).all()

    keep = np.zeros(y.ncol(), dtype=np.bool_)
    for i in range(0, y.ncol(), 3):
        keep[i] = True

    filtered = filter_cells(y, keep, discard=False)
    assert filtered.ncol() == keep.sum()
    filtered = filter_cells(y, keep)
    assert filtered.ncol() == y.ncol() - keep.sum()
