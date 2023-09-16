import numpy as np
from mattress import tatamize
from scranpy.quality_control import FilterCellsOptions, filter_cells
import delayedarray as da

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def test_filter_cells(mock_data):
    x = mock_data.x
    y = tatamize(x)
    filtered = filter_cells(
        y, filter=np.asarray([2, 4, 6, 8]), options=FilterCellsOptions(discard=False)
    )

    assert filtered.ncol() == 4
    assert (filtered.column(0) == y.column(2)).all()
    assert (filtered.column(3) == y.column(8)).all()

    filtered = filter_cells(y, np.asarray([0, 1, 2, 3, 4]))
    assert filtered.ncol() == y.ncol() - 5
    assert (filtered.column(0) == y.column(5)).all()

    keep = np.zeros(y.ncol(), dtype=np.bool_)
    for i in range(0, y.ncol(), 3):
        keep[i] = True

    filtered = filter_cells(y, filter=keep, options=FilterCellsOptions(discard=False))
    assert filtered.ncol() == keep.sum()
    filtered = filter_cells(y, filter=keep)
    assert filtered.ncol() == y.ncol() - keep.sum()


def test_filter_cells_by_matrix(mock_data):
    x = mock_data.x
    out = filter_cells(
        x, filter=np.asarray([2, 4, 6, 8]), options=FilterCellsOptions(discard=True)
    )
    assert isinstance(out, da.DelayedArray)
    assert out.shape == (x.shape[0], x.shape[1] - 4)

    out = filter_cells(
        x,
        filter=np.asarray([2, 4, 6, 8]),
        options=FilterCellsOptions(discard=True, delayed=False),
    )
    assert isinstance(out, np.ndarray)


def test_filter_cells_multiple_arrays(mock_data):
    x = mock_data.x
    filtered = filter_cells(x, filter=(np.array([2, 4, 6, 8]), [1,3,5,7]))
    assert filtered.shape[0] == x.shape[0]
    assert filtered.shape[1] == x.shape[1] - 8
    assert (x[:,0] == filtered[:,0]).all()
    assert (x[:,9] == filtered[:,1]).all()

    filtered = filter_cells(
        x, 
        filter=(np.asarray([2, 4, 6, 8]), [1,2,3,4,5]),
        options=FilterCellsOptions(intersect=True)
    )
    assert filtered.shape[0] == x.shape[0]
    assert filtered.shape[1] == x.shape[1] - 2
    assert (x[:,1] == filtered[:,1]).all()
    assert (x[:,3] == filtered[:,2]).all()
    assert (x[:,5] == filtered[:,3]).all()
