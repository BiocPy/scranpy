import numpy as np
from mattress import tatamize

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def test_numpy():
    y = np.random.rand(1000, 100)
    ptr = tatamize(y)
    assert all(ptr.row(0) == y[0, :])
    assert all(ptr.column(1) == y[:, 1])


def test_numpy_with_order():
    y = np.ones((2, 3), order='F')
    ptr = tatamize(y, order="F")
    assert all(ptr.row(0) == y[0, :])
    assert all(ptr.column(1) == y[:, 1])


def test_numpy_with_dtype():
    y = (np.random.rand(50, 12) * 100).astype("i8")
    ptr = tatamize(y)
    assert all(ptr.row(0) == y[0, :])
    assert all(ptr.column(1) == y[:, 1])
