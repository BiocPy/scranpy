from typing import Sequence, Union

import numpy as np
from mattress import TatamiNumericPointer

from .. import cpphelpers as lib
from .._logging import logger
from ..types import MatrixTypes
from ..utils import to_logical

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def filter_cells(
    x: MatrixTypes, filter: np.ndarray, discard: bool = True
) -> TatamiNumericPointer:
    """Filter out low-quality cells.

    Args:
        x (MatrixTypes): Input matrix, either as a TatamiNumericPointer or
            something that can be converted into one.
        filter (np.ndarray): Boolean nd array containing integer
            indices or booleans, specifying the columns of `x` to keep/discard.
        discard (bool): Whether to discard the cells listed in `filter`.
            If `false`, the specified cells are retained instead, and all
            other cells are discarded.

    Returns:
        TatamiNumericPointer: If `x` is a TatamiNumericPointer,
        a TatamiNumericPointer is returned containing the filtered matrix.
    """
    if filter.dtype != np.bool_:
        filter = to_logical(filter, x.ncol())
    else:
        filter = filter.astype(np.uint8)

    if len(filter) != x.ncol():
        raise ValueError("length of 'filter' should equal number of columns in 'x'")

    if not isinstance(x, TatamiNumericPointer):
        raise ValueError("coming soon when DelayedArray support is implemented")

    outptr = lib.filter_cells(x.ptr, filter.ctypes.data, discard)
    return TatamiNumericPointer(ptr=outptr, obj=x.obj)
