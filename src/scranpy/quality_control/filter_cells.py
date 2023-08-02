from typing import Sequence

import numpy as np
from mattress import TatamiNumericPointer, tatamize

from .._logging import logger
from .. import cpphelpers as lib
from ..types import MatrixTypes
from ..utils import to_logical

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"

def filter_cells(x : MatrixTypes, filter : Sequence, discard : bool = False):
    if filter.dtype != np.bool_:
        filter = to_logical(filter, x.ncol())
    
    if len(filter) != x.ncol():
        raise ValueError("length of 'filter' should equal number of columns in 'x'")

    if not isinstance(x, TatamiNumericPointer):
        raise ValueError("coming soon when DelayedArray support is implemented")

    outptr = lib.filter_cells(x.ptr, filter.ctypes.data, discard)
    return TatamiNumericPointer(ptr=outptr, obj=x.obj)
