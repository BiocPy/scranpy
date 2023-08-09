import numpy as np
from mattress import TatamiNumericPointer

from .. import cpphelpers as lib
from ..types import MatrixTypes
from ..utils import to_logical
from .argtypes import FilterCellsArgs

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def filter_cells(
    input: MatrixTypes, filter: np.ndarray, options: FilterCellsArgs = FilterCellsArgs()
) -> TatamiNumericPointer:
    """Filter out low-quality cells.

    Args:
        input (MatrixTypes): Input matrix, either as a TatamiNumericPointer or
            something that can be converted into one.
        filter (np.ndarray): Boolean nd array containing integer
            indices or booleans, specifying the columns of `x` to keep/discard.
        options (FilterCellsArgs): additional arguments defined
            by `FilterCellsArgs`.

    Returns:
        TatamiNumericPointer: If `input` is a TatamiNumericPointer,
        a TatamiNumericPointer is returned containing the filtered matrix.
    """
    filter = to_logical(filter, input.ncol())

    if len(filter) != input.ncol():
        raise ValueError("length of 'filter' should equal number of columns in 'x'")

    if not isinstance(input, TatamiNumericPointer):
        raise ValueError("coming soon when DelayedArray support is implemented")

    outptr = lib.filter_cells(input.ptr, filter, options.discard)
    return TatamiNumericPointer(ptr=outptr, obj=input.obj)
