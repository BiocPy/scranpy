from dataclasses import dataclass

import numpy as np
from mattress import TatamiNumericPointer

from .. import cpphelpers as lib
from ..types import MatrixTypes
from ..utils import to_logical

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


@dataclass
class FilterCellsArgs:
    """Arguments to filter cells -
    :py:meth:`~scranpy.quality_control.filter_cells.filter_cells`.

    Attributes:
        discard (bool): Whether to discard the cells listed in ``filter``.
            If False, the specified cells are retained instead, and all
            other cells are discarded. Defaults to True.
        verbose (bool, optional): Display logs?. Defaults to False.
    """

    discard: bool = True
    verbose: bool = False


def filter_cells(
    input: MatrixTypes, filter: np.ndarray, options: FilterCellsArgs = FilterCellsArgs()
) -> TatamiNumericPointer:
    """Filter out low-quality cells.

    Args:
        input (MatrixTypes): Input matrix, either as a
            :py:class:`~mattress.TatamiNumericPointer` or a supported matrix that
            can be converted into one.
        filter (np.ndarray): Boolean :py:class:`~numpy.ndarray` containing integer
            indices or booleans, specifying the columns of `input` to keep/discard.
        options (FilterCellsArgs): Optional parameters.

    Returns:
        TatamiNumericPointer: If `input` is a
        :py:class:`~mattress.TatamiNumericPointer`,
        a :py:class:`~mattress.TatamiNumericPointer` is returned
        containing the filtered matrix.
    """
    filter = to_logical(filter, input.ncol())

    if len(filter) != input.ncol():
        raise ValueError("Length of 'filter' should equal number of columns in 'x'")

    if not isinstance(input, TatamiNumericPointer):
        raise ValueError("Coming soon when `DelayedArray` support is implemented")

    outptr = lib.filter_cells(input.ptr, filter, options.discard)
    return TatamiNumericPointer(ptr=outptr, obj=input.obj)
