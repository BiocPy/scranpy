from dataclasses import dataclass

from mattress import TatamiNumericPointer
from numpy import ndarray

from .. import cpphelpers as lib
from ..types import MatrixTypes
from ..utils import to_logical

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


@dataclass
class FilterCellsOptions:
    """Optional arguments for :py:meth:`~scranpy.quality_control.filter_cells.filter_cells`.

    Attributes:
        discard (bool): Whether to discard the cells listed in ``filter``.
            If False, the specified cells are retained instead, and all
            other cells are discarded. Defaults to True.
        verbose (bool, optional): Whether to print logs. Defaults to False.
    """

    discard: bool = True
    verbose: bool = False


def filter_cells(
    input: MatrixTypes,
    filter: ndarray,
    options: FilterCellsOptions = FilterCellsOptions(),
) -> TatamiNumericPointer:
    """Filter out low-quality cells, usually based on metrics and filter thresholds defined from the data, e.g.,
    :py:meth:`~scranpy.quality_control.rna.create_rna_qc_filter`.

    Args:
        input (MatrixTypes):
            Matrix-like object containing cells in columns and features in rows.
            This should be a matrix class that can be converted into a :py:class:`~mattress.TatamiNumericPointer`.
            Developers may also provide the :py:class:`~mattress.TatamiNumericPointer` itself.

        filter (Sequence[int] | Sequence[bool]):
            Array of integers containing indices to the columns of `input` to keep/discard.

            Alternatively, an array of booleans of length equal to the number of cells,
            specifying the columns of `input` to keep/discard.

        options (FilterCellsOptions): Optional parameters.

    Returns:
        TatamiNumericPointer: If ``input`` is a
        :py:class:`~mattress.TatamiNumericPointer`,
        a :py:class:`~mattress.TatamiNumericPointer` is returned
        containing the filtered matrix.
    """
    filter = to_logical(filter, input.ncol())

    if len(filter) != input.ncol():
        raise ValueError("length of 'filter' should equal number of columns in 'x'")

    if not isinstance(input, TatamiNumericPointer):
        raise ValueError("Coming soon when `DelayedArray` support is implemented")

    outptr = lib.filter_cells(input.ptr, filter, options.discard)
    return TatamiNumericPointer(ptr=outptr, obj=input.obj)
