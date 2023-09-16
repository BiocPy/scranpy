from dataclasses import dataclass

from mattress import TatamiNumericPointer
from numpy import logical_not, logical_or, logical_and, zeros, ones, uint8
from delayedarray import DelayedArray
from typing import Union, Sequence

from .. import cpphelpers as lib
from .._utils import to_logical

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

        intersect (bool): Whether to take the intersection or union of
            multiple ``filter`` arrays, to create a combined filtering
            array. Note that this is orthogonal to ``discard``.

        delayed (bool): Whether to force the filtering operation to be
            delayed. This reduces memory usage by avoiding unnecessary
            copies of the count matrix.
    """

    discard: bool = True
    intersect: bool = False
    delayed: bool = True


FilterArray = Union[Sequence[int], Sequence[bool]]


def filter_cells(
    input,
    filter: Union[tuple[FilterArray], FilterArray],
    options: FilterCellsOptions = FilterCellsOptions(),
):
    """Filter out low-quality cells, usually based on metrics and filter thresholds defined from the data, e.g.,
    :py:meth:`~scranpy.quality_control.rna.create_rna_qc_filter`.

    Args:
        input:
            Matrix-like object containing cells in columns and features in rows.
            This should be a matrix class that can be converted into a :py:class:`~mattress.TatamiNumericPointer`.
            Developers may also provide the :py:class:`~mattress.TatamiNumericPointer` itself.

        filter (FilterArray, list[FilterArray]):
            Array of integers containing indices to the columns of `input` to keep/discard.

            Alternatively, an array of booleans of length equal to the number of cells,
            specifying the columns of `input` to keep/discard.

            Alternatively, a tuple of such arrays, to be combined into a single
            filtering vector according to ``options.intersect``.

        options (FilterCellsOptions): Optional parameters.

    Returns:
        The filtered matrix, either as a :py:class:`~mattress.TatamiNumericPointer`
        if ``input`` is also a :py:class:`~mattress.TatamiNumericPointer`; as a
        :py:class:`~delayedarray.DelayedArray`, if ``input`` is array-like and
        ``delayed = True``; or an object of the same type as ``input`` otherwise.
    """
    is_ptr = isinstance(input, TatamiNumericPointer)

    ncols = None
    if is_ptr:
        ncols = input.ncol()
    else:
        ncols = input.shape[1]

    if isinstance(filter, tuple):
        # Duplicates logic in the C++ libraries, oh well.
        if not options.intersect:
            combined = zeros(ncols, dtype=bool)
            for f in filter:
                ll = to_logical(f, ncols, dtype=bool)
                combined = logical_or(combined, ll)
        else:
            combined = ones(ncols, dtype=bool)
            for f in filter:
                ll = to_logical(f, ncols, dtype=bool)
                combined = logical_and(combined, ll)
        filter = combined.astype(uint8)
    else:
        filter = to_logical(filter, ncols)

    if is_ptr:
        outptr = lib.filter_cells(input.ptr, filter, options.discard)
        return TatamiNumericPointer(ptr=outptr, obj=input.obj)
    else:
        if options.delayed and not isinstance(input, DelayedArray):
            input = DelayedArray(input)
        if options.discard:
            filter = filter == 0
        else:
            filter = filter != 0
        return input[:, filter]
