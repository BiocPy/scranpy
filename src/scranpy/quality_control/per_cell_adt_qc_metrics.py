from dataclasses import dataclass
from typing import Mapping, Optional, Union, Sequence

from biocframe import BiocFrame
from numpy import float64, int32, ndarray

from .. import cpphelpers as lib
from .._utils import to_logical, tatamize_input, create_pointer_array, MatrixTypes
from ._utils import create_subset_buffers, create_subset_frame


@dataclass
class PerCellAdtQcMetricsOptions:
    """Optional arguments for :py:meth:`~scranpy.quality_control.per_cell_adt_qc_metrics.per_cell_adt_qc_metrics`.

    Attributes:
        subsets (Mapping, optional): Dictionary of feature subsets.
            Each key is the name of the subset and each value is an array.

            Each array may contain integer indices to the rows of `input` belonging to the subset.
            Alternatively, each array is of length equal to the number of rows in ``input``
            and contains booleans specifying that the corresponding row belongs to the subset.

            Defaults to {}.

        assay_type (Union[int, str]):
            Assay to use from ``input`` if it is a
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.

        cell_names (Sequence[str], optional):
            Sequence of cell names of length equal to the number of columns  in ``input``.
            If provided, this is used as the row names of the output data frames.

        num_threads (int, optional): Number of threads to use. Defaults to 1.
    """

    subsets: Optional[Mapping] = None
    assay_type: Union[int, str] = 0
    cell_names: Optional[Sequence[str]] = None
    num_threads: int = 1


def per_cell_adt_qc_metrics(
    input: MatrixTypes,
    options: PerCellAdtQcMetricsOptions = PerCellAdtQcMetricsOptions(),
) -> BiocFrame:
    """Compute per-cell quality control metrics for ADT data. This includes the number of detected tags per cell, where
    low values are indicative of problems with transcript capture; and the total count for particular tag subsets,
    typically isotype controls where high values are indicative of protein aggregates. We also report the total count
    for each cell for diagnostic purposes.

    Args:
        input (MatrixTypes): Matrix-like object where rows are features and columns are cells, typically containing
            expression values of some kind. This should be a matrix class that can be converted into a
            :py:class:`~mattress.TatamiNumericPointer.TatamiNumericPointer`.

            Alternatively, a :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`
            containing such a matrix in its assays.

            Developers may also provide a :py:class:`~mattress.TatamiNumericPointer.TatamiNumericPointer` directly.

        options (PerCellAdtQcMetricsOptions): Optional parameters.

    Raises:
        TypeError: If ``input`` is not an expected matrix type.

    Returns:
        BiocFrame:
            A data frame containing one row per cell and the following fields -
            ``"sums"``, the total count for each cell;
            ``"detected"``, the number of detected features for each cell;
            and ``"subset_totals"``, a nested BiocFrame where each column is named
            after an entry in ``subsets`` and contains the proportion of counts in that subset.
    """
    x = tatamize_input(input, options.assay_type)

    if options.subsets is None:
        options.subsets = {}

    nr = x.nrow()
    nc = x.ncol()
    sums = ndarray((nc,), dtype=float64)
    detected = ndarray((nc,), dtype=int32)

    sub_keys = list(options.subsets.keys())
    num_subsets = len(sub_keys)
    collected_in = []
    for i, s in enumerate(sub_keys):
        in_arr = to_logical(options.subsets[s], nr)
        collected_in.append(in_arr)
    subset_in = create_pointer_array(collected_in)

    collected_out, subset_out = create_subset_buffers(nc, num_subsets)

    lib.per_cell_adt_qc_metrics(
        x.ptr,
        num_subsets,
        subset_in.ctypes.data,
        sums,
        detected,
        subset_out.ctypes.data,
        options.num_threads,
    )

    return BiocFrame(
        {
            "sums": sums,
            "detected": detected,
            "subset_totals": create_subset_frame(
                column_names=sub_keys,
                columns=collected_out,
                num_rows=nc,
            ),
        },
        row_names=options.cell_names,
    )
