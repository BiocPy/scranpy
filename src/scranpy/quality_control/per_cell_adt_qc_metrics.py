from dataclasses import dataclass
from typing import Mapping, Optional

from biocframe import BiocFrame
from numpy import float64, int32, ndarray

from .. import cpphelpers as lib
from .._logging import logger
from ..types import MatrixTypes
from ..utils import to_logical, validate_and_tatamize_input
from .utils import create_pointer_array


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

        num_threads (int, optional): Number of threads to use. Defaults to 1.
        verbose (bool, optional): Display logs?. Defaults to False.
    """

    subsets: Optional[Mapping] = None
    num_threads: int = 1
    verbose: bool = False


def per_cell_adt_qc_metrics(
    input: MatrixTypes,
    options: PerCellAdtQcMetricsOptions = PerCellAdtQcMetricsOptions(),
) -> BiocFrame:
    """Compute per-cell quality control metrics for ADT data. This includes the total count for each cell, where low
    values are indicative of problems with library preparation or sequencing; the number of detected features per cell,
    where low values are indicative of problems with transcript capture; and the total count in particular feature
    subsets, typically isotype controls where high values are indicative of protein aggregates.

    Args:
        input (MatrixTypes):
            Matrix-like object containing cells in columns and features in rows, typically with "count" data.
            This should be a matrix class that can be converted into a :py:class:`~mattress.TatamiNumericPointer`.
            Developers may also provide the :py:class:`~mattress.TatamiNumericPointer` itself.

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
    x = validate_and_tatamize_input(input)

    if options.subsets is None:
        options.subsets = {}

    nr = x.nrow()
    nc = x.ncol()
    sums = ndarray((nc,), dtype=float64)
    detected = ndarray((nc,), dtype=int32)

    keys = list(options.subsets.keys())
    num_subsets = len(keys)
    collected_in = []
    collected_out = {}

    for i in range(num_subsets):
        in_arr = to_logical(options.subsets[keys[i]], nr)
        collected_in.append(in_arr)
        out_arr = ndarray((nc,), dtype=float64)
        collected_out[keys[i]] = out_arr

    subset_in = create_pointer_array(collected_in)
    subset_out = create_pointer_array(collected_out)
    if options.verbose is True:
        logger.info(subset_in)
        logger.info(subset_out)

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
            "subset_totals": BiocFrame(collected_out, number_of_rows=nc),
        }
    )
