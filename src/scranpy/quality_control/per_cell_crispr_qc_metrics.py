from dataclasses import dataclass

from biocframe import BiocFrame
from numpy import float64, int32, ndarray
from typing import Union

from .. import cpphelpers as lib
from .._utils import tatamize_input, MatrixTypes


@dataclass
class PerCellCrisprQcMetricsOptions:
    """Optional arguments for :py:meth:`~scranpy.quality_control.per_cell_crispr_qc_metrics.per_cell_crispr_qc_metrics`.

    Attributes:
        assay_type (Union[int, str]):
            Assay to use from ``input`` if it is a 
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.

        num_threads (int, optional): Number of threads to use. Defaults to 1.

        verbose (bool, optional): Display logs?. Defaults to False.
    """

    assay_type: Union[int, str] = 0
    num_threads: int = 1
    verbose: bool = False


def per_cell_crispr_qc_metrics(
    input: MatrixTypes,
    options: PerCellCrisprQcMetricsOptions = PerCellCrisprQcMetricsOptions(),
) -> BiocFrame:
    """Compute per-cell quality control metrics for CRISPR data. This includes the total count for each cell, where low
    values are indicative of unsuccessful transfection or problems with library preparation or sequencing; the number of
    detected guides per cell, where high values represent multiple transfections; the proportion of counts in the most
    abundant guide construct, where low values indicate that the cell was transfected with multiple guides.  The
    identity of the most abundant guide is also reported.

    Args:
        input (MatrixTypes): Matrix-like object where rows are features and columns are cells, typically containing
            expression values of some kind. This should be a matrix class that can be converted into a
            :py:class:`~mattress.TatamiNumericPointer.TatamiNumericPointer`. 

            Alternatively, a :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`
            containing such a matrix in its assays.

            Developers may also provide a :py:class:`~mattress.TatamiNumericPointer.TatamiNumericPointer` directly.

        options (PerCellCrisprQcMetricsOptions): Optional parameters.

    Raises:
        TypeError: If ``input`` is not an expected matrix type.

    Returns:
        BiocFrame:
            A data frame containing one row per cell and the following fields -
            ``"sums"``, the total count for each cell;
            ``"detected"``, the number of detected features for each cell;
            ``"max_proportion"``, the proportion of counts in the most abundant guide;
            and ``"max_index"``, the row index of the most abundant guide.
    """
    x = tatamize_input(input, options.assay_type)

    x.nrow()
    nc = x.ncol()
    sums = ndarray((nc,), dtype=float64)
    detected = ndarray((nc,), dtype=int32)
    max_prop = ndarray((nc,), dtype=float64)
    max_index = ndarray((nc,), dtype=int32)

    lib.per_cell_crispr_qc_metrics(
        x.ptr,
        sums,
        detected,
        max_prop,
        max_index,
        options.num_threads,
    )

    return BiocFrame(
        {
            "sums": sums,
            "detected": detected,
            "max_proportion": max_prop,
            "max_index": max_index,
        }
    )
