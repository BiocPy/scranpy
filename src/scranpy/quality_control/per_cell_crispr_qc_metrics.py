from dataclasses import dataclass
from typing import Optional, Sequence, Union

from biocframe import BiocFrame
from numpy import float64, int32, zeros

from .. import _cpphelpers as lib
from .._utils import MatrixTypes, tatamize_input


@dataclass
class PerCellCrisprQcMetricsOptions:
    """Optional arguments for :py:meth:`~scranpy.quality_control.per_cell_crispr_qc_metrics.per_cell_crispr_qc_metrics`.

    Attributes:
        assay_type:
            Assay to use from ``input`` if it is a
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.

        cell_names:
            Sequence of cell names of length equal to the number of columns  in ``input``.
            If provided, this is used as the row names of the output data frames.

        num_threads: 
            Number of threads to use. Defaults to 1.
    """

    assay_type: Union[int, str] = 0
    cell_names: Optional[Sequence[str]] = None
    num_threads: int = 1


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
        input: 
            Matrix-like object where rows are features and columns are cells, typically containing
            expression values of some kind. This should be a matrix class that can be converted into a
            :py:class:`~mattress.TatamiNumericPointer.TatamiNumericPointer`.

            Alternatively, a :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`
            containing such a matrix in its assays.

            Developers may also provide a :py:class:`~mattress.TatamiNumericPointer.TatamiNumericPointer` directly.

        options: 
            Optional parameters.

    Raises:
        TypeError: 
            If ``input`` is not an expected matrix type.

    Returns:
        A data frame containing one row per cell and the following fields -
        ``"sums"``, the total count for each cell;
        ``"detected"``, the number of detected features for each cell;
        ``"max_proportion"``, the proportion of counts in the most abundant guide;
        and ``"max_index"``, the row index of the most abundant guide.
    """
    x = tatamize_input(input, options.assay_type)

    x.nrow()
    nc = x.ncol()
    sums = zeros((nc,), dtype=float64)
    detected = zeros((nc,), dtype=int32)
    max_prop = zeros((nc,), dtype=float64)
    max_index = zeros((nc,), dtype=int32)

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
        },
        row_names=options.cell_names,
    )
