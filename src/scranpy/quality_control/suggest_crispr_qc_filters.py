from dataclasses import dataclass
from typing import Optional, Sequence

from biocframe import BiocFrame
from numpy import float64, array, zeros

from .. import _cpphelpers as lib
from .._utils import process_block
from ._utils import check_custom_thresholds


@dataclass
class SuggestCrisprQcFiltersOptions:
    """Optional arguments for :py:meth:`~scranpy.quality_control.suggest_crispr_qc_filters.suggest_crispr_qc_filters`.

    Attributes:
        block:
            Block assignment for each cell.
            Thresholds are computed within each block to avoid inflated variances from
            inter-block differences.

            If provided, this should have length equal to the number of cells, where
            cells have the same value if and only if they are in the same block.
            Defaults to None, indicating all cells are part of the same block.

        num_mads:
            Number of median absolute deviations for computing an outlier threshold.
            Larger values will result in a less stringent threshold.
            Defaults to 3.

        custom_thresholds:
            Data frame containing one or more columns with the same names as those in the return value of
            :py:meth:`~scranpy.quality_control.suggest_crispr_qc_filters.suggest_crispr_qc_filters`.
            If a column is present, it should contain custom thresholds for the corresponding metric
            and will override any suggested thresholds in the final BiocFrame.

            If ``block = None``, this data frame should contain one row.
            Otherwise, the number of rows should be equal to the number of blocks,
            where each row contains a block-specific threshold for the relevant metrics.
            The identity of each block should be stored in the row names.
    """

    block: Optional[Sequence] = None
    num_mads: int = 3
    custom_thresholds: Optional[BiocFrame] = None


def suggest_crispr_qc_filters(
    metrics: BiocFrame,
    options: SuggestCrisprQcFiltersOptions = SuggestCrisprQcFiltersOptions(),
) -> BiocFrame:
    """Suggest filter thresholds for CRISPR-based per-cell quality control (QC) metrics. This identifies outliers on the
    low tail of the distribution of the count for the most abundant guide across cells, aiming to remove cells that have
    low counts due to failed transfection. (Multiple transfections are not considered undesirable at this point.)

    Args:
        metrics:
            A data frame containing QC metrics for each cell,
            see the output of :py:meth:`~scranpy.quality_control.per_cell_crispr_qc_metrics.per_cell_crispr_qc_metrics`
            for the expected format.

        options:
            Optional parameters.

    Raises:
        ValueError, TypeError:
            If provided ``inputs`` are incorrect type or do
            not contain expected metrics.

    Returns:
        A data frame containing one row per block and the following fields -
        ``"max_count"``, the suggested (lower) threshold on the maximum count.

        If ``options.block`` is None, all cells are assumed to belong to a single
        block, and the output BiocFrame contains a single row.
    """
    if not isinstance(metrics, BiocFrame):
        raise TypeError("'metrics' is not a `BiocFrame` object.")

    num_cells = metrics.shape[0]
    use_block, num_blocks, block_names, block_info, block_offset = process_block(
        options.block, num_cells
    )

    sums = array(metrics.column("sums"), dtype=float64, copy=False)
    max_prop = array(metrics.column("max_proportion"), dtype=float64, copy=False)
    max_count_out = zeros((num_blocks,), dtype=float64)

    lib.suggest_crispr_qc_filters(
        num_cells,
        sums,
        max_prop,
        num_blocks,
        block_offset,
        max_count_out,
        options.num_mads,
    )

    custom_thresholds = check_custom_thresholds(
        num_blocks, block_names, options.custom_thresholds
    )
    if custom_thresholds is not None:
        if custom_thresholds.has_column("max_count"):
            max_count_out = custom_thresholds.column("max_count")

    return BiocFrame(
        {
            "max_count": max_count_out,
        },
        number_of_rows=num_blocks,
        row_names=block_names,
    )
