from dataclasses import dataclass
from typing import Optional, Sequence

from biocframe import BiocFrame
from numpy import array, float64, int32, zeros

from .. import _cpphelpers as lib
from .._utils import process_block
from ._utils import (
    check_custom_thresholds,
    create_subset_buffers,
    create_subset_frame,
    process_subset_columns,
)


@dataclass
class SuggestAdtQcFiltersOptions:
    """Optional arguments for :py:meth:`~scranpy.quality_control.suggest_adt_qc_filters.suggest_adt_qc_filters`.

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
            :py:meth:`~scranpy.quality_control.suggest_adt_qc_filters.suggest_adt_qc_filters`.
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


def suggest_adt_qc_filters(
    metrics: BiocFrame,
    options: SuggestAdtQcFiltersOptions = SuggestAdtQcFiltersOptions(),
) -> BiocFrame:
    """Suggest filter thresholds for ADT-based per-cell quality control (QC) metrics. This identifies outliers on the
    relevant tail of the distribution of each relevant QC metric (namely the number of detected tags and the isotype
    subset totals; total counts per cell are diagnostic and are not used here). Outlier cells are considered to be low-
    quality and should be removed before further analysis.

    Args:
        metrics:
            A data frame containing QC metrics for each cell,
            see the output of :py:meth:`~scranpy.quality_control.per_cell_adt_qc_metrics.per_cell_adt_qc_metrics`
            for the expected format.

        options:
            Optional parameters.

    Raises:
        ValueError, TypeError:
            If provided ``inputs`` are incorrect type or do
            not contain expected metrics.

    Returns:
        A data frame containing one row per block and the following fields -
        ``"detected"``, the suggested (lower) threshold on the number of detected features for each cell;
        and ``"subset_totals"``, a nested BiocFrame where each column is named
        after an entry in ``subsets`` and contains the suggested (upper) threshold
        on the total count in that subset.

        If ``options.block`` is None, all cells are assumed to belong to a single
        block, and the output BiocFrame contains a single row.
    """
    if not isinstance(metrics, BiocFrame):
        raise TypeError("'metrics' is not a `BiocFrame` object.")

    num_cells = metrics.shape[0]
    use_block, num_blocks, block_names, block_info, block_offset = process_block(
        options.block, num_cells
    )

    detected = array(metrics.column("detected"), dtype=int32, copy=False)
    detected_out = zeros((num_blocks,), dtype=float64)

    subsets = metrics.column("subset_totals")
    num_subsets = subsets.shape[1]
    subset_in, subset_in_ptrs = process_subset_columns(subsets)
    raw_subset_out, subset_out_ptrs = create_subset_buffers(num_blocks, num_subsets)

    lib.suggest_adt_qc_filters(
        num_cells,
        num_subsets,
        detected,
        subset_in_ptrs.ctypes.data,
        num_blocks,
        block_offset,
        detected_out,
        subset_out_ptrs.ctypes.data,
        options.num_mads,
    )

    subset_out = create_subset_frame(
        column_names=subsets.column_names,
        columns=raw_subset_out,
        num_rows=num_blocks,
        row_names=block_names,
    )

    custom_thresholds = check_custom_thresholds(
        num_blocks, block_names, options.custom_thresholds
    )
    if custom_thresholds is not None:
        if custom_thresholds.has_column("detected"):
            detected_out = custom_thresholds.column("detected")
        if custom_thresholds.has_column("subset_totals"):
            custom_subs = custom_thresholds.column("subset_totals")
            for s in subset_out.column_names:
                if custom_subs.has_column(s):
                    subset_out[s] = custom_subs.column(s)

    return BiocFrame(
        {
            "detected": detected_out,
            "subset_totals": subset_out,
        },
        number_of_rows=num_blocks,
        row_names=block_names,
    )
