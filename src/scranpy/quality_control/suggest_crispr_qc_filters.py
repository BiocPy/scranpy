from dataclasses import dataclass
from typing import Optional, Sequence

from biocframe import BiocFrame
from numpy import float64, ndarray

from .. import cpphelpers as lib
from ..utils import factorize, match_lists


@dataclass
class SuggestCrisprQcFiltersOptions:
    """Optional arguments for :py:meth:`~scranpy.quality_control.suggest_crispr_qc_filters.suggest_crispr_qc_filters`.

    Attributes:
        block (Sequence, optional):
            Block assignment for each cell.
            Thresholds are computed within each block to avoid inflated variances from
            inter-block differences.

            If provided, this should have length equal to the number of cells, where
            cells have the same value if and only if they are in the same block.
            Defaults to None, indicating all cells are part of the same block.

        num_mads (int, optional):
            Number of median absolute deviations for computing an outlier threshold.
            Larger values will result in a less stringent threshold.
            Defaults to 3.

        custom_thresholds (BiocFrame, optional):
            Data frame containing one or more columns with the same names as those in the return value of
            :py:meth:`~scranpy.quality_control.suggest_crispr_qc_filters.suggest_crispr_qc_filters`.
            If a column is present, it should contain custom thresholds for the corresponding metric
            and will override any suggested thresholds in the final BiocFrame.

            If ``block = None``, this data frame should contain one row.
            Otherwise, the number of rows should be equal to the number of blocks,
            where each row contains a block-specific threshold for the relevant metrics.
            The identity of each block should be stored in the row names.

        verbose (bool, optional): Whether to print logging information.
            Defaults to False.
    """

    block: Optional[Sequence] = None
    num_mads: int = 3
    custom_thresholds: Optional[BiocFrame] = None
    verbose: bool = False


def suggest_crispr_qc_filters(
    metrics: BiocFrame,
    options: SuggestCrisprQcFiltersOptions = SuggestCrisprQcFiltersOptions(),
) -> BiocFrame:
    """Suggest filter thresholds for CRISPR-based per-cell quality control (QC) metrics. This identifies outliers on the
    low tail of the distribution of the count for the most abundant guide across cells, aiming to remove cells that have
    low counts due to failed transfection. (Multiple transfections are not considered undesirable at this point.)

    Args:
        metrics (BiocFrame): A data frame containing QC metrics for each cell,
            see the output of :py:meth:`~scranpy.quality_control.per_cell_crispr_qc_metrics.per_cell_crispr_qc_metrics`
            for the expected format.

        options (SuggestCrisprQcFiltersOptions): Optional parameters.

    Raises:
        ValueError, TypeError: if provided ``inputs`` are incorrect type or do
            not contain expected metrics.

    Returns:
        BiocFrame:
            A data frame containing one row per block and the following fields -
            ``"max_count"``, the suggested (lower) threshold on the maximum count.

            If ``options.block`` is None, all cells are assumed to belong to a single
            block, and the output BiocFrame contains a single row.
    """
    use_block = options.block is not None
    block_info = None
    block_offset = 0
    num_blocks = 1
    block_names = None

    if not isinstance(metrics, BiocFrame):
        raise TypeError("'metrics' is not a `BiocFrame` object.")

    if use_block:
        if len(options.block) != metrics.shape[0]:
            raise ValueError(
                "number of rows in 'metrics' should equal the length of 'block'"
            )

        block_info = factorize(options.block)
        block_offset = block_info.indices.ctypes.data
        block_names = block_info.levels
        num_blocks = len(block_names)

    sums = metrics.column("sums")
    if sums.dtype != float64:
        raise TypeError("expected the 'sums' column to be a float64 array.")
    max_prop = metrics.column("max_proportion")
    if max_prop.dtype != float64:
        raise TypeError("expected the 'max_proportion' column to be a float64 array.")
    max_count_out = ndarray((num_blocks,), dtype=float64)

    lib.suggest_crispr_qc_filters(
        metrics.shape[0],
        sums,
        max_prop,
        num_blocks,
        block_offset,
        max_count_out,
        options.num_mads,
    )

    custom_thresholds = options.custom_thresholds
    if custom_thresholds is not None:
        if num_blocks != custom_thresholds.shape[0]:
            raise ValueError(
                "number of rows in 'custom_thresholds' should equal the number of blocks"
            )
        if num_blocks > 1 and custom_thresholds.rownames != block_names:
            m = match_lists(block_names, custom_thresholds.rownames)
            if m is None:
                raise ValueError(
                    "row names of 'custom_thresholds' should equal the unique values of 'block'"
                )
            custom_thresholds = custom_thresholds[m, :]

        if custom_thresholds.has_column("max_count"):
            max_count_out = custom_thresholds.column("max_count")

    return BiocFrame(
        {
            "max_count": max_count_out,
        },
        number_of_rows=num_blocks,
        row_names=block_names,
    )
