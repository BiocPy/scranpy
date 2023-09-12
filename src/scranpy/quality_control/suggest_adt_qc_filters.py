from dataclasses import dataclass
from typing import Optional, Sequence

from biocframe import BiocFrame
from numpy import float64, int32, ndarray

from .. import cpphelpers as lib
from ..utils import factorize, match_lists
from .utils import create_pointer_array


@dataclass
class SuggestAdtQcFiltersOptions:
    """Optional arguments for :py:meth:`~scranpy.quality_control.suggest_adt_qc_filters.suggest_adt_qc_filters`.

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
            :py:meth:`~scranpy.quality_control.suggest_adt_qc_filters.suggest_adt_qc_filters`.
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


def suggest_adt_qc_filters(
    metrics: BiocFrame,
    options: SuggestAdtQcFiltersOptions = SuggestAdtQcFiltersOptions(),
) -> BiocFrame:
    """Suggest filter thresholds for ADT-based per-cell quality control (QC) metrics. This identifies outliers on the
    relevant tail of the distribution of each relevant QC metric (namely the number of detected tags and the 
    isotype subset totals; total counts per cell are diagnostic and are not used here). Outlier cells are considered 
    to be low-quality and should be removed before further analysis.

    Args:
        metrics (BiocFrame): A data frame containing QC metrics for each cell,
            see the output of :py:meth:`~scranpy.quality_control.per_cell_adt_qc_metrics.per_cell_adt_qc_metrics` 
            for the expected format.

        options (SuggestAdtQcFiltersOptions): Optional parameters.

    Raises:
        ValueError, TypeError: if provided ``inputs`` are incorrect type or do
            not contain expected metrics.

    Returns:
        BiocFrame:
            A data frame containing one row per block and the following fields -
            ``"detected"``, the suggested (lower) threshold on the number of detected features for each cell;
            and ``"subset_totals"``, a nested BiocFrame where each column is named
            after an entry in ``subsets`` and contains the suggested (upper) threshold
            on the total count in that subset.

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

    detected = metrics.column("detected")
    if detected.dtype != int32:
        raise TypeError("expected the 'detected' column to be an int32 array.")
    detected_out = ndarray((num_blocks,), dtype=float64)

    subsets = metrics.column("subset_totals")
    skeys = subsets.column_names
    num_subsets = len(skeys)
    subset_in = []
    subset_out = {}

    for i in range(num_subsets):
        cursub = subsets.column(i)
        subset_in.append(cursub.astype(float64, copy=False))
        curout = ndarray((num_blocks,), dtype=float64)
        subset_out[skeys[i]] = curout

    subset_in_ptrs = create_pointer_array(subset_in)
    subset_out_ptrs = create_pointer_array(subset_out)

    lib.suggest_adt_qc_filters(
        metrics.shape[0],
        num_subsets,
        detected,
        subset_in_ptrs.ctypes.data,
        num_blocks,
        block_offset,
        detected_out,
        subset_out_ptrs.ctypes.data,
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

        if custom_thresholds.has_column("detected"):
            detected_out = custom_thresholds.column("detected")
        if custom_thresholds.has_column("subset_totals"):
            custom_subs = custom_thresholds.column("subset_totals")
            for s in subset_out.keys():
                if custom_subs.has_column(s):
                    subset_out[s] = custom_subs.column(s)

    return BiocFrame(
        {
            "detected": detected_out,
            "subset_totals": BiocFrame(
                subset_out,
                column_names=skeys,
                number_of_rows=num_blocks,
                row_names=block_names,
            ),
        },
        number_of_rows=num_blocks,
        row_names=block_names,
    )
