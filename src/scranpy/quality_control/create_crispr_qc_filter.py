from dataclasses import dataclass
from typing import Optional, Sequence

from biocframe import BiocFrame
from numpy import bool_, float64, ndarray, uint8, zeros

from .. import _cpphelpers as lib
from .._utils import process_block

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


@dataclass
class CreateCrisprQcFilterOptions:
    """Optional arguments for :py:meth:`~scranpy.quality_control.create_crispr_qc_filter.create_crispr_qc_filter`.

    Attributes:
        block:
            Block assignment for each cell.
            This should be the same as that used in
            in :py:meth:`~scranpy.quality_control.rna.suggest_crispr_qc_filters`.
    """

    block: Optional[Sequence] = None


def create_crispr_qc_filter(
    metrics: BiocFrame,
    thresholds: BiocFrame,
    options: CreateCrisprQcFilterOptions = CreateCrisprQcFilterOptions(),
) -> ndarray:
    """Defines a filtering vector based on the RNA-derived per-cell quality control (QC) metrics and thresholds.

    Args:
        metrics: 
            Data frame of metrics,
            see :py:meth:`~scranpy.quality_control.per_cell_crispr_qc_metrics.per_cell_crispr_qc_metrics`
            for the expected format.

        thresholds: 
            Data frame of filter thresholds,
            see :py:meth:`~scranpy.quality_control.suggest_crispr_qc_filters.suggest_crispr_qc_filters`
            for the expected format.

        options: 
            Optional parameters.

    Returns:
        A boolean array where True entries mark the cells to be discarded.
    """

    if not isinstance(metrics, BiocFrame):
        raise TypeError("'metrics' is not a `BiocFrame` object.")

    if not isinstance(thresholds, BiocFrame):
        raise TypeError("'thresholds' is not a `BiocFrame` object.")

    num_cells = metrics.shape[0]
    use_block, num_blocks, block_names, block_info, block_offset = process_block(
        options.block, num_cells
    )

    tmp_sums_in = metrics.column("sums").astype(float64, copy=False)
    tmp_max_proportions_in = metrics.column("detected").astype(float64, copy=False)
    tmp_max_count_thresh = thresholds.column("max_count").astype(float64, copy=False)
    output = zeros(num_cells, dtype=uint8)

    lib.create_crispr_qc_filter(
        num_cells,
        tmp_sums_in,
        tmp_max_proportions_in,
        num_blocks,
        block_offset,
        tmp_max_count_thresh,
        output,
    )

    return output.astype(bool_, copy=False)
