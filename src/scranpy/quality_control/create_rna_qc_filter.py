from dataclasses import dataclass
from typing import Optional, Sequence

from biocframe import BiocFrame
from numpy import array, bool_, float64, int32, ndarray, uint8, zeros

from .. import _cpphelpers as lib
from .._utils import process_block
from ._utils import process_subset_columns

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


@dataclass
class CreateRnaQcFilterOptions:
    """Optional arguments for :py:meth:`~scranpy.quality_control.rna.create_rna_qc_filter`.

    Attributes:
        block:
            Block assignment for each cell.
            This should be the same as that used in
            in :py:meth:`~scranpy.quality_control.rna.suggest_rna_qc_filters`.
    """

    block: Optional[Sequence] = None


def create_rna_qc_filter(
    metrics: BiocFrame,
    thresholds: BiocFrame,
    options: CreateRnaQcFilterOptions = CreateRnaQcFilterOptions(),
) -> ndarray:
    """Defines a filtering vector based on the RNA-derived per-cell quality control (QC) metrics and thresholds.

    Args:
        metrics: 
            Data frame of metrics,
            see :py:meth:`~scranpy.quality_control.rna.per_cell_rna_qc_metrics` for the expected format.

        thresholds: 
            Data frame of filter thresholds,
            see :py:meth:`~scranpy.quality_control.rna.suggest_rna_qc_filters` for the expected format.

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

    subprop = metrics.column("subset_proportions")
    num_subsets = subprop.shape[1]
    subpropthresh = thresholds.column("subset_proportions")
    subset_in, subset_in_ptr = process_subset_columns(subprop)
    filter_in, filter_in_ptr = process_subset_columns(subpropthresh)

    use_block, num_blocks, block_names, block_info, block_offset = process_block(
        options.block, num_cells
    )

    tmp_sums_in = array(metrics.column("sums"), dtype=float64, copy=False)
    tmp_detected_in = array(metrics.column("detected"), dtype=int32, copy=False)
    tmp_sums_thresh = array(thresholds.column("sums"), dtype=float64, copy=False)
    tmp_detected_thresh = array(
        thresholds.column("detected"), dtype=float64, copy=False
    )
    output = zeros(num_cells, dtype=uint8)

    lib.create_rna_qc_filter(
        num_cells,
        num_subsets,
        tmp_sums_in,
        tmp_detected_in,
        subset_in_ptr.ctypes.data,
        num_blocks,
        block_offset,
        tmp_sums_thresh,
        tmp_detected_thresh,
        filter_in_ptr.ctypes.data,
        output,
    )

    return output.astype(bool_, copy=False)
