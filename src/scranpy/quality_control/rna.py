from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np
from biocframe import BiocFrame

from .. import cpphelpers as lib
from .._logging import logger
from ..types import MatrixTypes
from ..utils import factorize, to_logical, validate_and_tatamize_input

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def create_pointer_array(arrs):
    num = len(arrs)
    output = np.ndarray((num,), dtype=np.uintp)

    if isinstance(arrs, list):
        for i in range(num):
            output[i] = arrs[i].ctypes.data
    else:
        i = 0
        for k in arrs:
            output[i] = arrs[k].ctypes.data
            i += 1

    return output


@dataclass
class PerCellRnaQcMetricsOptions:
    """Optional arguments for 
    :py:meth:`~scranpy.quality_control.rna.per_cell_rna_qc_metrics`.

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


def per_cell_rna_qc_metrics(
    input: MatrixTypes,
    options: PerCellRnaQcMetricsOptions = PerCellRnaQcMetricsOptions(),
) -> BiocFrame:
    """Compute per-cell quality control metrics for RNA data.
    This includes the total count for each cell, where low values are indicative of problems with library preparation or sequencing;
    the number of detected features per cell, where low values are indicative of problems with transcript capture;
    and the proportion of counts in particular feature subsets, 
    typically mitochondrial genes where high values are indicative of cell damage.

    Args:
        input (MatrixTypes): 
            Matrix-like object containing cells in columns and features in rows, typically with count data.
            This should be a matrix class that can be converted into a :py:class:`~mattress.TatamiNumericPointer`.
            Developers may also provide the :py:class:`~mattress.TatamiNumericPointer` itself.
        options (PerCellRnaQcMetricsOptions): Optional parameters.

    Raises:
        TypeError: If ``input`` is not an expected matrix type.

    Returns:
        BiocFrame: 
            A data frame containing one row per cell and the following fields - 
            ``"sums"``, the total count for each cell;
            ``"detected"``, the number of detected features for each cell;
            and ``"subset_proportions"``, a nested BiocFrame where each column is named after an entry in ``subsets`` and contains the proportion of counts in that subset.
    """
    x = validate_and_tatamize_input(input)

    if options.subsets is None:
        options.subsets = {}

    nr = x.nrow()
    nc = x.ncol()
    sums = np.ndarray((nc,), dtype=np.float64)
    detected = np.ndarray((nc,), dtype=np.int32)

    keys = list(options.subsets.keys())
    num_subsets = len(keys)
    collected_in = []
    collected_out = {}

    for i in range(num_subsets):
        in_arr = to_logical(options.subsets[keys[i]], nr)
        collected_in.append(in_arr)
        out_arr = np.ndarray((nc,), dtype=np.float64)
        collected_out[keys[i]] = out_arr

    subset_in = create_pointer_array(collected_in)
    subset_out = create_pointer_array(collected_out)
    if options.verbose is True:
        logger.info(subset_in)
        logger.info(subset_out)

    lib.per_cell_rna_qc_metrics(
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
            "subset_proportions": BiocFrame(collected_out, numberOfRows=nc),
        }
    )


@dataclass
class SuggestRnaQcFiltersOptions:
    """Optional arguments for
    :py:meth:`~scranpy.quality_control.rna.suggest_rna_qc_filters`.

    Attributes:
        block (Sequence, optional): 
            Block assignment for each cell.
            Thresholds are computed within each block to avoid inflated variances from inter-block differences.

            If provided, this should have length equal to the number of cells, where cells have the same value if and only if they are in the same block.
            Defaults to None, indicating all cells are part of the same block.
        num_mads (int, optional): Number of median absolute deviations to
            use to compute a threshold for low-quality outliers. Defaults to 3.
        verbose (bool, optional): Display logs?. Defaults to False.
    """

    block: Optional[Sequence] = None
    num_mads: int = 3
    verbose: bool = False


def suggest_rna_qc_filters(
    metrics: BiocFrame, options: SuggestRnaQcFiltersOptions = SuggestRnaQcFiltersOptions()
) -> BiocFrame:
    """Suggest filter thresholds for RNA-based per-cell quality control (QC) metrics.
    This identifies outliers on the relevant tail of the distribution of each QC metric.
    Outlier cells are considered to be low-quality and should be removed before further analysis.

    Args:
        metrics (BiocFrame): A data frame containing QC metrics for each cell,
            see the output of :py:meth:`~scranpy.quality_control.rna.per_cell_rna_qc_metrics` for the expected format.
        options (SuggestRnaQcFilters): Optional parameters.

    Raises:
        ValueError, TypeError: if provided ``inputs`` are incorrect type or do
            not contain expected metrics.

    Returns:
        BiocFrame: 
            A data frame containing one row per block and the following fields - 
            ``"sums"``, the suggested (lower) threshold on the total count for each cell;
            ``"detected"``, the suggested (lower) threshold on the number of detected features for each cell;
            and ``"subset_proportions"``, a nested BiocFrame where each column is named after an entry in ``subsets`` and contains the suggested (upper) threshold on the proportion of counts in that subset.

            If ``options.block`` is None, all cells are assumed to belong to a single block, and the output BiocFrame contains a single row.
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
    if sums.dtype != np.float64:
        raise TypeError("expected the 'sums' column to be a float64 array.")
    sums_out = np.ndarray((num_blocks,), dtype=np.float64)

    detected = metrics.column("detected")
    if detected.dtype != np.int32:
        raise TypeError("expected the 'detected' column to be an int32 array.")
    detected_out = np.ndarray((num_blocks,), dtype=np.float64)

    subsets = metrics.column("subset_proportions")
    skeys = subsets.columnNames
    num_subsets = len(skeys)
    subset_in = []
    subset_out = {}

    for i in range(num_subsets):
        cursub = subsets.column(i)
        subset_in.append(cursub.astype(np.float64, copy=False))
        curout = np.ndarray((num_blocks,), dtype=np.float64)
        subset_out[skeys[i]] = curout

    subset_in_ptrs = create_pointer_array(subset_in)
    subset_out_ptrs = create_pointer_array(subset_out)

    lib.suggest_rna_qc_filters(
        metrics.shape[0],
        num_subsets,
        sums,
        detected,
        subset_in_ptrs.ctypes.data,
        num_blocks,
        block_offset,
        sums_out,
        detected_out,
        subset_out_ptrs.ctypes.data,
        options.num_mads,
    )

    return BiocFrame(
        {
            "sums": sums_out,
            "detected": detected_out,
            "subset_proportions": BiocFrame(
                subset_out,
                columnNames=skeys,
                numberOfRows=num_blocks,
                rowNames=block_names,
            ),
        },
        numberOfRows=num_blocks,
        rowNames=block_names,
    )


@dataclass
class CreateRnaQcFilterOptions:
    """Optional arguments for 
    :py:meth:`~scranpy.quality_control.rna.create_rna_qc_filter`.

    Attributes:
        block (Sequence, optional): 
            Block assignment for each cell.
            This should be the same as that used in 
            in :py:meth:`~scranpy.quality_control.rna.suggest_rna_qc_filters`.
        verbose (bool, optional): Whether to print logs. Defaults to False.
    """

    block: Optional[Sequence] = None
    verbose: bool = False


def create_rna_qc_filter(
    metrics: BiocFrame,
    thresholds: BiocFrame,
    options: CreateRnaQcFilterOptions = CreateRnaQcFilterOptions(),
) -> np.ndarray:
    """Defines a filtering vector based on the RNA-derived per-cell quality control (QC) metrics and thresholds.

    Args:
        metrics (BiocFrame): Data frame of metrics,
            see :py:meth:`~scranpy.quality_control.rna.per_cell_rna_qc_metrics` for the expected format.
        thresholds (BiocFrame): Data frame of filter thresholds,
            see :py:meth:`~scranpy.quality_control.rna.suggest_rna_qc_filters` for the expected format.
        options (CreateRnaQcFilter): Optional parameters.

    Returns:
        np.ndarray: a numpy boolean array filled with 1 for cells to filter.
    """

    if not isinstance(metrics, BiocFrame):
        raise TypeError("'metrics' is not a `BiocFrame` object.")

    if not isinstance(thresholds, BiocFrame):
        raise TypeError("'thresholds' is not a `BiocFrame` object.")

    subprop = metrics.column("subset_proportions")
    num_subsets = subprop.shape[1]
    subset_in = []
    filter_in = []

    for i in range(num_subsets):
        cursub = subprop.column(i)
        subset_in.append(cursub.astype(np.float64, copy=False))
        curfilt = thresholds.column(i)
        filter_in.append(curfilt.astype(np.float64, copy=False))

    subset_in_ptr = create_pointer_array(subset_in)
    filter_in_ptr = create_pointer_array(filter_in)

    num_blocks = 1
    block_offset = 0
    block_info = None

    if options.block is not None:
        block_info = factorize(options.block)
        block_offset = block_info.indices.ctypes.data
        num_blocks = len(block_info.levels)

    tmp_sums_in = metrics.column("sums").astype(np.float64, copy=False)
    tmp_detected_in = metrics.column("detected").astype(np.int32, copy=False)
    tmp_sums_thresh = thresholds.column("sums").astype(np.float64, copy=False)
    tmp_detected_thresh = thresholds.column("detected").astype(np.float64, copy=False)
    output = np.zeros(metrics.shape[0], dtype=np.uint8)

    lib.create_rna_qc_filter(
        metrics.shape[0],
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

    return output.astype(np.bool_, copy=False)


def guess_mito_from_symbols(
    symbols: Sequence[str], prefix: str = "mt-"
) -> Sequence[int]:
    """Guess mitochondrial genes from their gene symbols.

    Args:
        symbols (Sequence[str]): List of gene symbols.
        prefix (str): Case-insensitive prefix to guess mitochondrial genes.

    Return:
        Sequence[int]: List of integer indices for the guessed mitochondrial genes.
    """

    prefix = prefix.lower()
    output = []
    for i, symb in enumerate(symbols):
        if symb.lower().startswith(prefix):
            output.append(i)

    return output
