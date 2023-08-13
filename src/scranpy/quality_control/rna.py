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
    """Arguments to compute per cell QC metrics (RNA) -
    :py:meth:`~scranpy.quality_control.rna.per_cell_rna_qc_metrics`.

    Attributes:
        subsets (Mapping, optional): Dictionary of feature subsets.
            Each key is the name of the subset and each value is an array of
            integer indices or booleans, specifying the rows of `input` belonging to the
            subset. Defaults to {}.
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
    """Compute QC metrics (RNA).

    Note: rows are features, columns are cells.

    Args:
        input (MatrixTypes): Input matrix.
        options (PerCellRnaQcMetricsOptions): Optional parameters.

    Raises:
        TypeError: If ``input`` is not an expected matrix type.

    Returns:
        BiocFrame: DataFrame containing per-cell 'counts', 'sums', 'number of detected
        features' and the 'proportion' of counts in each subset.
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
class SuggestRnaQcFilters:
    """Arguments to suggest QC Filters (RNA) -
    :py:meth:`~scranpy.quality_control.rna.suggest_rna_qc_filters`.

    Attributes:
        block (Sequence, optional): Block assignment for each cell.
            This is used to segregate cells in order to perform comparisons within
            each block. Defaults to None, indicating all cells are part of the same
            block.
        num_mads (int, optional): Number of median absolute deviations to
            filter low-quality cells. Defaults to 3.
        verbose (bool, optional): Display logs?. Defaults to False.
    """

    block: Optional[Sequence] = None
    num_mads: int = 3
    verbose: bool = False


def suggest_rna_qc_filters(
    metrics: BiocFrame, options: SuggestRnaQcFilters = SuggestRnaQcFilters()
) -> BiocFrame:
    """Suggest filters for qc (RNA).

    Args:
        metrics (BiocFrame): A BiocFrame that contains 'sums', 'detected'
            and 'proportions' for each cell. Usually the result of
            :py:meth:`~scranpy.quality_control.rna.per_cell_rna_qc_metrics` method.
        options (SuggestRnaQcFilters): Optional parameters.

    Raises:
        ValueError, TypeError: if provided ``inputs`` are incorrect type or do
            not contain expected metrics.

    Returns:
        BiocFrame: suggested filters for each metric.
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
class CreateRnaQcFilter:
    """Arguments to create an RNA QC Filter -
    :py:meth:`~scranpy.quality_control.rna.create_rna_qc_filter`.

    Attributes:
        block (Sequence, optional): Block assignment for each cell.
            This is used to segregate cells in order to perform comparisons within
            each block. Defaults to None, indicating all cells are part of the same
            block.
        verbose (bool, optional): Display logs?. Defaults to False.
    """

    block: Optional[Sequence] = None
    verbose: bool = False


def create_rna_qc_filter(
    metrics: BiocFrame,
    thresholds: BiocFrame,
    options: CreateRnaQcFilter = CreateRnaQcFilter(),
) -> np.ndarray:
    """Defines a QC filter (RNA) based on the per-cell
    QC metrics computed from an RNA count matrix and thresholds suggested
    by the :py:meth:`~scranpy.quality_control.rna.suggest_rna_qc_filters`.

    Args:
        metrics (BiocFrame): DataFrame of results from
            :py:meth:`~scranpy.quality_control.rna.per_cell_rna_qc_metrics`
            function.
        thresholds (BiocFrame): Suggested (or modified) filters from
            :py:meth:`~scranpy.quality_control.rna.suggest_rna_qc_filters`
            function.
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
    """Guess mitochondrial genes based on the gene symbols.

    Args:
        symbols (Sequence[str]): List of symbols, one per gene.
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
