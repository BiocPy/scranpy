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


def per_cell_rna_qc_metrics(
    x: MatrixTypes,
    subsets: Optional[Mapping] = None,
    num_threads: int = 1,
    verbose: bool = False,
) -> BiocFrame:
    """Compute qc metrics (RNA).

    This function expects the matrix (`x`) to be features (rows) by cells (columns) and
    not the other way around!

    Args:
        x (MatrixTypes): Input matrix.
        subsets (Mapping, optional): Dictionary of feature subsets.
            Each key is the name of the subset and each value is an array of
            integer indices, specifying the rows of `x` belonging to the subset.
            Defaults to {}.
        num_threads (int, optional): Number of threads to use. Defaults to 1.
        verbose (bool, optional): Display logs?. Defaults to False.

    Raises:
        TypeError: If x is not an expected matrix type.

    Returns:
        BiocFrame: Data frame containing per-cell count sums, number of detected
        features and the proportion of counts in each subset.
    """
    x = validate_and_tatamize_input(x)

    if subsets is None:
        subsets = {}

    nr = x.nrow()
    nc = x.ncol()
    sums = np.ndarray((nc,), dtype=np.float64)
    detected = np.ndarray((nc,), dtype=np.int32)

    keys = list(subsets.keys())
    num_subsets = len(keys)
    collected_in = []
    collected_out = {}

    for i in range(num_subsets):
        in_arr = to_logical(subsets[keys[i]], nr)
        collected_in.append(in_arr)
        out_arr = np.ndarray((nc,), dtype=np.float64)
        collected_out[keys[i]] = out_arr

    subset_in = create_pointer_array(collected_in)
    subset_out = create_pointer_array(collected_out)
    if verbose is True:
        logger.info(subset_in)
        logger.info(subset_out)

    lib.per_cell_rna_qc_metrics(
        x.ptr,
        num_subsets,
        subset_in.ctypes.data,
        sums.ctypes.data,
        detected.ctypes.data,
        subset_out.ctypes.data,
        num_threads,
    )

    return BiocFrame(
        {
            "sums": sums,
            "detected": detected,
            "subset_proportions": BiocFrame(collected_out, numberOfRows=nc),
        }
    )


def suggest_rna_qc_filters(
    metrics: BiocFrame, block: Optional[Sequence] = None, num_mads: int = 3
) -> BiocFrame:
    """Suggest filters for qc (RNA).

    metrics is usually the result of calling `per_cell_rna_qc_metrics` method.

    Args:
        metrics (BiocFrame): A BiocFrame that contains sums, detected and proportions
            for each cell. Usually the result of `per_cell_rna_qc_metrics` method.
        block (Sequence, optional): block assignment for each cell.
            This is used to segregate cells in order to perform comparisons within
            each block. Defaults to None, indicating all cells are part of the same
            block.
        num_mads (int, optional): Number of median absolute deviations to
            filter low-quality cells. Defaults to 3.

    Raises:
        ValueError, TypeError: if provided objects are incorrect type or do not contain
            expected metrics.

    Returns:
        BiocFrame: suggested filters for each metric.
    """
    use_block = block is not None
    block_info = None
    block_offset = 0
    num_blocks = 1
    block_names = None

    if use_block:
        if len(block) != metrics.shape[0]:
            raise ValueError(
                "number of rows in 'metrics' should equal the length of 'block'"
            )

        block_info = factorize(block)
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
        sums.ctypes.data,
        detected.ctypes.data,
        subset_in_ptrs.ctypes.data,
        num_blocks,
        block_offset,
        sums_out.ctypes.data,
        detected_out.ctypes.data,
        subset_out_ptrs.ctypes.data,
        num_mads,
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


def create_rna_qc_filter(
    metrics: BiocFrame, thresholds: BiocFrame, block: Optional[Sequence] = None
) -> np.ndarray:
    """Define an qc filter (RNA) based on the per-cell
    QC metrics computed from an RNA count matrix and thresholds suggested
    by tge suggest_rna_qc_filters.

    Args:
        metrics (BiocFrame): A BiocFrame object returned by
            `per_cell_rna_qc_metrics` function\.
        thresholds (BiocFrame): Suggested (or modified) filters from
            `suggest_rna_qc_filters` function.
        block (Sequence, optional): Block assignment for each cell.
            This is used to segregate cells in order to perform comparisons within
            each block. Defaults to None, indicating all cells are part of the same
            block.

    Returns:
        np.ndarray: a numpy boolean array filled with 1 for cells to filter.
    """
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
    if block is not None:
        block_info = factorize(block)
        block_offset = block_info.indices.ctypes.data
        num_blocks = len(block_info.levels)

    output = np.zeros(metrics.shape[0], dtype=np.uint8)
    lib.create_rna_qc_filter(
        metrics.shape[0],
        num_subsets,
        metrics.column("sums").astype(np.float64, copy=False).ctypes.data,
        metrics.column("detected").astype(np.int32, copy=False).ctypes.data,
        subset_in_ptr.ctypes.data,
        num_blocks,
        block_offset,
        thresholds.column("sums").astype(np.float64, copy=False).ctypes.data,
        thresholds.column("detected").astype(np.float64, copy=False).ctypes.data,
        filter_in_ptr.ctypes.data,
        output.ctypes.data,
    )

    return output.astype(np.bool_)


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
