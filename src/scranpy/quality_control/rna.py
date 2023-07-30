from typing import Optional, Sequence

import numpy as np
from biocframe import BiocFrame
from mattress import TatamiNumericPointer, tatamize

from .._logging import logger
from ..cpphelpers import lib
from ..types import MatrixTypes, is_matrix_expected_type
from ..utils import factorize, to_logical

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def per_cell_rna_qc_metrics(
    x: MatrixTypes, subsets: dict = {}, num_threads: int = 1, verbose: bool = False
) -> BiocFrame:
    """Compute qc metrics (RNA).

    This function expects the matrix (`x`) to be features (rows) by cells (columns) and
    not the other way around!

    Args:
        x (MatrixTypes): input matrix.
        subsets (dict, optional): named feature subsets.
            Each key is the name of the subset and each value is an array of
            integer indices, specifying the rows of `x` belonging to the subset.
            Defaults to {}.
        num_threads (int, optional): number of threads to use. Defaults to 1.
        verbose (bool, optional): display logs?. Defaults to False.

    Raises:
        TypeError: if x is not an expected matrix type.

    Returns:
        BiocFrame: data frame containing per-cell count sums, number of detected
        features and the proportion of counts in each subset.
    """
    if not is_matrix_expected_type(x):
        raise TypeError(
            f"Input must be a tatami, numpy or sparse matrix, provided {type(x)}."
        )

    if not isinstance(x, TatamiNumericPointer):
        x = tatamize(x)

    nc = x.ncol()
    sums = np.ndarray((nc,), dtype=np.float64)
    detected = np.ndarray((nc,), dtype=np.int32)

    keys = list(subsets.keys())
    num_subsets = len(keys)
    subset_in = np.ndarray((num_subsets,), dtype=np.uintp)
    subset_out = np.ndarray((num_subsets,), dtype=np.uintp)
    collected_in = []
    collected_out = {}

    nr = x.nrow()

    for i in range(num_subsets):
        in_arr = to_logical(subsets[keys[i]], nr)
        collected_in.append(in_arr)
        subset_in[i] = in_arr.ctypes.data

        out_arr = np.ndarray((nc,), dtype=np.float64)
        collected_out[keys[i]] = out_arr
        subset_out[i] = out_arr.ctypes.data

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
        block (Optional[Sequence], optional): Array containing the block/batch
            assignment for each cell. Defaults to None.
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
    subset_in = np.ndarray((num_subsets,), dtype=np.uintp)
    subset_out = {}
    subset_out_ptrs = np.ndarray((num_subsets,), dtype=np.uintp)

    for i in range(num_subsets):
        cursub = subsets.column(i)
        if cursub.dtype != np.float64:
            raise TypeError(
                "expected all 'subset_proportions' columns to be a float64 array."
            )
        subset_in[i] = cursub.ctypes.data

        curout = np.ndarray((num_blocks,), dtype=np.float64)
        subset_out[skeys[i]] = curout
        subset_out_ptrs[i] = curout.ctypes.data

    lib.suggest_rna_qc_filters(
        metrics.shape[0],
        num_subsets,
        sums.ctypes.data,
        detected.ctypes.data,
        subset_in.ctypes.data,
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
