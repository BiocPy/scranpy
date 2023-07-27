from typing import Sequence

import numpy as np
from mattress import TatamiNumericPointer, tatamize

from .._logging import logger
from ..cpphelpers import lib
from ..types import MatrixTypes, RnaQcResult, is_matrix_expected_type

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def per_cell_rna_qc_metrics(
    x: MatrixTypes, subsets: Sequence = [], num_threads: int = 1, verbose: bool = False
) -> RnaQcResult:
    """Compute qc metrics (RNA).

    This function expects the matrix (`x`) to be features (rows) by cells (columns) and
    not the other way around!

    Args:
        x (MatrixTypes): input matrix.
        subsets (Sequence, optional): parameter to specify batches or subsets.
            Defaults to [].
        num_threads (int, optional): number of threads to use. Defaults to 1.
        verbose (bool, optional): display logs?. Defaults to False.

    Raises:
        TypeError: if x is not an expected matrix type.

    Returns:
        RnaQcResult: a named tuple with sums, detected and subset proportions.
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

    num_subsets = len(subsets)
    subset_in = np.ndarray((num_subsets,), dtype=np.uintp)
    subset_out = np.ndarray((num_subsets,), dtype=np.uintp)
    collected_in = []
    collected_out = []

    nr = x.nrow()

    for i in range(num_subsets):
        in_arr = np.ndarray((nr,), dtype=np.uint8)
        in_arr.fill(0)
        for j in subsets[i]:
            in_arr[j] = 1
        collected_in.append(in_arr)
        subset_in[i] = in_arr.ctypes.data

        out_arr = np.ndarray((nc,), dtype=np.float64)
        collected_out.append(out_arr)
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

    return RnaQcResult(sums, detected, collected_out)
