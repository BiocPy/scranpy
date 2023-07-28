import numpy as np
from biocframe import BiocFrame
from mattress import TatamiNumericPointer, tatamize

from .._logging import logger
from ..cpphelpers import lib
from ..types import MatrixTypes, is_matrix_expected_type

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
        BiocFrame: data frame containing per-cell count sums, number of detected features
        and the proportion of counts in each subset.
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
        in_arr = np.zeros((nr,), dtype=np.uint8)
        in_arr[subsets[keys[i]]] = 1
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
