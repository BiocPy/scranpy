from typing import Optional, Sequence

import numpy as np
from biocframe import BiocFrame
from mattress import TatamiNumericPointer, tatamize

from ..cpphelpers import lib
from ..types import MatrixTypes, is_matrix_expected_type
from ..utils import factorize

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def model_gene_variances(
    x: MatrixTypes,
    block: Optional[Sequence] = None,
    span: float = 0.3,
    num_threads: int = 1,
    verbose: bool = False,
) -> BiocFrame:
    """Compute model gene variances.

    Ideally, x would be a normalized log-expression matrix.

    This function expects the matrix (`x`) to be features (rows) by cells (columns) and
    not the other way around!

    Args:
        x (MatrixTypes): Input matrix. Ideally, x would be a normalized
            log-expression matrix.
        block (Optional[Sequence], optional): Array containing the block/batch
            assignment for each cell. Defaults to None.
        span (float, optional): Span to use for the LOWESS trend fitting.
            Defaults to 0.3.
        num_threads (int, optional): number of threads to use. Defaults to 1.
        verbose (bool, optional): display logs?. Defaults to False.

    Returns:
        BiocFrame: data frame with various metrics.
    """
    if not is_matrix_expected_type(x):
        raise TypeError(
            f"Input must be a tatami, numpy or sparse matrix, provided {type(x)}."
        )

    if not isinstance(x, TatamiNumericPointer):
        x = tatamize(x)

    NR = x.nrow()
    means = np.ndarray((NR,), dtype=np.float64)
    variances = np.ndarray((NR,), dtype=np.float64)
    fitted = np.ndarray((NR,), dtype=np.float64)
    residuals = np.ndarray((NR,), dtype=np.float64)
    extra = None

    if block is None:
        lib.model_gene_variances(
            x.ptr,
            means.ctypes.data,
            variances.ctypes.data,
            fitted.ctypes.data,
            residuals.ctypes.data,
            span,
            num_threads,
        )

        return BiocFrame(
            {
                "means": means,
                "variances": variances,
                "fitted": fitted,
                "residuals": residuals,
            },
            numberOfRows=NR,
        )
    else:
        NC = x.ncol()
        if len(block) != NC:
            raise ValueError(
                f"Must provide block assignments (provided: {len(block)})"
                f" for all cells (expected: {NC})."
            )

        fac = factorize(block)
        nlevels = len(fac.levels)

        all_means = []
        all_variances = []
        all_fitted = []
        all_residuals = []
        all_means_ptr = np.ndarray((nlevels,), dtype=np.uintp)
        all_variances_ptr = np.ndarray((nlevels,), dtype=np.uintp)
        all_fitted_ptr = np.ndarray((nlevels,), dtype=np.uintp)
        all_residuals_ptr = np.ndarray((nlevels,), dtype=np.uintp)

        for l in range(nlevels):
            cur_means = np.ndarray((NR,), dtype=np.float64)
            cur_variances = np.ndarray((NR,), dtype=np.float64)
            cur_fitted = np.ndarray((NR,), dtype=np.float64)
            cur_residuals = np.ndarray((NR,), dtype=np.float64)

            all_means_ptr[l] = cur_means.ctypes.data
            all_variances_ptr[l] = cur_variances.ctypes.data
            all_fitted_ptr[l] = cur_fitted.ctypes.data
            all_residuals_ptr[l] = cur_residuals.ctypes.data

            all_means.append(cur_means)
            all_variances.append(cur_variances)
            all_fitted.append(cur_fitted)
            all_residuals.append(cur_residuals)

        lib.model_gene_variances_blocked(
            x.ptr,
            means.ctypes.data,
            variances.ctypes.data,
            fitted.ctypes.data,
            residuals.ctypes.data,
            nlevels,
            fac.indices.ctypes.data,
            all_means_ptr.ctypes.data,
            all_variances_ptr.ctypes.data,
            all_fitted_ptr.ctypes.data,
            all_residuals_ptr.ctypes.data,
            span,
            num_threads,
        )

        extra = {}
        for i in range(nlevels):
            extra[fac.levels[i]] = BiocFrame(
                {
                    "means": all_means[i],
                    "variances": all_variances[i],
                    "fitted": all_fitted[i],
                    "residuals": all_residuals[i],
                }
            )

        return BiocFrame(
            {
                "means": means,
                "variances": variances,
                "fitted": fitted,
                "residuals": residuals,
                "per_block": BiocFrame(extra),
            },
            numberOfRows=NR,
        )
