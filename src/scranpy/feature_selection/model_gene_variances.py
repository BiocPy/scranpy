from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from biocframe import BiocFrame

from .. import cpphelpers as lib
from .._logging import logger
from ..types import MatrixTypes
from ..utils import factorize, validate_and_tatamize_input

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


@dataclass
class ModelGeneVariancesArgs:
    """Arguments to model gene variances -
    :py:meth:`~scranpy.feature_selection.model_gene_variances.model_gene_variances`.

    Attributes:
        block (Sequence, optional): Block assignment for each cell.
            This is used to segregate cells in order to perform comparisons within
            each block. Defaults to None, indicating all cells are part of the same
            block.
        span (float, optional): Span to use for the LOWESS trend fitting.
            Defaults to 0.3.
        num_threads (int, optional): Number of threads to use. Defaults to 1.
        verbose (bool, optional): Display logs?. Defaults to False.
    """

    block: Optional[Sequence] = None
    span: float = 0.3
    num_threads: int = 1
    verbose: bool = False


def model_gene_variances(
    input: MatrixTypes, options: ModelGeneVariancesArgs = ModelGeneVariancesArgs()
) -> BiocFrame:
    """Compute model gene variances.

    Ideally, ``input`` would be a normalized log-expression matrix from
    :py:meth:`~scranpy.normalization.log_norm_counts.log_norm_counts`.

    Note: rows are features, columns are cells.

    Args:
        input (MatrixTypes): Log-normalized expression matrix..
        options (ModelGeneVariancesArgs): Optional parameters.

    Returns:
        BiocFrame: Frame with metrics.
    """
    x = validate_and_tatamize_input(input)

    NR = x.nrow()
    means = np.ndarray((NR,), dtype=np.float64)
    variances = np.ndarray((NR,), dtype=np.float64)
    fitted = np.ndarray((NR,), dtype=np.float64)
    residuals = np.ndarray((NR,), dtype=np.float64)
    extra = None

    if options.block is None:
        if options.verbose is True:
            logger.info(
                "No block information was provided, running model_gene_variances..."
            )

        lib.model_gene_variances(
            x.ptr,
            means,
            variances,
            fitted,
            residuals,
            options.span,
            options.num_threads,
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
        if len(options.block) != NC:
            raise ValueError(
                f"Must provide block assignments (provided: {len(options.block)})"
                f" for all cells (expected: {NC})."
            )

        fac = factorize(options.block)
        nlevels = len(fac.levels)

        all_means = []
        all_variances = []
        all_fitted = []
        all_residuals = []
        all_means_ptr = np.ndarray((nlevels,), dtype=np.uintp)
        all_variances_ptr = np.ndarray((nlevels,), dtype=np.uintp)
        all_fitted_ptr = np.ndarray((nlevels,), dtype=np.uintp)
        all_residuals_ptr = np.ndarray((nlevels,), dtype=np.uintp)

        for lvl in range(nlevels):
            cur_means = np.ndarray((NR,), dtype=np.float64)
            cur_variances = np.ndarray((NR,), dtype=np.float64)
            cur_fitted = np.ndarray((NR,), dtype=np.float64)
            cur_residuals = np.ndarray((NR,), dtype=np.float64)

            all_means_ptr[lvl] = cur_means.ctypes.data
            all_variances_ptr[lvl] = cur_variances.ctypes.data
            all_fitted_ptr[lvl] = cur_fitted.ctypes.data
            all_residuals_ptr[lvl] = cur_residuals.ctypes.data

            all_means.append(cur_means)
            all_variances.append(cur_variances)
            all_fitted.append(cur_fitted)
            all_residuals.append(cur_residuals)

        if options.verbose is True:
            logger.info(
                "Block information was provided, running "
                "`model_gene_variances_blocked`..."
            )

        lib.model_gene_variances_blocked(
            x.ptr,
            means,
            variances,
            fitted,
            residuals,
            nlevels,
            fac.indices.ctypes.data,
            all_means_ptr.ctypes.data,
            all_variances_ptr.ctypes.data,
            all_fitted_ptr.ctypes.data,
            all_residuals_ptr.ctypes.data,
            options.span,
            options.num_threads,
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
