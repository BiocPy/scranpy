from dataclasses import dataclass
from typing import Optional, Sequence, Union

from biocframe import BiocFrame
from numpy import float64, ndarray, uintp

from .. import cpphelpers as lib
from .._utils import factorize, tatamize_input, MatrixTypes

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


@dataclass
class ModelGeneVariancesOptions:
    """Optional arguments for :py:meth:`~scranpy.feature_selection.model_gene_variances.model_gene_variances`.

    Attributes:
        block (Sequence, optional): Block assignment for each cell.
            Variance modelling is performed within each block to avoid interference from inter-block differences.

            If provided, this should have length equal to the number of cells, where cells have the same value if and
            only if they are in the same block. Defaults to None, indicating all cells are part of the same block.

        span (float, optional): Span to use for the LOWESS trend fitting.
            Larger values yield a smoother curve and reduces the risk of overfitting,
            at the cost of being less responsive to local variations.
            Defaults to 0.3.

        assay_type (Union[int, str]):
            Assay to use from ``input`` if it is a
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.

        feature_names (Sequence[str], optional):
            Sequence of feature names of length equal to the number of rows in ``input``.
            If provided, this is used as the row names of the output data frames.

        num_threads (int, optional): Number of threads to use. Defaults to 1.
    """

    block: Optional[Sequence] = None
    span: float = 0.3
    assay_type: Union[int, str] = "logcounts"
    feature_names: Optional[Sequence[str]] = None
    num_threads: int = 1


def model_gene_variances(
    input: MatrixTypes, options: ModelGeneVariancesOptions = ModelGeneVariancesOptions()
) -> BiocFrame:
    """Compute model gene variances.

    Ideally, ``input`` would be a normalized log-expression matrix from
    :py:meth:`~scranpy.normalization.log_norm_counts.log_norm_counts`.

    Note: rows are features, columns are cells.

    Args:
        input (MatrixTypes): Matrix-like object where rows are features and columns are cells, typically containing
            expression values of some kind. This should be a matrix class that can be converted into a
            :py:class:`~mattress.TatamiNumericPointer`.

            Alternatively, a :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`
            containing such a matrix in its assays.

            Developers may also provide a :py:class:`~mattress.TatamiNumericPointer` directly.

        options (ModelGeneVariancesOptions): Optional parameters.

    Returns:
        BiocFrame: Data frame with variance modelling results
        (means, variance, fitted, residuals).
    """
    x = tatamize_input(input, options.assay_type)

    NR = x.nrow()
    means = ndarray((NR,), dtype=float64)
    variances = ndarray((NR,), dtype=float64)
    fitted = ndarray((NR,), dtype=float64)
    residuals = ndarray((NR,), dtype=float64)
    extra = None

    if options.block is None:
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
            number_of_rows=NR,
            row_names=options.feature_names,
        )
    else:
        NC = x.ncol()
        if len(options.block) != NC:
            raise ValueError(
                f"Must provide block assignments (provided: {len(options.block)})"
                f" for all cells (expected: {NC})."
            )

        block_levels, block_indices = factorize(options.block)
        nlevels = len(block_levels)

        all_means = []
        all_variances = []
        all_fitted = []
        all_residuals = []
        all_means_ptr = ndarray((nlevels,), dtype=uintp)
        all_variances_ptr = ndarray((nlevels,), dtype=uintp)
        all_fitted_ptr = ndarray((nlevels,), dtype=uintp)
        all_residuals_ptr = ndarray((nlevels,), dtype=uintp)

        for lvl in range(nlevels):
            cur_means = ndarray((NR,), dtype=float64)
            cur_variances = ndarray((NR,), dtype=float64)
            cur_fitted = ndarray((NR,), dtype=float64)
            cur_residuals = ndarray((NR,), dtype=float64)

            all_means_ptr[lvl] = cur_means.ctypes.data
            all_variances_ptr[lvl] = cur_variances.ctypes.data
            all_fitted_ptr[lvl] = cur_fitted.ctypes.data
            all_residuals_ptr[lvl] = cur_residuals.ctypes.data

            all_means.append(cur_means)
            all_variances.append(cur_variances)
            all_fitted.append(cur_fitted)
            all_residuals.append(cur_residuals)

        lib.model_gene_variances_blocked(
            x.ptr,
            means,
            variances,
            fitted,
            residuals,
            nlevels,
            block_indices.ctypes.data,
            all_means_ptr.ctypes.data,
            all_variances_ptr.ctypes.data,
            all_fitted_ptr.ctypes.data,
            all_residuals_ptr.ctypes.data,
            options.span,
            options.num_threads,
        )

        extra = {}
        for i in range(nlevels):
            extra[block_levels[i]] = BiocFrame(
                {
                    "means": all_means[i],
                    "variances": all_variances[i],
                    "fitted": all_fitted[i],
                    "residuals": all_residuals[i],
                },
                row_names=options.feature_names,
            )

        return BiocFrame(
            {
                "means": means,
                "variances": variances,
                "fitted": fitted,
                "residuals": residuals,
                "per_block": BiocFrame(extra),
            },
            number_of_rows=NR,
            row_names=options.feature_names,
        )
