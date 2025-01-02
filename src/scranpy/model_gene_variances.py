from typing import Optional, Any, Sequence, Literal, Tuple
from dataclasses import dataclass

import biocutils
import mattress
import numpy

from . import lib_scranpy as lib


@dataclass
class ModelGeneVariancesResults:
    """Results of :py:func:`~model_gene_variances`."""

    mean: numpy.ndarray
    """Mean (log-)expression for each gene."""

    variance: numpy.ndarray
    """Variance in (log-)expression for each gene."""

    fitted: numpy.ndarray
    """Fitted value of the mean-variance trend for each gene."""

    residual: numpy.ndarray
    """Residual from the mean-variance trend for each gene."""

    per_block: Optional[biocutils.NamedList]
    """List of per-block results, obtained from modelling the variances separately for each block of cells.
    Each entry is another ``ModelGeneVariancesResults`` object, containing the statistics for the corresponding block.
    This is only filled if ``block=`` was used, otherwise it is set to ``None``."""

    def to_biocframe(self, include_per_block: bool = False):
        """Convert the results into a :py:class:`~biocframe.BiocFrame.BiocFrame`.

        Args:
            include_per_block:
                Whether to include the per-block results as a nested dataframe.

        Returns: 
            A :py:class:`~biocframe.BiocFrame.BiocFrame` where each row corresponds to a gene and each column corresponds to a statistic.
        """
        colnames = ["mean", "variance", "fitted", "residual"]
        cols = {}
        for n in colnames:
            cols[n] = getattr(self, n)

        import biocframe
        if include_per_block:
            blocknames = self.per_block.get_names()
            per_block = {}
            for i, n in enumerate(blocknames):
                per_block[n] = self.per_block[i].to_biocframe()
            colnames.append("per_block")
            cols["per_block"] = biocframe.BiocFrame(per_block, column_names=blocknames)

        return biocframe.BiocFrame(cols, column_names=colnames)


def model_gene_variances(
    x: Any,
    block: Optional[Sequence] = None,
    block_weight_policy: Literal["variable", "equal", "none"] = "variable",
    variable_block_weight: Tuple = (0, 1000),
    mean_filter: bool = True,
    min_mean: float = 0.1, 
    transform: bool = True, 
    span: float = 0.3,
    use_min_width: bool = False,
    min_width: float = 1,
    min_window_count: int = 200,
    num_threads: int = 1
) -> ModelGeneVariancesResults:
    """Compute the variance in (log-)expression values for each gene, and model
    the trend in the variances with respect to the mean.

    Args:
        x:
            A matrix-like object where rows correspond to genes or genomic
            features and columns correspond to cells.  It is typically expected
            to contain log-expression values, e.g., from
            :py:func:`~scranpy.normalize_counts.normalize_counts`.

        block:
            Array of length equal to the number of columns of ``x``, containing
            the block of origin (e.g., batch, sample) for each cell.
            Alternatively ``None``, if all cells are from the same block.

        block_weight_policy:
            Policy to use for weighting different blocks when computing the
            average for each statistic. Only used if ``block`` is provided.

        variable_block_weight:
            Tuple of length 2, specifying the parameters for variable block
            weighting. The first and second values are used as the lower and
            upper bounds, respectively, for the variable weight calculation.
            Only used if ``block`` is provided and ``block_weight_policy =
            "variable"``.

        mean_filter:
            Whether to filter on the means before trend fitting.

        min_mean:
            The minimum mean of genes to use in trend fitting. Only used if
            ``mean_filter = True``.

        transform:
            Whether a quarter-root transformation should be applied before
            trend fitting.

        span:
            Span of the LOWESS smoother. Ignored if ``use_min_width = TRUE``.

        use_min_width:
            Whether a minimum width constraint should be applied to the LOWESS
            smoother. Useful to avoid overfitting in high-density intervals.

        min_width:
            Minimum width of the window to use when ``use_min_width = TRUE``.

        min_window_count:
            Minimum number of observations in each window. Only used if
            ``use_min_width=TRUE``.

        num_threads:
            Number of threads to use.

    Returns:
        The results of the variance modelling for each gene.
    """
    if block is None:
        blocklev = [] 
        blockind = None
    else:
        blocklev, blockind = biocutils.factorize(block, sort_levels=True, dtype=numpy.uint32, fail_missing=True)

    ptr = mattress.initialize(x)
    mean, var, fit, resid, per_block = lib.model_gene_variances(
        ptr.ptr,
        blockind,
        len(blocklev),
        block_weight_policy,
        variable_block_weight,
        mean_filter,
        min_mean,
        transform,
        span,
        use_min_width,
        min_width,
        min_window_count,
        num_threads
    )

    if not per_block is None:
        pb = []
        for pbm, pbv, pbf, pbr in per_block:
            pb.append(ModelGeneVariancesResults(pbm, pbv, pbf, pbr, None))
        per_block = biocutils.NamedList(pb, blocklev)

    return ModelGeneVariancesResults(mean, var, fit, resid, per_block)
