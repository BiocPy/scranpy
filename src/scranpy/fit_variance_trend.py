from typing import Tuple

import numpy

from . import lib_scranpy as lib


def fit_variance_trend(
    mean: numpy.ndarray,
    variance: numpy.ndarray,
    mean_filter: bool = True,
    min_mean: float = 0.1, 
    transform: bool = True, 
    span: float = 0.3,
    use_min_width: bool = False,
    min_width: float = 1,
    min_window_count: int = 200,
    num_threads: int = 1
) -> Tuple:
    """
    Fit a trend to the per-cell variances with respect to the mean.

    Args:
        mean:
            Array containing the mean (log-)expression for each gene.

        variance:
            Array containing the variance in the (log-)expression for each
            gene. This should have length equal to ``mean``.

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
        A tuple of two arrays. The first array contains the fitted value of the
        trend for each gene while the second array contains the residual.
    """
    local_m = numpy.array(mean, dtype=numpy.float64, copy=False)
    local_v = numpy.array(variance, dtype=numpy.float64, copy=False)
    return lib.fit_variance_trend(
        local_m,
        local_v,
        mean_filter,
        min_mean,
        transform,
        span,
        use_min_width,
        min_width,
        min_window_count,
        num_threads
    )
