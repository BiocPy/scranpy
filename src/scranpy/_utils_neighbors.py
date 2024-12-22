import warnings
import numpy


def _check_indices(index, num_neighbors):
    if len(index.shape) != 2:
        raise ValueError("'index' should be a two-dimensional array")

    mn = index.min()
    if not numpy.isfinite(mn) or mn < 0:
        raise ValueError("'index' should contain finite positive integers")

    mx = index.max()
    if not numpy.isfinite(mx) or mx >= index.shape[0]:
        raise ValueError("'index' should contain finite integers no greater than the number of columns")

    if index.shape[1] != num_neighbors:
        warnings.warn("number of columns in 'index' is not consistent with 'num_neighbors'")
