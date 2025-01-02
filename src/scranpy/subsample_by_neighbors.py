from typing import Union

import knncolle
import numpy

from ._utils_neighbors import _check_indices
from . import lib_scranpy as lib


def subsample_by_neighbors(
    x: Union[numpy.ndarray, knncolle.FindKnnResults, knncolle.Index],
    num_neighbors: int = 20,
    min_remaining: int = 10,
    nn_parameters: knncolle.Parameters = knncolle.AnnoyParameters(),
    num_threads: int = 1
) -> numpy.ndarray:
    """Subsample a dataset by selecting cells to represent all of their nearest neighbors.

    Args:
        x:
            Numeric matrix where rows are dimensions and columns are cells, typically containing a low-dimensional representation from, e.g., :py:func:`~scranpy.run_pca.run_pca`.

            Alternatively, a :py:class:`~knncolle.Index.Index` object containing a pre-built search index for a dataset.

            Alternatively, a :py:class:`~knncolle.find_knn.FindKnnResults` object containing pre-computed search results for a dataset.
            The number of neighbors should be equal to ``num_neighbors``, otherwise a warning is raised.

        num_neighbors:
            Number of neighbors to use.
            Larger values result in greater downsampling.
            Only used if ``x`` does not contain existing neighbor search results.

        nn_parameters:
            Neighbor search algorithm to use.
            Only used if ``x`` does not contain existing neighbor search results.

        min_remaining:
            Minimum number of remaining (i.e., unselected) neighbors that a cell must have in order to be considered for selection.
            This should be less than or equal to ``num_neighbors``.

        num_threads:
            Number of threads to use for the nearest-neighbor search.
            Only used if ``x`` does not contain existing neighbor search results.

    Returns:
        Integer array with indices of the cells selected to be in the subsample.

    References:
        https://github.com/libscran/nenesub, for the rationale behind this approach.
    """
    if isinstance(x, knncolle.FindKnnResults):
        nnidx = x.index
        nndist = x.distance
        _check_indices(nnidx, num_neighbors)
    else:
        if not isinstance(x, knncolle.Index):
            x = knncolle.build_index(nn_parameters, x)
        x = knncolle.find_knn(x, num_neighbors=num_neighbors, num_threads=num_threads)
        nnidx = x.index
        nndist = x.distance

    return lib.subsample_by_neighbors(
        nnidx,
        nndist,
        min_remaining
    )
