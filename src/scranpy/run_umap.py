from typing import Optional, Union

import knncolle
import numpy

from . import lib_scranpy as lib
from ._utils_neighbors import _check_indices


def run_umap(
    x: Union[numpy.ndarray, knncolle.FindKnnResults, knncolle.Index],
    num_dim: int = 2,
    num_neighbors: int = 15, 
    num_epochs: Optional[int] = None,
    min_dist: float = 0.1, 
    seed: int = 1234567890, 
    num_threads: int = 1,
    parallel_optimization: bool = False,
    nn_parameters: knncolle.Parameters = knncolle.AnnoyParameters()
) -> numpy.ndarray:
    """Compute UMAP coordinates to visualize similarities between cells.

    Args:
        x: 
            Numeric matrix where rows are dimensions and columns are cells,
            typically containing a low-dimensional representation from, e.g.,
            :py:func:`~run_pca.run_pca`.

            Alternatively, a :py:class:`~knncolle.find_knn.FindKnnResults`
            object containing existing neighbor search results. The number of
            neighbors should be the same as ``num_neighbors``, otherwise a
            warning is raised.

            Alternatively, a :py:class:`~knncolle.Index.Index` object.

        num_dim:
            Number of dimensions in the UMAP embedding.

        num_neighbors:
            Number of neighbors to use in the UMAP algorithm. Larger values
            cause the embedding to focus on global structure.

        num_epochs:
            Number of epochs to perform. If set to None, an appropriate number
            of epochs is chosen based on the number of points in ``x``.

        min_dist:
            Minimum distance between points in the embedding. Larger values
            result in more visual clusters that are more dispersed.

        seed:
            Integer scalar specifying the seed to use. 

        num_threads:
            Number of threads to use.

        parallel_optimization:
            Whether to parallelize the optimization step.

        nn_parameters:
            The algorithm to use for the nearest-neighbor search. Only used if
            ``x`` is not a pre-built nearest-neighbor search index or a list of
            existing nearest-neighbor search results.

    Returns:
        Array containing the coordinates of each cell in a 2-dimensional
        embedding. Each row corresponds to a dimension and each column
        represents a cell.
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

    if num_epochs is None:
        num_epochs = -1

    return lib.run_umap(
        nnidx,
        nndist,
        num_dim,
        min_dist,
        seed,
        num_epochs,
        num_threads,
        parallel_optimization
    )
