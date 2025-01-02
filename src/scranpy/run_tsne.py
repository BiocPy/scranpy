from typing import Optional, Union

import knncolle
import numpy

from . import lib_scranpy as lib
from ._utils_neighbors import _check_indices


def run_tsne(
    x: Union[numpy.ndarray, knncolle.FindKnnResults, knncolle.Index],
    perplexity: float = 30,
    num_neighbors: Optional[int] = None,
    max_depth: int = 20,
    leaf_approximation: bool = False,
    max_iterations: int = 500,
    seed: int = 42,
    num_threads: int = 1, 
    nn_parameters: knncolle.Parameters = knncolle.AnnoyParameters()
) -> numpy.ndarray:
    """Compute t-SNE coordinates to visualize similarities between cells.

    Args:
        x: 
            Numeric matrix where rows are dimensions and columns are cells, typically containing a low-dimensional representation from, e.g., :py:func:`~scranpy.run_pca.run_pca`.

            Alternatively, a :py:class:`~knncolle.find_knn.FindKnnResults` object containing existing neighbor search results.
            The number of neighbors should be the same as ``num_neighbors``, otherwise a warning is raised.

            Alternatively, a :py:class:`~knncolle.Index.Index` object.

        perplexity:
            Perplexity to use in the t-SNE algorithm.
            Larger values cause the embedding to focus on global structure.

        num_neighbors
            Number of neighbors in the nearest-neighbor graph.
            Typically derived from ``perplexity`` using :py:func:`~tsne_perplexity_to_neighbors`.

        max_depth:
            Maximum depth of the Barnes-Hut quadtree.
            Smaller values (7-10) improve speed at the cost of accuracy.

        leaf_approximation:
            Whether to use the "leaf approximation" approach, which sacrifices some accuracy for greater speed.
            Only effective when ``max_depth`` is small enough for multiple cells to be assigned to the same leaf node of the quadtree.

        max_iterations:
            Maximum number of iterations to perform.

        seed:
            Random seed to use for generating the initial coordinates.

        num_threads:
            Number of threads to use.

        nn_parameters:
            The algorithm to use for the nearest-neighbor search.
            Only used if ``x`` is not a pre-built nearest-neighbor search index or a list of existing nearest-neighbor search results.

    Returns:
        Array containing the coordinates of each cell in a 2-dimensional embedding.
        Each row corresponds to a dimension and each column represents a cell.

    References:
        https://github.com/libscran/qdtsne, for some more details on the approximations.
    """
    if num_neighbors is None:
        num_neighbors = tsne_perplexity_to_neighbors(perplexity)

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

    return lib.run_tsne(
        nnidx,
        nndist,
        perplexity,
        leaf_approximation,
        max_depth,
        max_iterations,
        seed,
        num_threads
    )


def tsne_perplexity_to_neighbors(perplexity: float) -> int:
    """Determine the number of nearest neighbors required to support a given
    perplexity in the t-SNE algorithm.

    Args:
        perplexity:
            Perplexity to use in :py:func:`~run_tsne`.

    Returns:
        The corresponding number of nearest neighbors.
    """
    return lib.perplexity_to_neighbors(perplexity)
