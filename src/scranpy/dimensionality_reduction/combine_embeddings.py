from dataclasses import dataclass
from typing import Optional, List

from numpy import array, float64, int32, ndarray, ones, uintp, zeros

from .. import _cpphelpers as lib
from ..nearest_neighbors import (
    BuildNeighborIndexOptions,
    build_neighbor_index,
)


@dataclass
class CombineEmbeddingsOptions:
    """Options for :py:meth:`~scranpy.dimensionality_reduction.combine_embeddings.combine_embeddings`.

    Attributes:
        neighbors:
            Number of neighbors to use for approximating the relative variance.

        approximate:
            Whether to perform an approximate neighbor search.

        weights:
            Weights to apply to each entry of ``embeddings``. If None,
            all embeddings recieve equal weight. If any weight is zero, the corresponding embedding
            is omitted from the return value.

        num_threads:
            Number of threads to use for the neighbor search.
    """

    neighbors: int = 20
    approximate: bool = True
    weights: Optional[List[float]] = None
    num_threads: int = True


def combine_embeddings(
    embeddings: List[ndarray],
    options: CombineEmbeddingsOptions = CombineEmbeddingsOptions(),
) -> ndarray:
    """Combine multiple embeddings for the same set of cells (e.g., from multi-modal datasets) for integrated downstream
    analyses like clustering and visualization. This is done after adjusting for differences in local variance between
    embeddings.

    Args:
        embeddings:
            List of embeddings to be combined. Each embedding should be a
            row-major matrix where rows are cells and columns are dimensions.
            All embeddings should have the same number of rows.

        options:
            Optional parameters.

    Returns:
        Array containing the combined embedding, where rows are cells and
        columns are the dimensions from all embeddings with non-zero weight.
    """
    ncells = embeddings[0].shape[0]
    for x in embeddings:
        if len(x.shape) != 2:
            raise ValueError("all embeddings should be two-dimensional matrices")
        elif ncells != x.shape[0]:
            raise ValueError("all embeddings should have the same number of rows")

    if options.weights is None:
        weights = ones(len(embeddings), dtype=float64)
    else:
        if len(embeddings) != len(options.weights):
            raise ValueError(
                "'options.weights' should have the same length as 'embeddings'"
            )

        new_weights = []
        new_embeddings = []
        for i, w in enumerate(options.weights):
            if w > 0:
                new_embeddings.append(embeddings[i])
                new_weights.append(w)

        embeddings = new_embeddings
        weights = array(new_weights, dtype=float64)

    indices = []
    nembed = len(embeddings)
    ind_ptr = zeros(nembed, dtype=uintp)

    embeddings2 = []
    all_dims = zeros(nembed, dtype=int32)
    emb_ptr = zeros(nembed, dtype=uintp)

    for i, x in enumerate(embeddings):
        x = x.astype(float64, copy=False)
        embeddings2.append(x)
        emb_ptr[i] = x.ctypes.data
        all_dims[i] = x.shape[1]

        cur_ind = build_neighbor_index(
            x, options=BuildNeighborIndexOptions(approximate=options.approximate)
        )
        indices.append(cur_ind)
        ind_ptr[i] = cur_ind.ptr

    scaling = zeros(nembed, dtype=float64)
    lib.scale_by_neighbors(
        nembed,
        ind_ptr.ctypes.data,
        options.neighbors,
        scaling,
        options.num_threads,
    )
    scaling *= weights

    output = zeros((ncells, all_dims.sum()), dtype=float64)
    lib.combine_embeddings(
        nembed, all_dims, ncells, emb_ptr.ctypes.data, scaling, output
    )

    return output
