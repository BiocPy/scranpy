from numpy import ctypeslib, ndarray, copy, float64, int32, uintp
from dataclasses import dataclass

from .. import cpphelpers as lib
from .._logging import logger
from ..types import MatrixTypes
from ..utils import factorize, to_logical, validate_and_tatamize_input
from ..nearest_neighbors import (
    build_neighbor_index,
    BuildNeighborIndexOptions,
)


@dataclass
class CombineEmbeddingsOptions:
    """Options for 
    :py:meth:`~scranpy.dimensionality_reduction.combine_embeddings.combine_embeddings`.

    Attributes:
        neighbors (int): Number of neighbors to use for approximating the relative variance.

        approximate (bool): Whether to perform an approximate neighbor search.

        num_threads (int): Number of threads to use for the neighbor search.
    """
    neighbors: int = 20
    approximate: bool = True
    num_threads: int = True


def combine_embeddings(
    embeddings: list[ndarray],
    options: CombineEmbeddingsOptions = CombineEmbeddingsOptions(),
) -> ndarray:
    """Combine multiple embeddings for the same set of cells (e.g., from multi-modal datasets)
    for integrated downstream analyses like clustering and visualization.
    This is done after adjusting for differences in local variance between embeddings.

    Args:
        embeddings (list[ndarray]):
            List of embeddings to be combined. Each embedding should be a
            row-major matrix where rows are cells and columns are dimensions.
            All embeddings should have the same number of rows.

        options (CombineEmbeddingsOptions):
            Further options.
            
    Returns:
        ndarray: Array containing the combined embedding, where rows are cells
        and columns are the dimensions from all embeddings.
    """
    indices = []
    ncells = None 
    nembed = len(embeddings)
    ind_ptr = ndarray(nembed, dtype=uintp)
    
    embeddings2 = []
    all_dims = ndarray(nembed, dtype=int32)
    emb_ptr = ndarray(nembed, dtype=uintp)

    for i, x in enumerate(embeddings):
        if len(x.shape) != 2:
            raise ValueError("all embeddings should be two-dimensional matrices")
        if ncells is None:
            ncells = x.shape[0]
        elif ncells != x.shape[0]:
            raise ValueError("all embeddings should have the same number of rows")

        x = x.astype(float64, copy=False)
        embeddings2.append(x)
        emb_ptr[i] = x.ctypes.data
        all_dims[i] = x.shape[1]

        cur_ind = build_neighbor_index(
            x,
            options = BuildNeighborIndexOptions(
                approximate = options.approximate
            )
        )
        indices.append(cur_ind)
        ind_ptr[i] = cur_ind.ptr

    scaling = ndarray(nembed, dtype=float64)
    lib.scale_by_neighbors(
        nembed,
        ind_ptr.ctypes.data,
        options.neighbors,
        scaling,
        options.num_threads,
    )

    output = ndarray((ncells, all_dims.sum()), dtype=float64)
    lib.combine_embeddings(
        nembed,
        all_dims,
        ncells,
        emb_ptr.ctypes.data,
        scaling,
        output
    )

    return output
