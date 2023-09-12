from numpy import ctypeslib, ndarray, copy, float64, int32, uintp

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
    neighbors: int = 20
    approximate: bool = True
    num_threads: int = True


def combine_embeddings(
    embeddings: list[ndarray],
    options: CombineEmbeddingsOptions = CombineEmbeddingsOptions(),
):
    indices = []
    first_dim = 0
    nembed = len(embeddings)
    ind_ptr = ndarray(nembed, dtype=uintp)

    for i, x in enumerate(embeddings):
        if len(x.shape) != 2:
            raise ValueError("all embeddings should be two-dimensional matrices")
        if first_dim is None:
            first_dim = x.shape[0]
        elif first_dim != x.shape[0]:
            raise ValueError("all embeddings should have the same number of rows")

        x = x.astype(float64, copy=False)
        cur_ind = build_neighbor_index(
            x,
            options = BuildNeighborIndexOptions(
                approximate = options.approximate
            )
        )
        indices.append(cur_ind)
        ind_ptr[i] = cur_ind.ctypes.data

    scaling = ndarray(nembed, dtype=float64)
    lib.scale_by_neighbors(
        nembed,
        ind_ptr.ctypes.data,
        options.neighbors,
        scaling,
        options.num_threads,
    )

    # Assembling the output embeddings.
    return scaling
