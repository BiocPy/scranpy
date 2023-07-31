import ctypes as ct
from copy import deepcopy
from typing import Literal

import igraph as ig
import numpy as np

from ..cpphelpers import lib
from ..nearest_neighbors import (
    NeighborIndex,
    NeighborResults,
    build_neighbor_index,
    find_nearest_neighbors,
)
from ..types import NeighborIndexOrResults, is_neighbor_class

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def build_snn_graph(
    input: NeighborIndexOrResults,
    num_neighbors: int = 10,
    approximate: bool = True,
    weight_scheme: Literal["ranked", "jaccard", "number"] = "ranked",
    num_threads: int = 1,
) -> ig.Graph:
    """Build Shared nearest neighbor graph.

    `input` is either a pre-built neighbor search index for the dataset, or a
    pre-computed set of neighbor search results for all cells. If `input` is a matrix,
    we compute the nearest neighbors for each cell, assuming it represents the
    coordinates for each cell, usually the result of PCA step.
    rows are variables, columns are cells.

    Args:
        input (NeighborIndexOrResults): Input matrix or pre-computed neighbors.
        num_neighbors (int, optional): Number of neighbors to use.
            Ignored if `input` is a `NeighborResults` object. Defaults to 15.
        approximate (bool, optional): Whether to build an index for an approximate
            neighbor search. Defaults to True.
        weight_scheme (Literal["ranked", "jaccard", "number"], optional): Weighting
            scheme for the edges between cells. This can be based on the top ranks
            of the shared neighbors ("rank"), the number of shared neighbors ("number")
            or the Jaccard index of the neighbor sets between cells ("jaccard").
            Defaults to "ranked".
        num_threads (int, optional): Number of threads to use. Defaults to 1.

    Raises:
        TypeError, ValueError: If inputs do not match expectations.

    Returns:
        ig.Graph: An igraph object.
    """
    if not is_neighbor_class(input):
        raise TypeError(
            "`input` must be either the nearest neighbor search index, search results "
            "or a matrix."
        )

    if weight_scheme not in ["ranked", "jaccard", "number"]:
        raise ValueError(
            '\'weight_scheme\' must be one of "ranked", "jaccard", "number"'
            f"provided {weight_scheme}"
        )

    graph = None
    scheme = weight_scheme.encode("UTF-8")

    if not isinstance(input, NeighborResults):
        if not isinstance(input, NeighborIndex):
            input = build_neighbor_index(input, approximate=approximate)
        built = lib.build_snn_graph_from_nn_index(
            input.ptr, num_neighbors, scheme, num_threads
        )
    else:
        built = lib.build_snn_graph_from_nn_results(input.ptr, scheme, num_threads)

    try:
        nedges = lib.fetch_snn_graph_edges(built)
        idx_pointer = ct.cast(lib.fetch_snn_graph_indices(built), ct.POINTER(ct.c_int))
        idx_array = np.ctypeslib.as_array(idx_pointer, shape=(nedges * 2,))
        w_pointer = ct.cast(lib.fetch_snn_graph_weights(built), ct.POINTER(ct.c_double))
        w_array = np.ctypeslib.as_array(w_pointer, shape=(nedges,))

        edge_list = []
        for i in range(nedges):
            edge_list.append((idx_array[2 * i], idx_array[2 * i + 1]))

        nc = input.num_cells()
        graph = ig.Graph(n=nc, edges=edge_list)
        graph.es["weight"] = deepcopy(w_array)

    finally:
        lib.free_snn_graph(built)

    return graph
