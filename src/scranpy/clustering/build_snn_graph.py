from copy import deepcopy

import igraph as ig
import numpy as np

from .. import cpphelpers as lib
from ..nearest_neighbors import (
    NeighborIndex,
    NeighborResults,
    build_neighbor_index,
)
from ..types import NeighborIndexOrResults, is_neighbor_class
from .argtypes import BuildSnnGraphArgs

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def build_snn_graph(
    input: NeighborIndexOrResults, options: BuildSnnGraphArgs = BuildSnnGraphArgs()
) -> ig.Graph:
    """Build Shared nearest neighbor graph.

    `input` is either a pre-built neighbor search index for the dataset, or a
    pre-computed set of neighbor search results for all cells. If `input` is a matrix,
    we compute the nearest neighbors for each cell, assuming it represents the
    coordinates for each cell, usually the result of PCA step.
    rows are variables, columns are cells.

    Args:
        input (NeighborIndexOrResults): Input matrix or pre-computed neighbors.
        options (BuildSnnGraphArgs): Optional arguments specified by
            `BuildSnnGraphArgs`.

    Raises:
        TypeError: If `input` is not a nearest neight search index or search result.

    Returns:
        ig.Graph: An igraph object.
    """
    if not is_neighbor_class(input):
        raise TypeError(
            "`input` must be either the nearest neighbor search index, search results "
            "or a matrix."
        )

    graph = None
    scheme = options.weight_scheme.encode("UTF-8")

    if not isinstance(input, NeighborResults):
        if not isinstance(input, NeighborIndex):
            input = build_neighbor_index(input, approximate=options.approximate)
        built = lib.build_snn_graph_from_nn_index(
            input.ptr, options.num_neighbors, scheme, options.num_threads
        )
    else:
        built = lib.build_snn_graph_from_nn_results(
            input.ptr, scheme, options.num_threads
        )

    try:
        nedges = lib.fetch_snn_graph_edges(built)
        idx_pointer = lib.fetch_snn_graph_indices(built)
        idx_array = np.ctypeslib.as_array(idx_pointer, shape=(nedges * 2,))
        w_pointer = lib.fetch_snn_graph_weights(built)
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
