from dataclasses import dataclass, field
from typing import Literal, Union

from igraph import Graph
from numpy import copy, ctypeslib, ndarray

from .. import _cpphelpers as lib
from ..nearest_neighbors import (
    BuildNeighborIndexOptions,
    NeighborIndex,
    NeighborResults,
    build_neighbor_index,
)

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


@dataclass
class BuildSnnGraphOptions:
    """Optional arguments for :py:meth:`~scranpy.clustering.build_snn_graph.build_snn_graph`.

    Attributes:
        num_neighbors:
            Number of neighbors to use.
            Larger values result in a more interconnected graph and generally broader clusters from community detection.
            Ignored if ``input`` is a :py:class:`~scranpy.nearest_neighbors.find_nearest_neighbors.NeighborResults`
            object. Defaults to 15.

        weight_scheme:
            Weighting scheme for the edges between cells. This can be based on the top ranks
            of the shared neighbors ("rank"), the number of shared neighbors ("number")
            or the Jaccard index of the neighbor sets between cells ("jaccard").
            Defaults to "ranked".

        build_neighbor_index_options:
            Optional arguments to use for building a nearest neighbors index.
            Only used if ``input`` is a :py:class:`~numpy.ndarray`.

        num_threads:
            Number of threads to use for the SNN graph construction.
            This is also used for the neighbor search if ``input`` is not already a
            :py:class:`~scranpy.nearest_neighbors.find_nearest_neighbors.NeighborResults`.
            Defaults to 1.

    Raises:
        ValueError:
            If ``weight_scheme`` is not an expected value.
    """

    num_neighbors: int = 15
    weight_scheme: Literal["ranked", "jaccard", "number"] = "ranked"
    build_neighbor_index_options: BuildNeighborIndexOptions = field(
        default_factory=BuildNeighborIndexOptions
    )
    num_threads: int = 1

    def __post_init__(self):
        if self.weight_scheme not in ["ranked", "jaccard", "number"]:
            raise ValueError(
                '\'weight_scheme\' must be one of "ranked", "jaccard", "number"'
                f"provided {self.weight_scheme}"
            )

    def set_threads(self, num_threads: int):
        self.num_threads = num_threads


def build_snn_graph(
    input: Union[NeighborIndex, NeighborResults, ndarray],
    options: BuildSnnGraphOptions = BuildSnnGraphOptions(),
) -> Graph:
    """Build a shared nearest neighbor (SNN) graph where each cell is a node and edges are formed between cells that
    share one or more nearest neighbors. This can be used for community detection to define clusters of similar cells.

    Args:
        input:
            Object containing per-cell nearest neighbor results or data that can be used to derive them.

            This may be a a 2-dimensional :py:class:`~numpy.ndarray` containing per-cell
            coordinates, where rows are cells and columns are dimensions.
            This is most typically the result of the PCA step
            (:py:meth:`~scranpy.dimensionality_reduction.run_pca.run_pca`).

            Alternatively, ``input`` may be a pre-built neighbor search index
            (:py:class:`~scranpy.nearest_neighbors.build_neighbor_index.NeighborIndex`)
            for the dataset, typically constructed from the PC coordinates for all cells.

            Alternatively, ``input`` may be a pre-computed set of neighbor
            search results
            (:py:class:`~scranpy.nearest_neighbors.find_nearest_neighbors.NeighborResults`).
            for all cells in the dataset.

        options:
            Optional parameters.

    Raises:
        TypeError:
            If ``input`` is not a nearest neighbor search index or search result
            (:py:class:`~scranpy.nearest_neighbors.build_neighbor_index.NeighborIndex`,
            :py:class:`~scranpy.nearest_neighbors.find_nearest_neighbors.NeighborResults`).

    Returns:
        An igraph object.
    """
    graph = None
    scheme = options.weight_scheme.encode("UTF-8")

    if not isinstance(input, NeighborResults):
        if not isinstance(input, NeighborIndex):
            input = build_neighbor_index(input, options.build_neighbor_index_options)
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
        idx_array = ctypeslib.as_array(idx_pointer, shape=(nedges * 2,))
        w_pointer = lib.fetch_snn_graph_weights(built)
        w_array = ctypeslib.as_array(w_pointer, shape=(nedges,))

        edge_list = []
        for i in range(nedges):
            edge_list.append((idx_array[2 * i], idx_array[2 * i + 1]))

        nc = input.num_cells()

        graph = Graph(n=nc, edges=edge_list)
        graph.es["weight"] = copy(w_array)

    finally:
        lib.free_snn_graph(built)

    return graph
