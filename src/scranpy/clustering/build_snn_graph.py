from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import igraph as ig
import numpy as np

from .. import cpphelpers as lib
from .._logging import logger
from ..nearest_neighbors import (
    BuildNeighborIndexOptions,
    NeighborIndex,
    NeighborResults,
    NeighborlyInputs, 
    build_neighbor_index,
)

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


@dataclass
class BuildSnnGraphOptions:
    """Optional arguments for building a shared nearest neighbor (SNN) graph
    via :py:meth:`~scranpy.clustering.build_snn_graph.build_snn_graph`,
    typically in preparation for clustering by community detection.

    Attributes:
        num_neighbors (int, optional): Number of neighbors to use. Larger values result 
            in a more interconnected graph and generally broader clusters from community detection.
            Ignored if ``input`` is a
            :py:class:`~scranpy.nearest_neighbors.find_nearest_neighbors.NeighborResults`
            object. Defaults to 15.
        approximate (bool, optional): Whether to use an approximate
            neighbor search, which sacrifices some accuracy for speed.
            Ignored if ``input`` is a 
            :py:class:`~scranpy.nearest_neighbors.find_nearest_neighbors.NeighborResults`
            or 
            :py:class:`~scranpy.nearest_neighbors.build_neighbor_index.NeighborIndex`
            object. Defaults to True.
        weight_scheme (Literal["ranked", "jaccard", "number"], optional): Weighting
            scheme for the edges between cells. This can be based on the top ranks
            of the shared neighbors ("rank"), the number of shared neighbors ("number")
            or the Jaccard index of the neighbor sets between cells ("jaccard").
            Defaults to "ranked".
        num_threads (int, optional): Number of threads to use for neighbor detection
            and SNN graph construction. Defaults to 1.
        verbose (bool): Whether to print logging information. Defaults to False.

    Raises:
        ValueError: If ``weight_scheme`` is not an expected value.
    """

    num_neighbors: int = 15
    approximate: bool = True
    weight_scheme: Literal["ranked", "jaccard", "number"] = "ranked"
    verbose: bool = False
    num_threads: int = 1

    def __post_init__(self):
        if self.weight_scheme not in ["ranked", "jaccard", "number"]:
            raise ValueError(
                '\'weight_scheme\' must be one of "ranked", "jaccard", "number"'
                f"provided {self.weight_scheme}"
            )


def build_snn_graph(
    input: NeighborlyInputs,
    options: BuildSnnGraphOptions = BuildSnnGraphOptions(),
) -> ig.Graph:
    """Build a shared nearest neighbor (SNN) graph where each cell is a node and
    edges are formed between cells that share one or more nearest neighbors.
    This can be used for community detection to detect clusters of similar cells.

    Args:
        input  (NeighborIndex | NeighborResults | np.ndarray):
            Object containing per-cell nearest neighbor results or data that can be used to derive them.

            This may be a a 2-dimensional :py:class:`~numpy.ndarray` containing per-cell
            coordinates, where rows are features/dimensions and columns are
            cells. This is most typically the result of the PCA step
            (:py:meth:`~scranpy.dimensionality_reduction.run_pca.run_pca`).

            Alternatively, ``input`` may be a pre-built neighbor search index
            (:py:class:`~scranpy.nearest_neighbors.build_neighbor_index.NeighborIndex`)
            for the dataset, typically constructed from the PC coordinates for all cells.

            Alternatively, ``input`` may be a pre-computed set of neighbor
            search results 
            (:py:class:`~scranpy.nearest_neighbors.find_nearest_neighbors.NeighborResults`).
            for all cells in the dataset.

        options (BuildSnnGraphOptions): Optional parameters.

    Raises:
        TypeError: If ``input`` is not a nearest neighbor search index or search result
            (:py:class:`~scranpy.nearest_neighbors.build_neighbor_index.NeighborIndex`,
            :py:class:`~scranpy.nearest_neighbors.find_nearest_neighbors.NeighborResults`).

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
            if options.verbose is True:
                logger.info("`input` is a matrix, building nearest neighbor index...")

            input = build_neighbor_index(
                input, BuildNeighborIndexOptions(approximate=options.approximate)
            )

        if options.verbose is True:
            logger.info("Building shared nearest neighbor graph...")

        built = lib.build_snn_graph_from_nn_index(
            input.ptr, options.num_neighbors, scheme, options.num_threads
        )
    else:
        if options.verbose is True:
            logger.info("Building the shared nearest neighbor graph from `input`")

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

        if options.verbose is True:
            logger.info("Generating the iGraph object...")

        graph = ig.Graph(n=nc, edges=edge_list)
        graph.es["weight"] = deepcopy(w_array)

    finally:
        lib.free_snn_graph(built)

    return graph
