from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import igraph as ig
import numpy as np

from .. import cpphelpers as lib
from .._logging import logger
from ..nearest_neighbors import (
    BuildNeighborIndexArgs,
    NeighborIndex,
    NeighborResults,
    build_neighbor_index,
)
from ..types import NeighborIndexOrResults, is_neighbor_class

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


@dataclass
class BuildSnnGraphArgs:
    """Arguments to build a shared nearest neighbor
    graph - :py:meth:`~scranpy.clustering.build_snn_graph.build_snn_graph`.

    Attributes:
        num_neighbors (int, optional): Number of neighbors to use.
            Ignored if ``input`` is a
            :py:class:`~scranpy.nearest_neighbors.find_nearest_neighbors.NeighborResults`
            object. Defaults to 15.
        approximate (bool, optional): Whether to build an index for an approximate
            neighbor search. Defaults to True.
        weight_scheme (Literal["ranked", "jaccard", "number"], optional): Weighting
            scheme for the edges between cells. This can be based on the top ranks
            of the shared neighbors ("rank"), the number of shared neighbors ("number")
            or the Jaccard index of the neighbor sets between cells ("jaccard").
            Defaults to "ranked".
        num_threads (int, optional): Number of threads to use. Defaults to 1.
        verbose (bool): Display logs? Defaults to False.

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
    input: NeighborIndexOrResults, options: BuildSnnGraphArgs = BuildSnnGraphArgs()
) -> ig.Graph:
    """Build Shared nearest neighbor graph.

    ``input`` is either a pre-built neighbor search index for the dataset
    (:py:class:`~scranpy.nearest_neighbors.build_neighbor_index.NeighborIndex`), or a
    pre-computed set of neighbor search results for all cells
    (:py:class:`~scranpy.nearest_neighbors.find_nearest_neighbors.NeighborResults`).
    If ``input`` is a matrix (:py:class:`numpy.ndarray`),
    we compute the nearest neighbors for each cell, assuming it represents the
    coordinates for each cell, usually the result of PCA step
    (:py:meth:`~scranpy.dimensionality_reduction.run_pca.run_pca`).

    Note: rows are features, columns are cells.

    Args:
        input (NeighborIndexOrResults): Input matrix, pre-computed neighbor index
            or neighbors.
        options (BuildSnnGraphArgs): Optional parameters.

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
                input, BuildNeighborIndexArgs(approximate=options.approximate)
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
