from typing import Optional, Union, Literal
from dataclasses import dataclass

import knncolle
import numpy

from . import lib_scranpy as lib
from ._utils_neighbors import _check_indices


@dataclass
class GraphComponents:
    """Components of a (possibly weighted) graph.
    Typically, nodes are cells and edges are formed between cells with similar expression profiles."""

    vertices: int
    """Number of vertices in the graph (i.e., cells)."""

    edges: numpy.ndarray
    """Array of indices for graph edges.
    Pairs of values represent the endpoints of an (undirected) edge, i.e., ``edges[0:2]`` form the first edge, ``edges[2:4]`` form the second edge and so on."""

    weights: Optional[numpy.ndarray]
    """Array of weights for each edge. This has length equal to half the length of ``edges``; the first weight corresponds to the first edge, and so on.
    If ``None``, the graph is assumed to be unweighted."""

    def as_igraph(self):
        """Convert to a ``Graph`` from the **igraph** package.

        Returns:
            A ``Graph`` for use with methods in the **igraph** package.
        """
        edges = []
        for i in range(len(self.edges)//2):
            j = i * 2
            edges.append((self.edges[j], self.edges[j+1])) 

        import igraph
        g = igraph.Graph(edges, n=self.vertices, directed=False)
        if self.weights is not None:
            g.es['weight'] = self.weights
        return g


def build_snn_graph(
    x: Union[numpy.ndarray, knncolle.FindKnnResults, knncolle.Index],
    num_neighbors: int = 10,
    weight_scheme: Literal["ranked", "number", "jaccard"] = "ranked",
    num_threads: int = 1, 
    nn_parameters: knncolle.Parameters = knncolle.AnnoyParameters()
) -> GraphComponents:
    """Build a shared nearest neighbor (SNN) graph where each node is a cell.
    Edges are formed between cells that share one or more nearest neighbors, weighted by the number or importance of those shared neighbors.

    Args:
        x: 
            Numeric matrix where rows are dimensions and columns are cells, typically containing a low-dimensional representation from, e.g., :py:func:`~scranpy.run_pca.run_pca`.

            Alternatively, a :py:class:`~knncolle.find_knn.FindKnnResults` object containing existing neighbor search results.
            The number of neighbors should be the same as ``num_neighbors``, otherwise a warning is raised.

            Alternatively, a :py:class:`~knncolle.Index.Index` object.

        num_neighbors:
            Number of neighbors in the nearest-neighbor graph.
            Larger values generally result in broader clusters during community detection.

        weight_scheme:
            Weighting scheme to use for the edges of the SNN graph, based on the number or ranking of the shared nearest neighbors.

        num_threads:
            Number of threads to use.

        nn_parameters:
            The algorithm to use for the nearest-neighbor search.
            Only used if ``x`` is not a pre-built nearest-neighbor search index or a list of existing nearest-neighbor search results.

    Results:
        The components of the SNN graph, to be used in community detection.

    References:
        The ``build_snn_graph`` function in the `scran_graph_cluster <https://github.com/libscran/scran_graph_cluster>`_ C++ library, which provides some more details on the weighting.
    """
    if isinstance(x, knncolle.FindKnnResults):
        nnidx = x.index
        _check_indices(nnidx, num_neighbors)
    else:
        if not isinstance(x, knncolle.Index):
            x = knncolle.build_index(nn_parameters, x.T)
        x = knncolle.find_knn(x, num_neighbors=num_neighbors, num_threads=num_threads)
        nnidx = x.index

    ncells, edges, weights = lib.build_snn_graph(nnidx, weight_scheme, num_threads)
    return GraphComponents(ncells, edges, weights)
