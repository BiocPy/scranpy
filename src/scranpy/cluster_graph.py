from typing import Literal
from dataclasses import dataclass

import numpy

from . import lib_scranpy as lib
from .build_snn_graph import GraphComponents


@dataclass
class ClusterGraphResults:
    """Clustering results from :py:func:`~cluster_graph`."""

    status: int
    """Status of the clustering.
    A non-zero value indicates that the algorithm did not complete successfully."""

    membership: numpy.ndarray
    """Array containing the cluster assignment for each vertex, i.e., cell.
    All values are in [0, N) where N is the total number of clusters."""


@dataclass
class ClusterGraphMultilevelResults(ClusterGraphResults):
    """Clustering results from :py:func:`~cluster_graph` when ``method = "multilevel"``."""

    levels: tuple[numpy.ndarray]
    """Clustering at each level of the algorithm.
    Each array corresponds to one level and contains the cluster assignment for each cell at that level."""

    modularity: numpy.ndarray
    """Modularity at each level.
    This has length equal to :py:attr:`~levels`, and the largest value corresponds to the assignments reported in :py:attr:`~ClusterGraphResults.membership`."""


@dataclass
class ClusterGraphLeidenResults(ClusterGraphResults):
    """Clustering results from :py:func:`~cluster_graph` when ``method = "leiden"``."""

    quality: float
    """Quality of the clustering.
    This is the same as the modularity if ``leiden_objective = "modularity"``."""


@dataclass
class ClusterGraphWalktrapResults(ClusterGraphResults):
    """Clustering results from :py:func:`~cluster_graph` when ``method = "walktrap"``."""

    merges: numpy.ndarray
    """Matrix containing two columns.
    Each row specifies the pair of cells or clusters that were merged at each step of the algorithm."""

    modularity: numpy.ndarray
    """Array of modularity scores at each merge step."""


def cluster_graph(
    x: GraphComponents,
    method: Literal["multilevel", "leiden", "walktrap"] = "multilevel",
    multilevel_resolution: float = 1, 
    leiden_resolution: float = 1, 
    leiden_objective: Literal["modularity", "cpm"] = "modularity",
    walktrap_steps: int = 4,
    seed: int = 42
) -> ClusterGraphResults:
    """Identify clusters of cells using a variety of community detection methods from a graph where similar cells are connected.

    Args:
        x:
            Components of the graph to be clustered, typically produced by :py:func:`~build_snn_graph.build_snn_graph`.

        method:
            Community detection algorithm to use.

        multilevel_resolution:
            Resolution of the clustering when ``method = "multilevel"``.
            Larger values result in finer clusters.

        leiden_resolution:
            Resolution of the clustering when ``method = "leiden"``.
            Larger values result in finer clusters.

        leiden_objective:
            Objective function to use when ``method = "leiden"``.

        walktrap_steps:
            Number of steps to use when ``method = "walktrap"``.

        seed:
            Random seed to use for ``method = "multilevel"`` or ``"leiden"``.

    Returns:
        Clustering results, as a:

        - :py:class:`~ClusterGraphMultilevelResults`, if ``method = "multilevel"``.
        - :py:class:`~ClusterGraphLeidenResults`, if ``method = "leiden"``.
        - :py:class:`~ClusterGraphWalktrapResults`, if ``method = "walktrap"``.

        All objects contain at least ``status``, an indicator of whether the
        algorithm successfully completed; and ``membership``, an array of
        cluster assignments for each node in ``x``.

    References:
        The various ``cluster_*`` functions in the `scran_graph_cluster <https://github.com/libscran/scran_graph_cluster>`_ C++ library, which provides some more details on each algorithm.
    """
    graph = (x.vertices, x.edges, x.weights)

    if method == "multilevel":
        out = lib.cluster_multilevel(graph, multilevel_resolution, seed)
        return ClusterGraphMultilevelResults(*out)

    elif method == "leiden":
        out = lib.cluster_leiden(graph, leiden_resolution, leiden_objective == "cpm", seed)
        return ClusterGraphLeidenResults(*out)

    elif method == "walktrap":
        out = lib.cluster_walktrap(graph, walktrap_steps)
        return ClusterGraphWalktrapResults(*out)

    else:
        raise NotImplementedError("unsupported community detection method '" + method + "'")
