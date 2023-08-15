from dataclasses import dataclass, field
from typing import Optional, Sequence

from igraph import Graph

from ..types import validate_object_type
from .build_snn_graph import BuildSnnGraphOptions

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


@dataclass
class ClusteringOptions:
    """Options for clustering.

    Attributes:
        build_snn_graph (BuildSnnGraphOptions): Optional arguments to build the SNN graph
        (:py:meth:`~scranpy.clustering.build_snn_graph.build_snn_graph`).
    """

    build_snn_graph: BuildSnnGraphOptions = field(default_factory=BuildSnnGraphOptions)
    resolution: int = 1

    def __post_init__(self):
        validate_object_type(self.build_snn_graph, BuildSnnGraphOptions)

    def set_threads(self, num_threads: int = 1):
        """Number of threads to use in steps that can be parallelized.

        Args:
            num_threads (int, optional): Number of threads. Defaults to 1.
        """
        self.build_snn_graph.num_threads = num_threads

    def set_verbose(self, verbose: bool = False):
        """Set verbose to display logs.

        Args:
            verbose (bool, optional): Whether to display logs. Defaults to False.
        """
        self.build_snn_graph.verbose = verbose


@dataclass
class ClusteringResults:
    """Results of the clustering step.

    Attributes:
        build_snn_graph (Graph, optional): The output of
            :py:meth:`~scranpy.clustering.build_snn_graph.build_snn_graph`.
        clusters (Sequence, optional): Clusters identified by igraph.
    """

    build_snn_graph: Optional[Graph] = None
    clusters: Optional[Sequence] = None
