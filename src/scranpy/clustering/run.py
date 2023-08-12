from dataclasses import dataclass

from .._abstract import AbstractStepOptions
from ..types import validate_object_type
from .build_snn_graph import BuildSnnGraphOptions

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


@dataclass
class ClusterStepOptions(AbstractStepOptions):
    """Arguments to run the clustering step.

    Attributes:
        build_snn_graph (BuildSNNGraphOptions): Arguments to build the SNN graph
        (:py:meth:`~scranpy.clustering.build_snn_graph.build_snn_graph`).
    """

    build_snn_graph: BuildSnnGraphOptions = BuildSnnGraphOptions()
    resolution: int = 1

    def __post_init__(self):
        validate_object_type(self.build_snn_graph, BuildSnnGraphOptions)

    def set_threads(self, num_threads: int = 1):
        """Set number of threads to use.

        Args:
            num_threads (int, optional): Number of threads. Defaults to 1.
        """
        self.build_snn_graph.num_threads = num_threads

    def set_verbose(self, verbose: bool = False):
        """Set verbose to display logs.

        Args:
            verbose (bool, optional): Display logs? Defaults to False.
        """
        self.build_snn_graph.verbose = verbose
