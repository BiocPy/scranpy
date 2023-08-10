from dataclasses import dataclass

from ..types import validate_object_type
from .build_snn_graph import BuildSnnGraphArgs

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


@dataclass
class ClusterStepArgs:
    """Arguments to run the clustering step.

    Attributes:
        build_snn_graph (BuildSNNGraphArgs): Arguments to build the SNN graph 
        (:py:meth:`~scranpy.clustering.build_snn_graph.build_snn_graph`).
    """

    build_snn_graph: BuildSnnGraphArgs = BuildSnnGraphArgs()

    def __post_init__(self):
        validate_object_type(self.build_snn_graph, BuildSnnGraphArgs)


def run(input, options: ClusterStepArgs = ClusterStepArgs()):
    pass
