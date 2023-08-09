from dataclasses import dataclass
from typing import Literal

from ..types import validate_object_type

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


@dataclass
class BuildSnnGraphArgs:
    """Arguments to build a shared nearest neighbor graph.

    Attributes:
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
        verbose (bool): display logs? Defaults to False.

    Raises:
        ValueError: if `weight_scheme` is not an expected value.
    """

    num_neighbors: int = 10
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


@dataclass
class ClusterStepArgs:
    """Arguments to run the clustering step.

    Attributes:
        build_snn_graph (BuildSNNGraphArgs): Arguments to build the SNN graph.
    """

    build_snn_graph: BuildSnnGraphArgs = BuildSnnGraphArgs()

    def __post_init__(self):
        validate_object_type(self.build_snn_graph, BuildSnnGraphArgs)
