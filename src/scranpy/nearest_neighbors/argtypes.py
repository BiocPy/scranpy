from dataclasses import dataclass

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


@dataclass
class BuildNeighborIndexArgs:
    """Arguments to build nearest neighbor index.

    Attributes:
        approximate (bool, optional): Whether to build an index for an approximate
            neighbor search. Defaults to True.
        verbose (bool, optional): Display logs?. Defaults to False.
    """

    approximate: bool = True
    verbose: bool = False


@dataclass
class FindNearestNeighborsArgs:
    """Arguments to find nearest neighbors.

    Attributes:
        k (int): Number of neighbors to find.
        num_threads (int, optional): Number of threads to use. Defaults to 1.
        verbose (bool, optional): Display logs?. Defaults to False.
    """

    k: int = 10
    num_threads: int = 1
    verbose: bool = False


@dataclass
class NearestNeighborStepArgs:
    """Arguments to run the nearest neighbor step.

    Attributes:
        score_markers (ScoreMarkersArgs): Arguments to score markers.
    """

    build_nn_index: BuildNeighborIndexArgs = BuildNeighborIndexArgs()
    find_nn: FindNearestNeighborsArgs = FindNearestNeighborsArgs()

    def __post_init__(self):
        from ..types import validate_object_type

        validate_object_type(self.build_nn_index, BuildNeighborIndexArgs)
        validate_object_type(self.find_nn, FindNearestNeighborsArgs)
