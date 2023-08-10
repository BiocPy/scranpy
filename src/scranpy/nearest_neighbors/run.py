from dataclasses import dataclass

from .build_neighbor_index import BuildNeighborIndexArgs
from .find_nearest_neighbors import FindNearestNeighborsArgs

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


@dataclass
class NearestNeighborStepArgs:
    """Arguments to run the nearest neighbor step.

    Attributes:
        build_nn_index (BuildNeighborIndexArgs): Arguments to build nearest
            neighbor index
            (:py:meth:`~scranpy.nearest_neighbors.build_neighbor_index.build_neighbor_index`).
        find_nn (FindNearestNeighborsArgs): Arguments to find nearest neighbors
            (:py:meth:`~scranpy.nearest_neighbors.find_nearest_neighbors.find_nearest_neighbors`).
    """

    build_nn_index: BuildNeighborIndexArgs = BuildNeighborIndexArgs()
    find_nn: FindNearestNeighborsArgs = FindNearestNeighborsArgs()

    def __post_init__(self):
        from ..types import validate_object_type

        validate_object_type(self.build_nn_index, BuildNeighborIndexArgs)
        validate_object_type(self.find_nn, FindNearestNeighborsArgs)
