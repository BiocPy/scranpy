from dataclasses import dataclass

from .._abstract import AbstractStepOptions
from .build_neighbor_index import BuildNeighborIndexOptions
from .find_nearest_neighbors import FindNearestNeighborsOptions

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


@dataclass
class NearestNeighborStepOptions(AbstractStepOptions):
    """Arguments to run the nearest neighbor step.

    Attributes:
        build_nn_index (BuildNeighborIndexOptions): Arguments to build nearest
            neighbor index
            (:py:meth:`~scranpy.nearest_neighbors.build_neighbor_index.build_neighbor_index`).
        find_nn (FindNearestNeighborsOptions): Arguments to find nearest neighbors
            (:py:meth:`~scranpy.nearest_neighbors.find_nearest_neighbors.find_nearest_neighbors`).
    """

    build_nn_index: BuildNeighborIndexOptions = BuildNeighborIndexOptions()
    find_nn: FindNearestNeighborsOptions = FindNearestNeighborsOptions()

    def __post_init__(self):
        from ..types import validate_object_type

        validate_object_type(self.build_nn_index, BuildNeighborIndexOptions)
        validate_object_type(self.find_nn, FindNearestNeighborsOptions)

    def set_threads(self, num_threads: int = 1):
        """Set number of threads to use.

        Args:
            num_threads (int, optional): Number of threads. Defaults to 1.
        """
        self.find_nn.num_threads = num_threads

    def set_verbose(self, verbose: bool = False):
        """Set verbose to display logs.

        Args:
            verbose (bool, optional): Display logs? Defaults to False.
        """
        self.build_nn_index.verbose = verbose
        self.find_nn.verbose = verbose
