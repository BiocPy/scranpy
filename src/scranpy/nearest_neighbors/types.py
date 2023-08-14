from dataclasses import dataclass, field
from typing import Optional, Union
import numpy as np

from .build_neighbor_index import BuildNeighborIndexOptions, NeighborIndex
from .find_nearest_neighbors import FindNearestNeighborsOptions, NeighborResults

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


@dataclass
class NearestNeighborsOptions:
    """Arguments to run the nearest neighbor step.

    Attributes:
        build_neighbor_index (BuildNeighborIndexOptions): Optional arguments for building the nearest
            neighbor index
            (:py:meth:`~scranpy.nearest_neighbors.build_neighbor_index.build_neighbor_index`).
        find_nearest_neighbors (FindNearestNeighborsOptions): Optional arguments for finding nearest neighbors
            (:py:meth:`~scranpy.nearest_neighbors.find_nearest_neighbors.find_nearest_neighbors`).
    """

    build_neighbor_index: BuildNeighborIndexOptions = field(
        default_factory=BuildNeighborIndexOptions
    )
    find_nearest_neighbors: FindNearestNeighborsOptions = field(
        default_factory=FindNearestNeighborsOptions
    )

    def __post_init__(self):
        from ..types import validate_object_type

        validate_object_type(self.build_neighbor_index, BuildNeighborIndexOptions)
        validate_object_type(self.find_nearest_neighbors, FindNearestNeighborsOptions)

    def set_threads(self, num_threads: int = 1):
        """Set number of threads to use.

        Args:
            num_threads (int, optional): Number of threads. Defaults to 1.
        """
        self.find_nearest_neighbors.num_threads = num_threads

    def set_verbose(self, verbose: bool = False):
        """Set verbose to display logs.

        Args:
            verbose (bool, optional): Whether to print logs. Defaults to False.
        """
        self.build_neighbor_index.verbose = verbose
        self.find_nearest_neighbors.verbose = verbose

NeighborlyInputs = Union[NeighborIndex, NeighborResults, np.ndarray]
