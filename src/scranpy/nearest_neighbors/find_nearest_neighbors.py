import ctypes as ct
from collections import namedtuple
from dataclasses import dataclass

import numpy as np

from .. import cpphelpers as lib
from .._logging import logger
from .build_neighbor_index import NeighborIndex

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


NNResult = namedtuple("NNResult", ["index", "distance"])
NNResult.__doc__ = """\
Named tuple of results from nearest neighbor search.

index (np.ndarray): Indices of the neighbors for each cell.
distance (np.ndarray): Distances to the neighbors for each cell.
"""


class NeighborResults:
    """Class to manage the find nearest neighbor search results from scran.

    Options(AbstractStepOptions)
        ptr (ct.c_void_p): Pointer reference to libscran's `find_nearest_neighbor`
            result.
    """

    def __init__(self, ptr: ct.c_void_p):
        """Initialize the class."""
        self.__ptr = ptr

    def __del__(self):
        """Free the reference."""
        lib.free_neighbor_results(self.__ptr)

    @property
    def ptr(self) -> ct.c_void_p:
        """Get pointer to scran's NN search index.

        Returns:
            ct.c_void_p: Pointer reference.
        """
        return self.__ptr

    def num_cells(self) -> int:
        """Get number of cells.

        Returns:
            int: Number of cells
        """
        return lib.fetch_neighbor_results_nobs(self.__ptr)

    def num_neighbors(self) -> int:
        """Get number of neighbors.

        Returns:
            int: Number of neighbors.
        """
        return lib.fetch_neighbor_results_k(self.__ptr)

    def get(self, i: int) -> NNResult:
        """Get nearest neighbors of i-th observation/cell.

        Options(AbstractStepOptions)
            i (int): i-th observation to access.

        Returns:
            NNResult: A tuple with indices and distance.
        """
        k = lib.fetch_neighbor_results_k(self.__ptr)
        out_d = np.ndarray((k,), dtype=np.float64)
        out_i = np.ndarray((k,), dtype=np.int32)
        lib.fetch_neighbor_results_single(self.__ptr, i, out_i, out_d)
        return NNResult(out_i, out_d)

    def serialize(self) -> NNResult:
        """Get/serialize nearest neighbors for all observations/cells.

        Returns:
            NNResult: A tuple with indices and distance.
        """
        nobs = lib.fetch_neighbor_results_nobs(self.__ptr)
        k = lib.fetch_neighbor_results_k(self.__ptr)
        out_i = np.ndarray((k, nobs), dtype=np.int32)
        out_d = np.ndarray((k, nobs), dtype=np.float64)
        lib.serialize_neighbor_results(self.__ptr, out_i, out_d)
        return NNResult(out_i, out_d)

    @classmethod
    def unserialize(cls, content: NNResult) -> "NeighborResults":
        """Initialize a class from serialized NN results.

        Options(AbstractStepOptions)
            content (NNResult): Usually the result of `serialize` method.

        Returns:
            NeighborResults: Result object.
        """
        idx = content.index
        dist = content.distance
        ptr = lib.unserialize_neighbor_results(idx.shape[0], idx.shape[1], idx, dist)
        return cls(ptr)


@dataclass
class FindNearestNeighborsOptions:
    """Arguments to find nearest neighbors -
    :py:meth:`~scranpy.nearest_neighbors.find_nearest_neighbors.find_nearest_neighbors`.

    Attributes:
        k (int): Number of neighbors to find.
        num_threads (int, optional): Number of threads to use. Defaults to 1.
        verbose (bool, optional): Display logs?. Defaults to False.
    """

    k: int = 10
    num_threads: int = 1
    verbose: bool = False


def find_nearest_neighbors(
    idx: NeighborIndex,
    options: FindNearestNeighborsOptions = FindNearestNeighborsOptions(),
) -> NeighborResults:
    """Find the nearest neighbors for each cell.

    Args:
        idx (NeighborIndex): Object that holds the nearest neighbor search index.
            usually the result of
            :py:meth:`~scranpy.nearest_neighbors.build_neighbor_index.build_neighbor_index`.
        options (FindNearestNeighborsOptions): Optional parameters.

    Returns:
        NeighborResults: Object with search results.

    Raises:
        TypeError: If ``idx`` is not a nearest neighbor index.
    """
    if options.verbose is True:
        logger.info("Finding nearest neighbors...")

    if not isinstance(idx, NeighborIndex):
        raise TypeError(
            "'idx' is not a nearest neighbor index, "
            "run the `build_neighbor_index` function first."
        )

    ptr = lib.find_nearest_neighbors(idx.ptr, options.k, options.num_threads)
    return NeighborResults(ptr)
