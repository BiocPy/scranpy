import ctypes as ct
from collections import namedtuple
from dataclasses import dataclass

from numpy import float64, int32, ndarray

from .. import cpphelpers as lib
from .._logging import logger
from .build_neighbor_index import NeighborIndex

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"

SingleNeighborResults = namedtuple("SingleNeighborResults", ["index", "distance"])
SingleNeighborResults.__doc__ = """\
Named tuple of nearest neighbors for a single cell.

index (ndarray): 
    Array containing 0-based indices of a cell's neighbor neighbors,
    ordered by increasing distance.
distance (ndarray): 
    Array containing distances to a cell's nearest neighbors,
    ordered by increasing distance.
"""

SerializedNeighborResults = namedtuple(
    "SerializedNeighborResults", ["index", "distance"]
)
SerializedNeighborResults.__doc__ = """\
Named tuple of serialized results from the nearest neighbor search.

index (ndarray): 
    Row-major matrix containing 0-based indices of the neighbor neighbors for each cell.
    Each row is a cell and each column is a neighbor, ordered by increasing distance.
distance (ndarray): 
    Row-major matrix containing distances to the nearest neighbors for each cell.
    Each row is a cell and each column is a neighbor, ordered by increasing distance.
"""


class NeighborResults:
    """Nearest neighbor search results.
    This should not be constructed manually but instead should be created by
    :py:meth:`~scranpy.nearest_neighbors.find_nearest_neighbors.find_nearest_neighbors`.
    """

    def __init__(self, ptr: ct.c_void_p):
        self.__ptr = ptr

    def __del__(self):
        lib.free_neighbor_results(self.__ptr)

    @property
    def ptr(self) -> ct.c_void_p:
        """Get pointer to scran's NN search index.
        Returns:
            ct.c_void_p: Pointer reference.
        """
        return self.__ptr

    def num_cells(self) -> int:
        """Get the number of cells in this object.

        Returns:
            int: Number of cells.
        """
        return lib.fetch_neighbor_results_nobs(self.__ptr)

    def num_neighbors(self) -> int:
        """Get the number of neighbors used to build this object.

        Returns:
            int: Number of neighbors.
        """
        return lib.fetch_neighbor_results_k(self.__ptr)

    def get(self, i: int) -> SingleNeighborResults:
        """Get the nearest neighbors for a particular cell.

        Args:
            i (int): Index of the cell of interest.

        Returns:
            SingleNeighborResults: A tuple with indices and distances.
        """
        k = lib.fetch_neighbor_results_k(self.__ptr)
        out_d = ndarray((k,), dtype=float64)
        out_i = ndarray((k,), dtype=int32)
        lib.fetch_neighbor_results_single(self.__ptr, i, out_i, out_d)
        return SingleNeighborResults(out_i, out_d)

    def serialize(self) -> SerializedNeighborResults:
        """Serialize nearest neighbors for all cells, typically to
        save or transfer to a new process.
        This can be used to construct a new
        :py:class:`~scranpy.nearest_neighbors.find_nearest_neighbors.NeighborResults`
        object by calling
        :py:meth:`~scranpy.nearest_neighbors.find_nearest_neighbors.NeighborResults.unserialize`.

        Returns:
            SerializedNeighborResults: A tuple with indices and distances.
        """
        nobs = lib.fetch_neighbor_results_nobs(self.__ptr)
        k = lib.fetch_neighbor_results_k(self.__ptr)

        # C++ stores this data as column-major k*nobs, but NumPy defaults to 
        # row-major, so we just flip the dimensions to keep everyone happy.
        out_i = ndarray((nobs, k), dtype=int32)
        out_d = ndarray((nobs, k), dtype=float64)

        lib.serialize_neighbor_results(self.__ptr, out_i, out_d)
        return SerializedNeighborResults(out_i, out_d)

    @classmethod
    def unserialize(cls, content: SerializedNeighborResults) -> "NeighborResults":
        """Initialize an instance of this class from serialized nearest neighbor results.

        Args:
            content (SerializedNeighborResults): Result of
                :py:meth:`~scranpy.nearest_neighbors.find_nearest_neighbors.NeighborResults.serialize`.

        Returns:
            NeighborResults: Instance of this class, constructed from the data in ``content``.
        """
        idx = content.index
        if not idx.flags.c_contiguous:
            raise ValueError("expected 'content.index' to have a row-major layout")

        dist = content.distance
        if dist.shape != idx.shape: 
            raise ValueError("expected 'content.distance' and 'content.index' to have the same shape")
        if not dist.flags.c_contiguous:
            raise ValueError("expected 'content.index' to have a row-major layout")

        ptr = lib.unserialize_neighbor_results(idx.shape[0], idx.shape[1], idx, dist)
        return cls(ptr)


@dataclass
class FindNearestNeighborsOptions:
    """Optional arguments for
    :py:meth:`~scranpy.nearest_neighbors.find_nearest_neighbors.find_nearest_neighbors`.

    Attributes:
        num_threads (int, optional): Number of threads to use. Defaults to 1.
        verbose (bool, optional): Whether to print logs. Defaults to False.
    """

    num_threads: int = 1
    verbose: bool = False


def find_nearest_neighbors(
    idx: NeighborIndex,
    k: int,
    options: FindNearestNeighborsOptions = FindNearestNeighborsOptions(),
) -> NeighborResults:
    """Find the nearest neighbors for each cell.

    Args:
        idx (NeighborIndex): The nearest neighbor search index, usually built by
            :py:meth:`~scranpy.nearest_neighbors.build_neighbor_index.build_neighbor_index`.
        k (int): Number of neighbors to find for each cell.
        options (FindNearestNeighborsOptions): Optional parameters.

    Raises:
        TypeError: If ``idx`` is not a nearest neighbor index.

    Returns:
        NeighborResults: 'k' nearest neighbors for each cell.
    """
    if options.verbose is True:
        logger.info("Finding nearest neighbors...")

    if not isinstance(idx, NeighborIndex):
        raise TypeError(
            "'idx' is not a nearest neighbor index, "
            "run the `build_neighbor_index` function first."
        )

    ptr = lib.find_nearest_neighbors(idx.ptr, k, options.num_threads)
    return NeighborResults(ptr)
