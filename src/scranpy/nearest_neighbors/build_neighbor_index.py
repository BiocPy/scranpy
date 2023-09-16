import ctypes as ct
from dataclasses import dataclass

from numpy import ndarray

from .. import cpphelpers as lib

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


class NeighborIndex:
    """The nearest neighbor search index.

    This should not be manually constructed but should be created
    by :py:meth:`~scranpy.nearest_neighbors.build_neighbor_index.build_neighbor_index`.
    """

    def __init__(self, ptr: ct.c_void_p):
        self.__ptr = ptr

    def __del__(self):
        lib.free_neighbor_index(self.__ptr)

    @property
    def ptr(self) -> ct.c_void_p:
        """Get pointer to scran's NN search index.

        Returns:
            ct.c_void_p: Pointer reference.
        """
        return self.__ptr

    def num_cells(self) -> int:
        """Get number of cells in this index.

        Returns:
            int: Number of cells.
        """
        return lib.fetch_neighbor_index_nobs(self.__ptr)

    def num_dimensions(self) -> int:
        """Get number of dimensions in this index.

        Returns:
            int: Number of dimensions.
        """
        return lib.fetch_neighbor_index_ndim(self.__ptr)


@dataclass
class BuildNeighborIndexOptions:
    """Optional arguments for :py:meth:`~scranpy.nearest_neighbors.build_neighbor_index.build_neighbor_index`.

    Attributes:
        approximate (bool, optional): Whether to build an index for an approximate
            neighbor search. This sacrifices some accuracy for speed.
            Defaults to True.
    """

    approximate: bool = True


def build_neighbor_index(
    input: ndarray, options: BuildNeighborIndexOptions = BuildNeighborIndexOptions()
) -> NeighborIndex:
    """Build a search index for finding nearest neighbors between cells, for input into functions like
    :py:meth:`~scranpy.nearest_neighbors.find_nearest_neighbors.find_nearest_neighbors`.

    Args:
        input (ndarray): A matrix where rows are cells and dimensions are columns.
            This is usually the principal components matrix from
            :py:meth:`~scranpy.dimensionality_reduction.run_pca.run_pca`.
        options (BuildNeighborIndexOptions): Optional parameters.

    Returns:
        NeighborIndex: Nearest neighbor search index.
    """
    if not input.flags.c_contiguous:
        raise ValueError("expected 'input' to have row-major layout")

    ptr = lib.build_neighbor_index(
        input.shape[1], input.shape[0], input, options.approximate
    )
    return NeighborIndex(ptr)
