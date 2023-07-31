import ctypes as ct

import numpy as np

from ..cpphelpers import lib

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


class NeighborIndex:
    """Class for nearest neighbor search index.

    Args:
        ptr (ct.c_void_p): Pointer reference to scran's nearest neighbor index.
    """

    def __init__(self, ptr: ct.c_void_p):
        """Initialize the class."""
        self.__ptr = ptr

    def __del__(self):
        """Free the reference."""
        lib.free_neighbor_index(self.__ptr)

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
            int: Number of cells.
        """
        return lib.fetch_neighbor_index_nobs(self.__ptr)

    def num_dimensions(self) -> int:
        """Get number of dimensions.

        Returns:
            int: Number of dimensions.
        """
        return lib.fetch_neighbor_index_ndim(self.__ptr)


def build_neighbor_index(x: np.ndarray, approximate: bool = True) -> NeighborIndex:
    """Build the nearest neighbor search index.

    `x` represents coordinates fo each cell, usually the prinicpal components from the
    PCA step. rows are variables, columns are cells.

    Args:
        x (np.ndarray): Coordinates for each cell in the dataset.
        approximate (bool, optional): Whether to build an index for an approximate
            neighbor search. Defaults to True.

    Returns:
        NeighborIndex: Nearest neighbor search index.
    """
    ptr = lib.build_neighbor_index(x.shape[0], x.shape[1], x.ctypes.data, approximate)
    return NeighborIndex(ptr)
