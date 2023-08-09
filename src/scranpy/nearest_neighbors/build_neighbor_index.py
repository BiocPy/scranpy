import ctypes as ct
from dataclasses import dataclass

import numpy as np

from .. import cpphelpers as lib
from .._logging import logger

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


@dataclass
class BuildNeighborIndexArgs:
    """Arguments to build nearest neighbor index -
    :py:meth:`~scranpy.nearest_neighbors.build_neighbor_index.build_neighbor_index`.

    Attributes:
        approximate (bool, optional): Whether to build an index for an approximate
            neighbor search. Defaults to True.
        verbose (bool, optional): Display logs?. Defaults to False.
    """

    approximate: bool = True
    verbose: bool = False


def build_neighbor_index(
    input: np.ndarray, options: BuildNeighborIndexArgs = BuildNeighborIndexArgs()
) -> NeighborIndex:
    """Build the nearest neighbor search index.

    Note: rows are features, columns are cells.

    Args:
        input (np.ndarray): Co-ordinates for each cell in the dataset
            usually the prinicpal components from the PCA step
            (:py:meth:`~scranpy.dimensionality_reduction.run_pca.run_pca`).
        options (BuildNeighborIndexArgs): Optional parameters.

    Returns:
        NeighborIndex: Nearest neighbor search index.
    """
    if options.verbose is True:
        logger.info("Building nearest neighbor index...")

    ptr = lib.build_neighbor_index(
        input.shape[0], input.shape[1], input, options.approximate
    )
    return NeighborIndex(ptr)
