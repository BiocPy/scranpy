import ctypes as ct
from dataclasses import dataclass

import numpy as np

from .. import cpphelpers as lib
from .._logging import logger

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
    """Optional arguments to build the nearest neighbor index in 
    :py:meth:`~scranpy.nearest_neighbors.build_neighbor_index.build_neighbor_index`.

    Attributes:
        approximate (bool, optional): Whether to build an index for an approximate
            neighbor search. This sacrifices some accuracy for speed.
            Defaults to True.
        verbose (bool, optional): Whether to print logs. Defaults to False.
    """

    approximate: bool = True
    verbose: bool = False


def build_neighbor_index(
    input: np.ndarray, options: BuildNeighborIndexOptions = BuildNeighborIndexOptions()
) -> NeighborIndex:
    """Build a search index for finding nearest neighbors between cells.

    Args:
        input (np.ndarray): A matrix where rows are dimensions and cells are columns.
            This is usually the principal components from the PCA step
            (:py:meth:`~scranpy.dimensionality_reduction.run_pca.run_pca`).
        options (BuildNeighborIndexOptions): Optional parameters.

    Returns:
        NeighborIndex: Nearest neighbor search index.
    """
    if options.verbose is True:
        logger.info("Building nearest neighbor index...")

    ptr = lib.build_neighbor_index(
        input.shape[0], input.shape[1], input, options.approximate
    )
    return NeighborIndex(ptr)
