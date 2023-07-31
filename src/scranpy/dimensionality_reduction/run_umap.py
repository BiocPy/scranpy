import copy
import ctypes as ct
from collections import namedtuple
from typing import Optional

import numpy as np

from ..cpphelpers import lib
from ..nearest_neighbors import (
    NeighborIndex,
    NeighborResults,
    build_neighbor_index,
    find_nearest_neighbors,
)
from ..types import NeighborIndexOrResults, is_neighbor_class

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"

UmapEmbedding = namedtuple("UmapEmbedding", ["x", "y"])
UmapEmbedding.__doc__ = """Named tuple of coordinates from UMAP step.

x (np.ndarray): a numpy view of the first dimension.
y (np.ndarray): a numpy view of the second dimension.
"""


class UmapStatus:
    """Class to manage UMAP runs.

    Args:
        ptr (ct.c_void_p): Pointer that holds the result from
            scran's `initialize_umap` method.
        coordinates (np.ndarray): Object to hold the embeddings.
    """

    def __init__(self, ptr: ct.c_void_p, coordinates: np.ndarray):
        """Initialize the class."""
        self.__ptr = ptr
        self.coordinates = coordinates

    def __del__(self):
        """Free the reference."""
        lib.free_umap_status(self.__ptr)

    @property
    def ptr(self) -> ct.c_void_p:
        """Get pointer to scran's umap step.

        Returns:
            ct.c_void_p: Pointer reference.
        """
        return self.__ptr

    def num_cells(self) -> int:
        """Get number of cells.

        Returns:
            int: Number of cells.
        """
        return lib.fetch_umap_status_nobs(self.__ptr)

    def epoch(self) -> int:
        """Get current epoch from the state.

        Returns:
            int: epoch.
        """
        return lib.fetch_umap_status_epoch(self.__ptr)

    def num_epochs(self) -> int:
        """Get number of epochs.

        Returns:
            int: Number of epochs.
        """
        return lib.fetch_umap_status_num_epochs(self.__ptr)

    def clone(self) -> "UmapStatus":
        """deepcopy the current state.

        Returns:
            UmapStatus: Copy of the current state.
        """
        cloned = copy.deepcopy(self.coordinates)
        return UmapStatus(lib.clone_umap_status(self.__ptr, cloned.ctypes.data), cloned)

    def __deepcopy__(self, memo):
        """Same as clone."""
        return self.clone()

    def run(self, epoch_limit: Optional[int] = None):
        """Run the UMAP algorithm specified epoch limit.

        Args:
            epoch_limit (int): Number of epochs to run.
        """
        if epoch_limit is None:
            epoch_limit = 0
            # i.e., until the end.
        lib.run_umap(self.__ptr, epoch_limit)

    def extract(self) -> UmapEmbedding:
        """Access the first two dimensions.

        Returns:
            UmapEmbedding: Object with x and y coordinates.
        """
        return UmapEmbedding(self.coordinates[:, 0], self.coordinates[:, 1])


def initialize_umap(
    input: NeighborIndexOrResults,
    min_dist: float = 0.1,
    num_neighbors: int = 15,
    num_epochs: int = 500,
    num_threads: int = 1,
    seed: int = 42,
) -> UmapStatus:
    """Initialize the UMAP step.

    `input` is either a pre-built neighbor search index for the dataset, or a
    pre-computed set of neighbor search results for all cells. If `input` is a matrix,
    we compute the nearest neighbors for each cell, assuming it represents the
    coordinates for each cell, usually the result of PCA step.
    rows are variables, columns are cells.

    Args:
        input (NeighborIndexOrResults): Input matrix or pre-computed neighbors.
        min_dist (float, optional): Minimum distance between points. Defaults to 0.1.
        num_neighbors (int, optional): Number of neighbors to use in the UMAP algorithm.
            Ignored if `input` is a `NeighborResults` object. Defaults to 15.
        num_epochs (int, optional): Number of epochs to run. Defaults to 500.
        num_threads (int, optional): Number of threads to use. Defaults to 1.
        seed (int, optional): Seed to use for RNG. Defaults to 42.

    Raises:
        TypeError: If input does not match expectations.

    Returns:
        UmapStatus: a umap status object.
    """
    if not is_neighbor_class(input):
        raise TypeError(
            "`x` must be either the nearest neighbor search index, search results or "
            "a matrix."
        )

    if not isinstance(input, NeighborResults):
        if not isinstance(input, NeighborIndex):
            input = build_neighbor_index(input)
        input = find_nearest_neighbors(input, k=num_neighbors, num_threads=num_threads)

    coords = np.ndarray((input.num_cells(), 2), dtype=np.float64, order="C")
    ptr = lib.initialize_umap(
        input.ptr, num_epochs, min_dist, coords.ctypes.data, num_threads
    )

    return UmapStatus(ptr, coords)


def run_umap(**kwargs) -> UmapEmbedding:
    """Compute UMAP embedding.

    Args:
        **kwargs: Arguments specified by `initialize_umap` function.

    Returns:
        UmapEmbedding: Result containing the first two dimensions.
    """
    status = initialize_umap(**kwargs)
    status.run()

    output = status.extract()
    x = copy.deepcopy(output.x)  # is this really necessary?
    y = copy.deepcopy(output.y)

    return UmapEmbedding(x, y)
