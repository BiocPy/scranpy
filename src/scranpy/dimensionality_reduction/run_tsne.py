import copy
import ctypes as ct
from collections import namedtuple

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

TsneEmbedding = namedtuple("TsneEmbedding", ["x", "y"])
TsneEmbedding.__doc__ = """Named tuple of coordinates from t-SNE step.

x (np.ndarray): a numpy view of the first dimension.
y (np.ndarray): a numpy view of the second dimension.
"""


class TsneStatus:
    """Class to manage t-SNE runs.

    Args:
        ptr (ct.c_void_p): pointer that holds the result from
            scran's `initialize_tsne` method.
        coordinates (np.ndarray): object to hold the embeddings.
    """

    def __init__(self, ptr: ct.c_void_p, coordinates: np.ndarray):
        """Initialize the class."""
        self.__ptr = ptr
        self.coordinates = coordinates

    def __del__(self):
        """Free the reference."""
        lib.free_tsne_status(self.__ptr)

    @property
    def ptr(self) -> ct.c_void_p:
        """Get pointer to scran's tsne step.

        Returns:
            ct.c_void_p: pointer reference.
        """
        return self.__ptr

    def num_cells(self) -> int:
        """Get number of cells.

        Returns:
            int: number of cells.
        """
        return lib.fetch_tsne_status_nobs(self.__ptr)

    def iteration(self) -> int:
        """Get current iteration from the state.

        Returns:
            int: iteration.
        """
        return lib.fetch_tsne_status_iteration(self.__ptr)

    def clone(self) -> "TsneStatus":
        """deepcopy of the current state.

        Returns:
            TsneStatus: a copy of the current state.
        """
        cloned = copy.deepcopy(self.coordinates)
        return TsneStatus(lib.clone_tsne_status(self.__ptr), cloned)

    def __deepcopy__(self, memo):
        """Same as clone."""
        return self.clone()

    def run(self, iteration: int):
        """Run the t-SNE algorithm for specified iterations.

        Args:
            iteration (int): number of iteratons to run.
        """
        lib.run_tsne(self.__ptr, iteration, self.coordinates.ctypes.data)

    def extract(self) -> TsneEmbedding:
        """Access the first two dimensions.

        Returns:
            TsneEmbedding: object with x and y coordinates.
        """
        return TsneEmbedding(self.coordinates[:, 0], self.coordinates[:, 1])


def initialize_tsne(
    input: NeighborIndexOrResults,
    perplexity: int = 30,
    num_threads: int = 1,
    seed: int = 42,
) -> TsneStatus:
    """Initialize the t-SNE step.

    `input` is either a pre-built neighbor search index for the dataset, or a
    pre-computed set of neighbor search results for all cells. If `input` is a matrix,
    we compute the nearest neighbors for each cell, assuming it represents the
    coordinates for each cell, usually the result of PCA step.
    rows are variables, columns are cells.

    Args:
        input (NeighborIndexOrResults): Input matrix or pre-computed neighbors.
        perplexity (int, optional): Perplexity to use when computing neighbor
            probabilities. Defaults to 30.
        num_threads (int, optional): Number of threads to use. Defaults to 1.
        seed (int, optional): Seed to use for RNG. Defaults to 42.

    Raises:
        TypeError: if input does not match expectations.

    Returns:
        TsneStatus: a tsne status object.
    """
    if not is_neighbor_class(input):
        raise TypeError(
            "`x` must be either the nearest neighbor search index, search results or "
            "a matrix."
        )

    if not isinstance(input, NeighborResults):
        k = lib.perplexity_to_k(perplexity)
        if not isinstance(input, NeighborIndex):
            input = build_neighbor_index(input)
        input = find_nearest_neighbors(input, k=k, num_threads=num_threads)

    ptr = lib.initialize_tsne(input.ptr, perplexity, num_threads)
    coords = np.ndarray((input.num_cells(), 2), dtype=np.float64, order="C")
    lib.randomize_tsne_start(coords.shape[1], coords.ctypes.data, seed)

    return TsneStatus(ptr, coords)


def run_tsne(max_iterations: int = 500, **kwargs) -> TsneEmbedding:
    """Compute t-SNE embedding.

    Args:
        max_iterations (int, optional): Maximum number of iterations. Defaults to 500.
        **kwargs: Arguments specified by `initialize_tsne` function.


    Returns:
        TsneEmbedding: result containing first two dimensions.
    """
    status = initialize_tsne(**kwargs)
    status.run(max_iterations)

    output = status.extract()
    x = copy.deepcopy(output.x)  # is this really necessary?
    y = copy.deepcopy(output.y)

    return TsneEmbedding(x, y)
