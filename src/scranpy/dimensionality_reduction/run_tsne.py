import copy
import ctypes as ct
from collections import namedtuple
from dataclasses import dataclass, field

import numpy as np

from .. import cpphelpers as lib
from .._logging import logger
from ..nearest_neighbors import (
    FindNearestNeighborsOptions,
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
TsneEmbedding.__doc__ = """Named tuple of t-SNE coordinates.

x (np.ndarray): a NumPy view of length equal to the number of cells,
    containing the coordinate on the first dimension for each cell.
y (np.ndarray): a numpy view of length equal to the number of cells,
    containing the coordinate on the second dimension for each cell.
"""


class TsneStatus:
    """Status of a t-SNE run.
    This should not be constructed manually but should be returned by
    :py:meth:`~scranpy.dimensionality_reduction.run_tsne.initialize_tsne`.
    """

    def __init__(self, ptr: ct.c_void_p, coordinates: np.ndarray):
        self.__ptr = ptr
        self.coordinates = coordinates

    def __del__(self):
        lib.free_tsne_status(self.__ptr)

    def num_cells(self) -> int:
        """Get the number of cells.

        Returns:
            int: Number of cells.
        """
        return lib.fetch_tsne_status_nobs(self.__ptr)

    def iteration(self) -> int:
        """Get the current iteration number.

        Returns:
            int: The current iteration number.
        """
        return lib.fetch_tsne_status_iteration(self.__ptr)

    def clone(self) -> "TsneStatus":
        """Create a deep copy of the current state.

        Returns:
            TsneStatus: A new object containing a copy of the current state.
        """
        cloned = copy.deepcopy(self.coordinates)
        return TsneStatus(lib.clone_tsne_status(self.__ptr), cloned)

    def __deepcopy__(self, memo):
        return self.clone()

    def run(self, iteration: int):
        """Run the t-SNE algorithm up to the specified number of iterations.

        Args:
            iteration (int): Number of iterations to run to.
                This should be greater than the current iteration number
                in :func:`~scranpy.dimensionality_reduction.run_tsne.TsneStatus.iteration`.
        """
        lib.run_tsne(self.__ptr, iteration, self.coordinates)

    def extract(self) -> TsneEmbedding:
        """Extract the t-SNE coordinates for each cell at the current iteration.

        Returns:
            TsneEmbedding: x and y coordinates for all cells.
        """
        return TsneEmbedding(self.coordinates[:, 0], self.coordinates[:, 1])


def tsne_perplexity_to_neighbors(perplexity: float) -> int:
    """Convert the t-SNE perplexity to the required number of neighbors.
    This is typically used to perform a separate call to 
    :py:meth:`~scranpy.nearest_neighbors.find_nearest_neighbors.find_nearest_neighbors`
    before passing the nearest neighbor results to t-SNE functions.

    Args:
        perplexity (float): perplexity to use in the t-SNE algorithm.

    Returns:
        int: Number of neighbors to search for.
    """
    return lib.perplexity_to_k(perplexity)


@dataclass
class InitializeTsneOptions:
    """Optional arguments to initialize the t-SNE status in
    :py:meth:`~scranpy.dimensionality_reduction.run_tsne.initialize_tsne`.

    Attributes:
        perplexity (int, optional): Perplexity to use when computing neighbor
            probabilities. Defaults to 30.
        num_threads (int, optional): Number of threads to use for the 
            neighbor search and t-SNE iterations. Defaults to 1.
        seed (int, optional): Seed to use for random initialization of
            the t-SNE coordinates. Defaults to 42.
        verbose (bool): Whether to display logs. Defaults to False.
    """

    perplexity: int = 30
    seed: int = 42
    num_threads: int = 1
    verbose: bool = False


def initialize_tsne(
    input: NeighborIndexOrResults,
    options: InitializeTsneOptions = InitializeTsneOptions(),
) -> TsneStatus:
    """Initialize the t-SNE step.

    ``input`` is either a pre-built neighbor search index for the dataset
    (:py:class:`~scranpy.nearest_neighbors.build_neighbor_index.NeighborIndex`), or a
    pre-computed set of neighbor search results for all cells
    (:py:class:`~scranpy.nearest_neighbors.find_nearest_neighbors.NeighborResults`).
    If ``input`` is a matrix (:py:class:`numpy.ndarray`),
    we compute the nearest neighbors for each cell, assuming it represents the
    coordinates for each cell, usually the result of PCA step
    (:py:meth:`~scranpy.dimensionality_reduction.run_pca.run_pca`).

    Args:
        input (NeighborIndexOrResults): Input matrix, pre-computed neighbor index
            or neighbors.
        options (InitializeTsneOptions): Optional parameters.

    Raises:
        TypeError: If ``input`` is not an expected type.

    Returns:
        TsneStatus: A tsne status object.
    """
    if not is_neighbor_class(input):
        raise TypeError(
            "`input` must be either the nearest neighbor search index, search results "
            "or a matrix."
        )

    if not isinstance(input, NeighborResults):
        k = tsne_perplexity_to_neighbors(options.perplexity)
        if not isinstance(input, NeighborIndex):
            if options.verbose is True:
                logger.info("`input` is a matrix, building nearest neighbor index...")

            input = build_neighbor_index(input)

        if options.verbose is True:
            logger.info("Finding the nearest neighbors...")

        input = find_nearest_neighbors(
            input, FindNearestNeighborsOptions(k=k, num_threads=options.num_threads)
        )

    ptr = lib.initialize_tsne(input.ptr, options.perplexity, options.num_threads)
    coords = np.ndarray((input.num_cells(), 2), dtype=np.float64, order="C")
    lib.randomize_tsne_start(coords.shape[0], coords, options.seed)

    return TsneStatus(ptr, coords)


@dataclass
class RunTsneOptions:
    """Optional arguments to compute t-SNE embeddings in
    :py:meth:`~scranpy.dimensionality_reduction.run_tsne.run_tsne`.

    Attributes:
        max_iterations (int, optional): Maximum number of iterations. Defaults to 500.
        initialize_tsne (InitializeTsneOptions): Optional arguments for initialization
            with :py:meth:`~scranpy.dimensionality_reduction.run_tsne.initialize_tsne`.
        verbose (bool): Whether to print logs. Defaults to False.
    """

    max_iterations: int = 500
    initialize_tsne: InitializeTsneOptions = field(default_factory=InitializeTsneOptions)
    verbose: bool = False


def run_tsne(
    input: NeighborIndexOrResults, options: RunTsneOptions = RunTsneOptions()
) -> TsneEmbedding:
    """Compute t-SNE embedding.

    Args:
        input (NeighborIndexOrResults): Input matrix, pre-computed neighbor index
            or neighbors.
        options (RunTsneOptions): Optional parameters.

    Returns:
        TsneEmbedding: Result containing first two dimensions.
    """
    if options.verbose is True:
        logger.info("Initializing the t-SNE...")

    status = initialize_tsne(input, options=options.initialize_tsne)

    if options.verbose is True:
        logger.info(
            f"Running the t-SNE algorithm for {options.max_iterations} iterations..."
        )
    status.run(options.max_iterations)

    if options.verbose is True:
        logger.info("Done computing t-SNE embeddings...")

    output = status.extract()
    x = copy.deepcopy(output.x)  # is this really necessary?
    y = copy.deepcopy(output.y)

    return TsneEmbedding(x, y)
