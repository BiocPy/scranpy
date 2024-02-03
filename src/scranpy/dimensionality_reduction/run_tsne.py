import ctypes as ct
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Union

from numpy import copy, float64, ndarray, zeros

from .. import _cpphelpers as lib
from ..nearest_neighbors import (
    FindNearestNeighborsOptions,
    NeighborIndex,
    NeighborResults,
    build_neighbor_index,
    find_nearest_neighbors,
)

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"

TsneEmbedding = namedtuple("TsneEmbedding", ["x", "y"])
TsneEmbedding.__doc__ = """Named tuple of t-SNE coordinates.

x: 
    A NumPy view of length equal to the number of cells,
    containing the coordinate on the first dimension for each cell.

y: 
    A NumPy view of length equal to the number of cells,
    containing the coordinate on the second dimension for each cell.
"""


class TsneStatus:
    """Status of a t-SNE run.

    This should not be constructed manually but should be returned by
    :py:meth:`~scranpy.dimensionality_reduction.run_tsne.initialize_tsne`.
    """

    def __init__(self, ptr: ct.c_void_p, coordinates: ndarray):
        self.__ptr = ptr
        self.coordinates = coordinates

    def __del__(self):
        lib.free_tsne_status(self.__ptr)

    def num_cells(self) -> int:
        """Get the number of cells in the dataset.

        Returns:
            Number of cells.
        """
        return lib.fetch_tsne_status_nobs(self.__ptr)

    def iteration(self) -> int:
        """Get the current iteration number.

        Returns:
            The current iteration number.
        """
        return lib.fetch_tsne_status_iteration(self.__ptr)

    def clone(self) -> "TsneStatus":
        """Create a deep copy of the current state.

        Returns:
            Copy of the current state.
        """
        cloned = copy(self.coordinates)
        return TsneStatus(lib.clone_tsne_status(self.__ptr), cloned)

    def __deepcopy__(self, memo):
        return self.clone()

    def run(self, iteration: int):
        """Run the t-SNE algorithm up to the specified number of iterations.

        Args:
            iteration:
                Number of iterations to run to.
                This should be greater than the current iteration number
                in :func:`~scranpy.dimensionality_reduction.run_tsne.TsneStatus.iteration`.
        """
        lib.run_tsne(self.__ptr, iteration, self.coordinates)

    def extract(self) -> TsneEmbedding:
        """Extract the t-SNE coordinates for each cell at the current iteration.

        Returns:
            'x' and 'y' t-SNE coordinates for all cells.
        """
        return TsneEmbedding(self.coordinates[:, 0], self.coordinates[:, 1])


def tsne_perplexity_to_neighbors(perplexity: float) -> int:
    """Convert the t-SNE perplexity to the required number of neighbors. This is typically used to perform a separate
    call to :py:meth:`~scranpy.nearest_neighbors.find_nearest_neighbors.find_nearest_neighbors` before passing the
    nearest neighbor results to t-SNE functions.

    Args:
        perplexity:
            Perplexity to use in the t-SNE algorithm.

    Returns:
        Number of neighbors to search for.
    """
    return lib.perplexity_to_k(perplexity)


@dataclass
class InitializeTsneOptions:
    """Optional arguments for :py:meth:`~scranpy.dimensionality_reduction.run_tsne.initialize_tsne`.

    Attributes:
        perplexity:
            Perplexity to use when computing neighbor probabilities.
            Larger values cause the embedding to focus more on broad structure instead of local structure.
            Defaults to 30.

        num_threads:
            Number of threads to use for the
            neighbor search and t-SNE iterations. Defaults to 1.

        seed:
            Seed to use for random initialization of
            the t-SNE coordinates. Defaults to 42.
    """

    perplexity: int = 30
    seed: int = 42
    num_threads: int = 1

    def set_threads(self, num_threads: int):
        self.num_threads = num_threads


def initialize_tsne(
    input: Union[NeighborIndex, NeighborResults, ndarray],
    options: InitializeTsneOptions = InitializeTsneOptions(),
) -> TsneStatus:
    """Initialize the t-SNE algorithm. This is useful for fine-tuned control over the progress of the algorithm, e.g.,
    to pause/resume the optimization of the coordinates.

    Args:
        input:
            Object containing per-cell nearest neighbor results or data that can be used to derive them.

            This may be a a 2-dimensional :py:class:`~numpy.ndarray` containing per-cell
            coordinates, where rows are cells and columns are dimensions.
            This is most typically the result of
            :py:meth:`~scranpy.dimensionality_reduction.run_pca.run_pca`.

            Alternatively, ``input`` may be a pre-built neighbor search index
            (:py:class:`~scranpy.nearest_neighbors.build_neighbor_index.NeighborIndex`)
            for the dataset, typically constructed from the PC coordinates for all cells.

            Alternatively, ``input`` may be pre-computed neighbor search results
            (:py:class:`~scranpy.nearest_neighbors.find_nearest_neighbors.NeighborResults`).
            for all cells in the dataset.
            The number of neighbors should be consistent with the perplexity provided
            in :py:class:`~scranpy.dimensionality_reduction.run_tsne.InitializeTsneOptions`
            (see also :py:meth:`~scranpy.dimensionality_reduction.run_tsne.tsne_perplexity_to_neighbors`).

        options:
            Optional parameters.

    Raises:
        TypeError:
            If ``input`` is not an expected type.

    Returns:
        A t-SNE status object for further iterations.
    """
    if not isinstance(input, NeighborResults):
        k = tsne_perplexity_to_neighbors(options.perplexity)
        if not isinstance(input, NeighborIndex):
            input = build_neighbor_index(input)
        input = find_nearest_neighbors(
            input,
            k=k,
            options=FindNearestNeighborsOptions(num_threads=options.num_threads),
        )

    ptr = lib.initialize_tsne(input.ptr, options.perplexity, options.num_threads)
    coords = zeros((input.num_cells(), 2), dtype=float64, order="C")
    lib.randomize_tsne_start(coords.shape[0], coords, options.seed)

    return TsneStatus(ptr, coords)


@dataclass
class RunTsneOptions:
    """Optional arguments for :py:meth:`~scranpy.dimensionality_reduction.run_tsne.run_tsne`.

    Attributes:
        max_iterations:
            Maximum number of iterations.
            Larger numbers improve convergence at the cost of compute time.
            Defaults to 500.

        initialize_tsne:
            Optional arguments for
            :py:meth:`~scranpy.dimensionality_reduction.run_tsne.initialize_tsne`.
    """

    max_iterations: int = 500
    initialize_tsne: InitializeTsneOptions = field(
        default_factory=InitializeTsneOptions
    )

    def set_threads(self, num_threads: int):
        self.initialize_tsne.set_threads(num_threads)


def run_tsne(
    input: Union[NeighborResults, NeighborIndex, ndarray],
    options: RunTsneOptions = RunTsneOptions(),
) -> TsneEmbedding:
    """Compute a two-dimensional t-SNE embedding for the cells. Neighboring cells in high-dimensional space are placed
    next to each other on the embedding for intuitive visualization. This function is a wrapper around
    :py:meth:`~scranpy.dimensionality_reduction.run_tsne.initialize_tsne` with invocations of the
    :py:meth:`~scranpy.dimensionality_reduction.run_tsne.TsneStatus.run` method to the specified number of iterations.

    Args:
        input:
            Object containing per-cell nearest neighbor results or data that can be used to derive them.

            This may be a a 2-dimensional :py:class:`~numpy.ndarray` containing per-cell
            coordinates, where rows are cells and columns are features/dimensions.
            This is most typically the result of
            :py:meth:`~scranpy.dimensionality_reduction.run_pca.run_pca`.

            Alternatively, ``input`` may be a pre-built neighbor search index
            (:py:class:`~scranpy.nearest_neighbors.build_neighbor_index.NeighborIndex`)
            for the dataset, typically constructed from the PC coordinates for all cells.

            Alternatively, ``input`` may be pre-computed neighbor search results
            (:py:class:`~scranpy.nearest_neighbors.find_nearest_neighbors.NeighborResults`).
            for all cells in the dataset.
            The number of neighbors should be consistent with the perplexity provided
            in :py:class:`~scranpy.dimensionality_reduction.run_tsne.InitializeTsneOptions`
            (see also :py:meth:`~scranpy.dimensionality_reduction.run_tsne.tsne_perplexity_to_neighbors`).

        options:
            Optional parameters.

    Returns:
        Result containing first two dimensions.
    """
    status = initialize_tsne(input, options=options.initialize_tsne)
    status.run(options.max_iterations)
    output = status.extract()
    x = copy(output.x)  # realize NumPy slice views into concrete arrays.
    y = copy(output.y)
    return TsneEmbedding(x, y)
