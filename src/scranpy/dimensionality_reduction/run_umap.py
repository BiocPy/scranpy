import ctypes as ct
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Optional, Union

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

UmapEmbedding = namedtuple("UmapEmbedding", ["x", "y"])
UmapEmbedding.__doc__ = """Named tuple of UMAP coordinates.

x: 
    A NumPy view of length equal to the number of cells,
    containing the coordinate on the first dimension for each cell.

y: 
    A NumPy view of length equal to the number of cells,
    containing the coordinate on the second dimension for each cell.
"""


class UmapStatus:
    """Status of a UMAP run.

    This should not be constructed manually but should be returned by
    :py:meth:`~scranpy.dimensionality_reduction.run_tsne.initialize_umap`.
    """

    def __init__(self, ptr: ct.c_void_p, coordinates: ndarray):
        self.__ptr = ptr
        self.coordinates = coordinates

    def __del__(self):
        lib.free_umap_status(self.__ptr)

    def num_cells(self) -> int:
        """Get the number of cells in the dataset.

        Returns:
            Number of cells.
        """
        return lib.fetch_umap_status_nobs(self.__ptr)

    def epoch(self) -> int:
        """Get the current epoch of the UMAP state.

        Returns:
            The current epoch.
        """
        return lib.fetch_umap_status_epoch(self.__ptr)

    def num_epochs(self) -> int:
        """Get the total number of epochs for this UMAP run.

        Returns:
            Number of epochs.
        """
        return lib.fetch_umap_status_num_epochs(self.__ptr)

    def clone(self) -> "UmapStatus":
        """Create a deep copy of the current state.

        Returns:
            Copy of the current state.
        """
        cloned = copy(self.coordinates)
        return UmapStatus(lib.clone_umap_status(self.__ptr, cloned), cloned)

    def __deepcopy__(self, memo):
        return self.clone()

    def run(self, epoch_limit: Optional[int] = None):
        """Run the UMAP algorithm to the specified epoch limit.

        Args:
            epoch_limit:
                Number of epochs to run up to.
                This should be greater than the current epoch
                in :func:`~scranpy.dimensionality_reduction.run_umap.UmapStatus.epoch`.
        """
        if epoch_limit is None:
            epoch_limit = 0  # i.e., until the end.
        lib.run_umap(self.__ptr, epoch_limit)

    def extract(self) -> UmapEmbedding:
        """Extract the UMAP coordinates for each cell at the current epoch.

        Returns:
            `x` and `y` UMAP coordinates for all cells.
        """
        return UmapEmbedding(self.coordinates[:, 0], self.coordinates[:, 1])


@dataclass
class InitializeUmapOptions:
    """Optional arguments for :py:meth:`~scranpy.dimensionality_reduction.run_umap.initialize_umap`.

    Arguments:
        min_dist:
            Minimum distance between points.
            Larger values yield more inflated clumps of cells.
            Defaults to 0.1.

        num_neighbors:
            Number of neighbors to use in the UMAP algorithm.
            Larger values focus more on global structure than local structure.
            Ignored if ``input`` is a
            :py:class:`~scranpy.nearest_neighbors.find_nearest_neighbors.NeighborResults` object. Defaults to 15.

        num_epochs:
            Number of epochs to run.
            Larger values improve convergence at the cost of compute time.
            Defaults to 500.

        num_threads:
            Number of threads to use for neighbor detection and the UMAP initialization.
            Defaults to 1.

        seed:
            Seed to use for random number generation.
            Defaults to 42.
    """

    min_dist: float = 0.1
    num_neighbors: int = 15
    num_epochs: int = 500
    seed: int = 42
    num_threads: int = 1

    def set_threads(self, num_threads: int):
        self.num_threads = num_threads


def initialize_umap(
    input: Union[NeighborResults, NeighborIndex, ndarray],
    options: InitializeUmapOptions = InitializeUmapOptions(),
) -> UmapStatus:
    """Initialize the UMAP algorithm. This is useful for fine-tuned control over the progress of the algorithm, e.g., to
    pause/resume the optimization of the coordinates.

    ``input`` is either a pre-built neighbor search index for the dataset
    (:py:class:`~scranpy.nearest_neighbors.build_neighbor_index.NeighborIndex`), or a
    pre-computed set of neighbor search results for all cells
    (:py:class:`~scranpy.nearest_neighbors.find_nearest_neighbors.NeighborResults`).
    If ``input`` is a matrix (:py:class:`numpy.ndarray`),
    we compute the nearest neighbors for each cell, assuming it represents the
    coordinates for each cell, usually the result of PCA step
    (:py:meth:`~scranpy.dimensionality_reduction.run_pca.run_pca`).

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

        options:
            Optional parameters.

    Raises:
        TypeError:
            If ``input`` is not an expected type.

    Returns:
        A UMAP status object for iteration through the epochs.
    """
    if not isinstance(input, NeighborResults):
        if not isinstance(input, NeighborIndex):
            input = build_neighbor_index(input)
        input = find_nearest_neighbors(
            input,
            k=options.num_neighbors,
            options=FindNearestNeighborsOptions(num_threads=options.num_threads),
        )

    coords = zeros((input.num_cells(), 2), dtype=float64, order="C")
    ptr = lib.initialize_umap(
        input.ptr, options.num_epochs, options.min_dist, coords, options.num_threads
    )

    return UmapStatus(ptr, coords)


@dataclass
class RunUmapOptions:
    """Optional arguments for :py:meth:`~scranpy.dimensionality_reduction.run_umap.run_umap`.

    Attributes:
        initialize_umap:
            Optional arguments for
            :py:meth:`~scranpy.dimensionality_reduction.run_umap.initialize_umap`.
    """

    initialize_umap: InitializeUmapOptions = field(
        default_factory=InitializeUmapOptions
    )

    def set_threads(self, num_threads: int):
        self.initialize_umap.set_threads(num_threads)


def run_umap(
    input: Union[NeighborResults, NeighborIndex, ndarray],
    options: RunUmapOptions = RunUmapOptions(),
) -> UmapEmbedding:
    """Compute a two-dimensional UMAP embedding for the cells. Neighboring cells in high-dimensional space are placed
    next to each other on the embedding for intuitive visualization. This function is a wrapper around
    :py:meth:`~scranpy.dimensionality_reduction.run_umap.initialize_umap` with invocations of the
    :py:meth:`~scranpy.dimensionality_reduction.run_umap.UmapStatus.run` method to the maximum number of epochs.

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

        options:
            Optional parameters.

    Returns:
        Result containing the first two dimensions.
    """
    status = initialize_umap(input, options=options.initialize_umap)
    status.run()
    output = status.extract()
    x = copy(output.x)  # realize NumPy slicing views into standalone arrays.
    y = copy(output.y)
    return UmapEmbedding(x, y)
