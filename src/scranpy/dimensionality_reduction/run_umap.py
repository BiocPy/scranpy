import copy
import ctypes as ct
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Optional

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
        return UmapStatus(lib.clone_umap_status(self.__ptr, cloned), cloned)

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


@dataclass
class InitializeUmapOptions:
    """Arguments to initialize UMAP algorithm.

    Arguments:
        min_dist (float, optional): Minimum distance between points. Defaults to 0.1.
        num_neighbors (int, optional): Number of neighbors to use in the UMAP algorithm.
            Ignored if ``input`` is a
            :py:class:`~scranpy.nearest_neighbors.find_nearest_neighbors.NeighborResults`
            object. Defaults to 15.
        num_epochs (int, optional): Number of epochs to run. Defaults to 500.
        num_threads (int, optional): Number of threads to use. Defaults to 1.
        seed (int, optional): Seed to use for RNG. Defaults to 42.
        verbose (bool): Display logs? Defaults to False.
    """

    min_dist: float = 0.1
    num_neighbors: int = 15
    num_epochs: int = 500
    seed: int = 42
    num_threads: int = 1
    verbose: bool = False


def initialize_umap(
    input: NeighborIndexOrResults,
    options: InitializeUmapOptions = InitializeUmapOptions(),
) -> UmapStatus:
    """Initialize the UMAP step.

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
        options (InitializeUmapOptions): Optional parameters.

    Raises:
        TypeError: If ``input`` is not an expected type.

    Returns:
        UmapStatus: a umap status object.
    """
    if not is_neighbor_class(input):
        raise TypeError(
            "`input` must be either the nearest neighbor search index, "
            "search results or a matrix."
        )

    if not isinstance(input, NeighborResults):
        if not isinstance(input, NeighborIndex):
            if options.verbose is True:
                logger.info("`input` is a matrix, building nearest neighbor index...")

            input = build_neighbor_index(input)

        if options.verbose is True:
            logger.info("Finding the nearest neighbors...")

        input = find_nearest_neighbors(
            input,
            FindNearestNeighborsOptions(
                k=options.num_neighbors, num_threads=options.num_threads
            ),
        )

    coords = np.ndarray((input.num_cells(), 2), dtype=np.float64, order="C")
    ptr = lib.initialize_umap(
        input.ptr, options.num_epochs, options.min_dist, coords, options.num_threads
    )

    return UmapStatus(ptr, coords)


@dataclass
class RunUmapOptions:
    """Arguments to compute UMAP embeddings -
    :py:meth:`~scranpy.dimensionality_reduction.run_umap.run_umap`.

    Attributes:
        initialize_umap (InitializeUmapOptions): Arguments to initialize UMAP -
            :py:meth:`~scranpy.dimensionality_reduction.run_umap.initialize_umap`
            function.
        verbose (bool): Display logs? Defaults to False.
    """

    initialize_umap: InitializeUmapOptions = field(
        default_factory=InitializeUmapOptions
    )
    verbose: bool = False


def run_umap(
    input: NeighborIndexOrResults, options: RunUmapOptions = RunUmapOptions()
) -> UmapEmbedding:
    """Compute UMAP embedding.

    Args:
        input (NeighborIndexOrResults): Input matrix, pre-computed neighbor index
            or neighbors.
        options (RunUmapOptions): Optional parameters.

    Returns:
        UmapEmbedding: Result containing the first two dimensions.
    """
    if options.verbose is True:
        logger.info("Initializing UMAP...")

    status = initialize_umap(input, options=options.initialize_umap)

    if options.verbose is True:
        logger.info("Running the UMAP...")

    status.run()

    if options.verbose is True:
        logger.info("Done computing UMAP embeddings...")

    output = status.extract()
    x = copy.deepcopy(output.x)  # is this really necessary?
    y = copy.deepcopy(output.y)

    return UmapEmbedding(x, y)
