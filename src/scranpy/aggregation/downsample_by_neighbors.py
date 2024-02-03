from numpy import ndarray, array, int32, zeros
from typing import Union, Tuple

from .. import _cpphelpers as lib
from ..nearest_neighbors import (
    FindNearestNeighborsOptions,
    NeighborIndex,
    NeighborResults,
    build_neighbor_index,
    find_nearest_neighbors,
)


class DownsampleByNeighborsOptions:
    """Options to pass to `~scranpy.aggregation.downsample_by_neighbors.downsample_by_neighbors`.

    Attributes:
        num_threads: Number of threads to use.
    """

    num_threads: int = 1


def downsample_by_neighbors(
    input: Union[NeighborIndex, NeighborResults, ndarray],
    k: int,
    options: DownsampleByNeighborsOptions = DownsampleByNeighborsOptions(),
) -> Tuple[ndarray, ndarray]:
    """Downsample a dataset by its neighbors. We do by considering a cell to be a "representative" of its nearest
    neighbors, allowing us to downsample by removing all of its neighbors; this is repeated until all cells are assigned
    to a representative, starting from the cells in the densest part of the dataset and working our way down. This
    approach aims to preserve the relative density of points for a faithful downsampling while guaranteeing the
    representation of rare subpopulations.

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

        k: Number of neighbors to use for downsampling.
            Larger values result in more downsampling at the cost of speed.
            Only relevant if ``input`` is not a ``NeighborResults`` object.

        options:
            Further options.

    Returns:
        The first value is of length less than the number of observations, and
        contains the indices of the observations that were retained after
        downsampling. The second value is of length equal to the number of
        observations, and contains the index of the representative observation
        for each observation in the dataset.
    """
    if not isinstance(input, NeighborResults):
        if not isinstance(input, NeighborIndex):
            input = build_neighbor_index(input)
        input = find_nearest_neighbors(
            input,
            k=k,
            options=FindNearestNeighborsOptions(num_threads=options.num_threads),
        )

    output = zeros(input.num_cells(), dtype=int32)
    lib.downsample_by_neighbors(
        input.ptr,
        output,
        options.num_threads,
    )

    keep = []
    for i, x in enumerate(output):
        if i == x:
            keep.append(i)

    return array(keep, dtype=int32), output
