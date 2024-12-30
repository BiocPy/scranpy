from typing import Sequence, Optional
from dataclasses import dataclass

import knncolle
import numpy

from . import lib_scranpy as lib


@dataclass
class ScaleByNeighborsResults:
    """Results of :py:func:`~scale_by_neighbors`."""

    scaling: numpy.ndarray
    """Scaling factor to be aplied to each embedding in ``x``."""

    combined: numpy.ndarray
    """Matrix of scaled embeddings. Formed by scaling each entry of ``x`` by
    its corresponding entry of ``scaling``, and then concatenating them
    together by row."""


def scale_by_neighbors(
    x: Sequence,
    num_neighbors: int = 20,
    num_threads: int = 1,
    weights: Optional[Sequence] = None,
    nn_parameters: knncolle.Parameters = knncolle.AnnoyParameters()
) -> ScaleByNeighborsResults:
    """Scale multiple embeddings (usually derived from different modalities
    across the same set of cells) so that their within-population variances are
    comparable, and then combine them into a single embedding matrix for
    combined downstream analysis.

    Args:
        x: 
            Sequence of of numeric matrices of principal components or other
            embeddings, one for each modality. For each entry, rows are
            dimensions and columns are cells. All entries should have the same
            number of columns but may have different numbers of rows.

        num_neighbors:
            Number of neighbors to use to define the scaling factor.

        num_threads
            Number of threads to use.

        nn_parameters:
            Algorithm for the nearest-neighbor search.

        weights:
            Array of length equal to ``x``, specifying the weights to apply to
            each modality.  Each value represents a multiplier of the
            within-population variance of its modality, i.e., larger values
            increase the contribution of that modality in the combined output
            matrix. The default of ``None`` is equivalent to an all-1 vector,
            i.e., all modalities are scaled to have the same within-population
            variance.

    Returns:
        Scaling factors and the combined matrix from all modalities.
    """
    nmod = len(x)

    ncols = None
    for i, m in enumerate(x):
        if ncols is None:
            ncols = m.shape[1]
        elif ncols != m.shape[1]:
            raise ValueError("all entries of 'x' should have the same number of columns")

    distances = []
    for i, m in enumerate(x):
        idx = knncolle.build_index(nn_parameters, m)
        distances.append(knncolle.find_distance(idx, num_neighbors=num_neighbors, num_threads=num_threads))

    scaling = lib.scale_by_neighbors(distances)
    if not weights is None:
        if len(weights) != len(x):
            raise ValueError("'weights' should have the same length as 'x'");
        for i, w in enumerate(weights):
            scaling[i] *= w

    copies = []
    for i, m in enumerate(x):
        copies.append(m * scaling[i])

    return ScaleByNeighborsResults(scaling, numpy.concatenate(copies, axis=0))
