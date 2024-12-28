from typing import Any, Sequence, Tuple
from dataclasses import dataclass

import numpy
import mattress

from . import lib_scranpy as lib
from .combine_factors import combine_factors


@dataclass
class AggregateAcrossCellsResults:
    """Results of :py:func:`~aggregate_across_cells`."""

    sum: numpy.ndarray
    """Floating-point matrix where each row corresponds to a gene and each
    column corresponds to a unique combination of grouping levels. Each entry
    contains the summed expression across all cells with that combination."""

    detected: numpy.ndarray
    """Integer matrix where each row corresponds to a gene and each column
    corresponds to a unique combination of grouping levels.  Each entry
    contains the number of cells with detected expression in that
    combination."""

    combinations: Tuple
    """Sorted and unique combination of levels. Each entry of the tuple is a
    list that corresponds to a factor. Corresponding elements of each list
    define a single combination. Combinations are in the same order as the
    columns of :py:attr:`~sum` and :py:attr:`~detected`.""" 

    counts: numpy.ndarray
    """Number of cells associated with each combination. Each entry corresponds
    to a combination in :py:attr:`~combinations`."""

    index: numpy.ndarray
    """Integer vector of length equal to the number of cells. This specifies
    the combination in :py:attr:`~combinations` for each cell."""



def aggregate_across_cells(
    x: Any,
    factors: Sequence,
    num_threads: int = 1
) -> AggregateAcrossCellsResults:
    """Aggregate expression values across cells based on one or more grouping
    factors. This is primarily used to create pseudo-bulk profiles for each
    cluster/sample combination.

    Args:
        x: 
            A matrix-like object where rows correspond to genes or genomic
            features and columns correspond to cells. Values are typically
            expected to be counts.

        factors:
            One or more grouping factors, see
            :py:func:`~scranpy.combine_factors.combine_factors`.

        num_threads:
            Number of threads to use for aggregation.

    Returns:
        Results of the aggregation, including the sum and the number of
        detected cells in each group for each gene.
    """
    comblev, combind = combine_factors(factors)

    mat = mattress.initialize(x)
    outsum, outdet = lib.aggregate_across_cells(mat.ptr, combind, num_threads)

    counts = numpy.zeros(len(comblev[0]), dtype=numpy.uint32)
    for i in combind:
        counts[i] += 1

    return AggregateAcrossCellsResults(outsum, outdet, comblev, counts, combind)
