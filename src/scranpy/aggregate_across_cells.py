from typing import Any, Sequence
from dataclasses import dataclass

import numpy
import mattress
import biocutils

from . import lib_scranpy as lib
from .combine_factors import combine_factors


@dataclass
class AggregateAcrossCellsResults:
    """Results of :py:func:`~aggregate_across_cells`."""

    sum: numpy.ndarray
    """Floating-point matrix where each row corresponds to a gene and each column corresponds to a unique combination of grouping levels.
    Each matrix entry contains the summed expression across all cells with that combination."""

    detected: numpy.ndarray
    """Integer matrix where each row corresponds to a gene and each column corresponds to a unique combination of grouping levels.
    Each entry contains the number of cells with detected expression in that combination."""

    combinations: biocutils.NamedList
    """Sorted and unique combination of levels across all ``factors=`` in :py:func:`~aggregate_across_cells`.
    Each entry of the list is another list that corresponds to an entry of ``factors=``, where the ``i``-th combination is defined as the ``i``-th elements of all inner lists.
    Combinations are in the same order as the columns of :py:attr:`~sum` and :py:attr:`~detected`.""" 

    counts: numpy.ndarray
    """Number of cells associated with each combination.
    Each entry corresponds to a combination in :py:attr:`~combinations`."""

    index: numpy.ndarray
    """Integer vector of length equal to the number of cells.
    This specifies the combination in :py:attr:`~combinations` associated with each cell."""

    def to_summarizedexperiment(self, include_counts: bool = True):
        """Convert the results to a :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.

        Args:
            include_counts:
                Whether to include :py:attr:`~counts` in the column data.
                Users may need to set this to ``False`` if a ``"counts"`` factor is present in :py:attr:`~combinations`.

        Returns:
            A :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`
            where :py:attr:`~sum` and :py:attr:`~detected` are assays and :py:attr:`~combinations` is stored in the column data.
        """
        facnames = self.combinations.get_names()
        if facnames is None:
            facnames = [str(i) for i in range(len(self.combinations))]
        else:
            facnames = facnames.as_list()

        import biocframe
        combos = {}
        for i, f in enumerate(facnames):
            combos[f] = self.combinations[i]
        cd = biocframe.BiocFrame(combos, column_names=facnames)
        if include_counts:
            if cd.has_column("counts"):
                raise ValueError("conflicting 'counts' columns, consider setting 'include_counts = False'")
            cd.set_column("counts", self.counts, in_place=True)

        import summarizedexperiment
        return summarizedexperiment.SummarizedExperiment(
            { "sum": self.sum, "detected": self.detected },
            column_data=cd
        )


def aggregate_across_cells(
    x: Any,
    factors: Sequence,
    num_threads: int = 1
) -> AggregateAcrossCellsResults:
    """Aggregate expression values across cells based on one or more grouping factors.
    This is primarily used to create pseudo-bulk profiles for each cluster/sample combination.

    Args:
        x: 
            A matrix-like object where rows correspond to genes or genomic features and columns correspond to cells.
            Values are expected to be counts.

        factors:
            One or more grouping factors, see :py:func:`~scranpy.combine_factors.combine_factors`.
            If this is a :py:class:`~biocutils.NamedList.NamedList`, any names will be retained in the output.

        num_threads:
            Number of threads to use for aggregation.

    Returns:
        Results of the aggregation, including the sum and the number of detected cells in each group for each gene.
    """
    comblev, combind = combine_factors(factors)
    if isinstance(factors, biocutils.NamedList):
        facnames = factors.get_names()
    else:
        facnames = None
    comblev = biocutils.NamedList(comblev, facnames)

    mat = mattress.initialize(x)
    outsum, outdet = lib.aggregate_across_cells(mat.ptr, combind, num_threads)

    counts = numpy.zeros(len(comblev[0]), dtype=numpy.uint32)
    for i in combind:
        counts[i] += 1

    return AggregateAcrossCellsResults(outsum, outdet, comblev, counts, combind)
