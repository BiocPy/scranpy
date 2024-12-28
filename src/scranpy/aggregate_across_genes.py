from typing import Any, Sequence, List

import numpy
import mattress

from . import lib_scranpy as lib


def aggregate_across_genes(
    x: Any,
    sets: Sequence,
    average: bool = False,
    num_threads: int = 1
) -> List:
    """Aggregate expression values across genes, potentially with weights.
    This is typically used to summarize expression values for gene sets into a
    single per-cell score.

    Args:
        x:
            Matrix-like object where rows correspond to genes or genomic
            features and columns correspond to cells.  Values are typically
            expected to be counts.

        sets:
            Sequence of integer arrays containing the row indices of genes in
            each set. Alternatively, each entry may be a tuple of length 2,
            containing an integer vector (row indices) and a numeric vector
            (weights).

        average:
            Whether to compute the average rather than the sum.

        num_threads: 
            Number of threads to be used for aggregation.

    Returns:
        List of length equal to that of ``sets``. Each entry is a numeric
        vector of length equal to the number of columns in ``x``, containing
        the (weighted) sum/mean of expression values for the corresponding set
        across all cells.
    """ 
    new_sets = [] 
    for s in sets:
        if isinstance(s, tuple):
            new_sets.append((
                numpy.array(s[0], copy=None, order="A", dtype=numpy.uint32),
                numpy.array(s[1], copy=None, order="A", dtype=numpy.float64)
            ))
        else:
            new_sets.append(numpy.array(s, copy=None, order="A", dtype=numpy.uint32))

    mat = mattress.initialize(x)
    output = lib.aggregate_across_genes(
        mat.ptr,
        new_sets,
        average,
        num_threads
    )
    return output
