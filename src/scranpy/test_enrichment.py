from typing import Sequence, Union

import numpy

from . import lib_scranpy as lib


def _count_overlaps(x: Sequence, sets: Sequence) -> numpy.ndarray:
    overlap = numpy.ndarray((len(sets),), dtype=numpy.uint32)
    xset = set(x)
    for s, curset in enumerate(sets):
        counter = 0
        for g in curset:
            counter += (g in xset)
        overlap[s] = counter
    return overlap


def test_enrichment(
    x: Sequence,
    sets: Sequence,
    universe: Union[int, Sequence],
    log: bool = False,
    num_threads: int = 1
) -> numpy.ndarray:
    """Perform a hypergeometric test for enrichment of interesting genes (e.g.,
    markers) in one or more pre-defined gene sets.

    Args:
        x: 
            Sequence of identifiers for the interesting genes.

        sets:
            Sequence of gene sets, where each entry corresponds to a gene set
            and contains a sequence of identifiers for genes in that set.

        universe:
            Sequence of identifiers for the universe of genes in the dataset.
            It is expected that ``x`` is a subset of ``universe``. Alternatively,
            an integer specifying the number of genes in the universe.

        log:
            Whether to report the log-transformed p-values.

        num_threads: 
            Number of threads to use.

    Returns:
        Array of (log-transformed) p-values to test for significant enrichment
        of ``x`` in each entry of ``sets``.
    """
    overlap = _count_overlaps(x, sets)

    if isinstance(universe, int):
        set_sizes = numpy.ndarray((len(sets),), dtype=numpy.uint32)
        for s, curset in enumerate(sets):
            set_sizes[s] = len(curset)
    else:
        set_sizes = _count_overlaps(universe, sets)
        universe = len(universe)

    return lib.test_enrichment(
        overlap,
        len(x),
        set_sizes,
        universe,
        log,
        num_threads
    )
