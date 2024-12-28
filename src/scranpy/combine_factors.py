from typing import Sequence, Tuple

import biocutils
import numpy

from . import lib_scranpy as lib


def combine_factors(factors: Sequence, keep_unused: bool = False) -> Tuple:
    """Combine multiple categorical factors based on the unique combinations of
    levels from each factor.

    Args:
        factors:
            Sequence containing factors of interest. Each entry corresponds to
            a factor and should be a sequence of the same length. Corresponding
            elements across all factors represent the combination of levels for
            a single observation.

        keep_unused:
            Whether to report unused combinations of levels.

    Returns:
        Tuple containing:

        - Sorted and unique combinations of levels as a tuple. Each entry of
          the tuple is a list that corresponds to a factor. Corresponding
          elements of each list define a single combination.
        - Integer array of length equal to each sequence of ``factors``,
          specifying the combination for each observation.
    """
    f0 = []
    levels0 = []
    for f, current in enumerate(factors):
        lev, ind = biocutils.factorize(current)
        f0.append(ind)
        levels0.append(lev)

    ind, combos = lib.combine_factors(
        f0,
        keep_unused,
        numpy.array([len(l) for l in levels0], dtype=numpy.uint32)
    )

    new_combinations = [] 
    for f, current in enumerate(combos):
        new_combinations.append(biocutils.subset_sequence(levels0[f], current))

    return (*new_combinations,), ind
