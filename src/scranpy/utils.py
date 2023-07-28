from typing import Sequence

import numpy as np

from .types import FactorizedArray

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def factorize(x: Sequence) -> FactorizedArray:
    """Factorize an array.

    Args:
        x (Sequence): an array.

    Returns:
        FactorizedArray: a factorized tuple.
    """

    if not isinstance(x, list):
        raise TypeError("x is not a list")

    levels = []
    mapping = {}
    output = np.zeros((len(x),), dtype=np.int32)

    for i in range(len(x)):
        lev = x[i]

        if lev not in mapping:
            mapping[lev] = len(levels)
            levels.append(len(levels))

        output[i] = mapping[lev]

    return FactorizedArray(levels=levels, indices=output)
