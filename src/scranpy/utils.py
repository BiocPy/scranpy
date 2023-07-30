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
            levels.append(lev)

        output[i] = mapping[lev]

    return FactorizedArray(levels=levels, indices=output)


def to_logical(indices: Sequence, length: int) -> np.ndarray:
    """Convert indices to a logical array.

    Args:
        indices (Sequence): array of integer indices.
        length (int): length of the output array, i.e.,
            the maximum possible index plus 1.

    Returns:
        np.ndarray: an array of unsigned 8-bit integers where
            the entries from indices are set to 1.
    """
    output = np.zeros((length,), dtype=np.uint8)
    output[indices] = 1
    return output
