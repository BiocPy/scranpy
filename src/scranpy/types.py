from collections import namedtuple
from typing import Any, Callable, Union

import numpy as np
import scipy.sparse as sp
from mattress import TatamiNumericPointer

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"

MatrixTypes = Union[TatamiNumericPointer, np.ndarray, sp.spmatrix]
FactorizedArray = namedtuple("FactorizedArray", ["levels", "indices"])


def is_matrix_expected_type(x: Any) -> bool:
    """Checks if `x` is an expect matrix type.

    Args:
        x (Any): any object.

    Returns:
        bool: True if `x` is supported.
    """
    return (
        isinstance(x, TatamiNumericPointer)
        or isinstance(x, np.ndarray)
        or isinstance(x, sp.spmatrix)
    )


def is_list_of_type(x: Any, target_type: Callable) -> bool:
    """Checks if `x` is a list of `target_type`.

    Args:
        x (Any): any object.
        target_type (Callable): Type to check for, e.g. str, int

    Returns:
        bool: True if `x` is list and all values are of the same type.
    """
    return (isinstance(x, list) or isinstance(x, tuple)) and all(
        isinstance(item, target_type) for item in x
    )
