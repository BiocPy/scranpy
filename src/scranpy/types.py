from collections import namedtuple
from typing import Any, Callable, Sequence, Union

import numpy
from mattress import TatamiNumericPointer
from scipy import sparse

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"

MatrixTypes = Union[TatamiNumericPointer, numpy.ndarray, sparse.spmatrix]
SelectionTypes = Union[Sequence, numpy.ndarray, range, slice]

FactorizedArray = namedtuple("FactorizedArray", ["levels", "indices"])
FactorizedArray.__doc__ = """Named tuple of a Factorized Array.

levels (numpy.ndarray): Levels in the array.
indices (numpy.ndarray): Indices.
"""

NDOutputArrays = namedtuple("NDOutputArrays", ["arrays", "references"])
NDOutputArrays.__doc__ = """Named tuple of a list of numpy ndarrays (used for outputs).

array (List[numpy.ndarray]): List of ndarray objects.
references (numpy.ndarray): References to each inner ndarray.
"""


def is_matrix_expected_type(x: Any) -> bool:
    """Checks if `x` is an expected matrix type.

    Args:
        x (Any): Any object.

    Returns:
        bool: True if 'x' is an expected matrix
        (``TatamiNumericPointer``, ``numpy`` or ``scipy`` matrix).
    """
    return (
        isinstance(x, TatamiNumericPointer)
        or isinstance(x, numpy.ndarray)
        or isinstance(x, sparse.spmatrix)
    )


def is_list_of_type(x: Any, target_type: Callable) -> bool:
    """Checks if `x` is a list of `target_type`.

    Args:
        x (Any): Any object.
        target_type (Callable): Type to check for, e.g. str, int.

    Returns:
        bool: True if 'x' is list and all values are of the same type.
    """
    return (isinstance(x, list) or isinstance(x, tuple)) and all(
        isinstance(item, target_type) for item in x
    )


def validate_matrix_types(x: MatrixTypes):
    """Validate if x is an expected matrix type.

    Args:
        x (MatrixTypes): Input Matrix.

    Raises:
        TypeError: If x is not a tatami, numpy or scipy matrix.
    """
    if not is_matrix_expected_type(x):
        raise TypeError(
            f"'x' must be a tatami, numpy or sparse matrix, provided {type(x)}."
        )


def validate_object_type(x: Any, target_type: Callable):
    """Validate if x is an expected object type.

    Args:
        x (MatrixTypes): Input Matrix.
        target_type (Callable): Type to check for, e.g. str, int.

    Raises:
        TypeError: if `x` is not the target type.
    """
    if not isinstance(x, target_type):
        raise TypeError("'x' is not an expected type.")
