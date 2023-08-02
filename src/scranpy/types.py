from collections import namedtuple
from typing import Any, Callable, Sequence, Union

import numpy as np
import scipy.sparse as sp
from mattress import TatamiNumericPointer

from .nearest_neighbors import NeighborIndex, NeighborResults

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"

MatrixTypes = Union[TatamiNumericPointer, np.ndarray, sp.spmatrix]
SelectionTypes = Union[Sequence, np.ndarray, range, slice]


FactorizedArray = namedtuple("FactorizedArray", ["levels", "indices"])
FactorizedArray.__doc__ = """Named tuple of a factorized array.

levels (np.ndarray): levels in the array.
indices (np.ndarray): indices.
"""
NeighborIndexOrResults = Union[NeighborIndex, NeighborResults, np.ndarray]
NDOutputArrays = namedtuple("NDOutputArrays", ["arrays", "references"])
NDOutputArrays.__doc__ = """Named tuple of a list of numpy ndarrays (used for outputs).

array (List[np.ndarray]): list of ndarray objects.
references (np.ndarray): references to each inner ndarray.
"""


def is_matrix_expected_type(x: Any) -> bool:
    """Checks if `x` is an expected matrix type.

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


def is_neighbor_class(x: Any) -> bool:
    """Checks if `x` is an expected nearest neighbor input.

    Args:
        x (Any): any object.

    Returns:
        bool: True if `x` is supported.
    """
    return (
        isinstance(x, NeighborIndex)
        or isinstance(x, NeighborResults)
        or isinstance(x, np.ndarray)
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


def validate_matrix_types(x: MatrixTypes):
    """Validate if x is an expected matrix type.

    Args:
        x (MatrixTypes): Inpute Matrix

    Raises:
        TypeError: if x is not a tatami, numpy or scipy matrix.
    """
    if not is_matrix_expected_type(x):
        raise TypeError(
            f"Input must be a tatami, numpy or sparse matrix, provided {type(x)}."
        )
