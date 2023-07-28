from typing import Any, Union

import numpy as np
import scipy.sparse as sp
from mattress import TatamiNumericPointer

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"

MatrixTypes = Union[TatamiNumericPointer, np.ndarray, sp.spmatrix]


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
