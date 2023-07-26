from collections import namedtuple
from typing import Any, Union

import numpy as np
import scipy.sparse as sp
from mattress import TatamiNumericPointer

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"

MatrixTypes = Union[TatamiNumericPointer, np.ndarray, sp.spmatrix]
RnaQcResult = namedtuple("RnaQcResult", ["sums", "detected", "subset_proportions"])


def is_matrix_expected_type(x: Any) -> bool:
    return (
        isinstance(x, TatamiNumericPointer)
        or isinstance(x, np.ndarray)
        or isinstance(x, sp.spmatrix)
    )
