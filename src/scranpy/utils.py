from typing import Sequence

import numpy as np
from mattress import TatamiNumericPointer, tatamize

from .types import FactorizedArray, MatrixTypes, NDOutputArrays, validate_matrix_types

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


def to_logical(selection: Sequence, length: int) -> np.ndarray:
    """Convert a selection to a logical array.

    Args:
        selection (Sequence): list/array of integer indices.
            Alternatively, a list/array of booleans.
        length (int): length of the output array, i.e.,
            the maximum possible index plus 1.

    Returns:
        An array of unsigned 8-bit integers where selected
        entries are marked with 1 and all others are zero.

        If `selection` is an array of indices, the entries at the
        specified indices are set to 1.

        If `selection` is an array of booleans, the entries are
        converted directly to unsigned 8 bit integers.
    """
    output = np.zeros((length,), dtype=np.uint8)
    if isinstance(selection, np.ndarray):
        if selection.dtype == np.bool_ and len(selection) != length:
            raise ValueError("length of 'selection' is not equal to 'length'")
        output[indices] = 1
        return output

    has_bool = False
    has_number = False
    has_other = False
    for x in selection:
        if isinstance(x, bool):
            has_bool = True
        elif isinstance(x, int):
            has_number = True
        else:
            has_other = True

    if has_other:
        raise TypeError("'selection' should only contain booleans or numbers")
    elif has_number and has_bool:
        raise TypeError("'selection' cannot contain mixed booleans and numbers")

    if has_bool:
        if len(selection) != length:
            raise ValueError("length of 'selection' is not equal to 'length'")
        output[:] = selection
    elif has_number:
        output[i] = 1

    return output

def validate_and_tatamize_input(x: MatrixTypes) -> TatamiNumericPointer:
    """Validate and tatamize the input matrix.

    Args:
        x (MatrixTypes): Input Matrix.

    Raises:
        TypeError: if x is not an expected matrix type.

    Returns:
        TatamiNumericPointer: tatami representation.
    """
    validate_matrix_types(x)

    if not isinstance(x, TatamiNumericPointer):
        x = tatamize(x)

    return x


def create_output_arrays(rows: int, columns: int) -> NDOutputArrays:
    """Create a list of ndarrays of shape (rows, columns).

    Args:
        rows (int): number of rows.
        columns (int): number of columns.

    Returns:
        NDOutputArrays: A tuple with list of
        ndarrays and their references.
    """
    outptrs = np.ndarray((columns,), dtype=np.uintp)
    outarrs = []
    for g in range(columns):
        curarr = np.ndarray((rows,), dtype=np.float64)
        outptrs[g] = curarr.ctypes.data
        outarrs.append(curarr)
    return NDOutputArrays(outarrs, outptrs)
