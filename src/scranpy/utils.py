from typing import Sequence

import numpy as np
from mattress import TatamiNumericPointer, tatamize

from .types import (
    FactorizedArray,
    MatrixTypes,
    NDOutputArrays,
    SelectionTypes,
    is_list_of_type,
    validate_matrix_types,
)

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def factorize(x: Sequence) -> FactorizedArray:
    """Factorize an array.

    Args:
        x (Sequence): Any array.

    Returns:
        FactorizedArray: A factorized tuple.
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


def to_logical(selection: SelectionTypes, length: int) -> np.ndarray:
    """Convert a selection to a logical array.

    Args:
        selection (SelectionTypes): List/array of integer indices.
            a list/array of booleans, a range or a slice object.
        length (int): Length of the output array, i.e.,
            the maximum possible index plus 1.

    Returns:
        np.ndarray: An array of unsigned 8-bit integers where selected
        entries are marked with 1 and all others are zero.

        If `selection` is an array of indices, the entries at the
        specified indices are set to 1.

        If `selection` is an array of booleans, the entries are
        converted directly to unsigned 8 bit integers.
    """
    output = np.zeros((length,), dtype=np.uint8)

    if isinstance(selection, range) or isinstance(selection, slice):
        output[selection] = 1
        return output

    if isinstance(selection, np.ndarray):
        if selection.dtype == np.bool_:
            if len(selection) != length:
                raise ValueError("length of 'selection' is not equal to 'length'.")
            output[selection] = 1
            return output
        elif selection.dtype == np.int_:
            output[selection] = 1
            return output
        else:
            raise TypeError(
                "'selection`s' dtype not supported, must be 'boolean' or 'int',"
                f"provided {selection.dtype}"
            )

    has_bool = is_list_of_type(selection, bool)
    has_number = is_list_of_type(selection, int)

    if (has_number and has_bool) or (not has_bool and not has_number):
        raise TypeError("'selection' should only contain booleans or numbers")

    if has_bool:
        if len(selection) != length:
            raise ValueError("length of 'selection' is not equal to 'length'.")
        output[:] = selection
    elif has_number:
        output[selection] = 1

    return output


def validate_and_tatamize_input(x: MatrixTypes) -> TatamiNumericPointer:
    """Validate and tatamize the input matrix.

    Args:
        x (MatrixTypes): Input Matrix.

    Raises:
        TypeError: If x is not an expected matrix type.

    Returns:
        TatamiNumericPointer: Tatami representation.
    """
    validate_matrix_types(x)

    if not isinstance(x, TatamiNumericPointer):
        x = tatamize(x)

    return x


def create_output_arrays(rows: int, columns: int) -> NDOutputArrays:
    """Create a list of ndarrays of shape (rows, columns).

    Args:
        rows (int): Number of rows.
        columns (int): Number of columns.

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
