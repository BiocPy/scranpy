from typing import Sequence

from mattress import TatamiNumericPointer, tatamize
from numpy import bool_, float64, int32, int_, ndarray, uint8, uintp, zeros

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
    output = zeros((len(x),), dtype=int32)

    for i in range(len(x)):
        lev = x[i]

        if lev not in mapping:
            mapping[lev] = len(levels)
            levels.append(lev)

        output[i] = mapping[lev]

    return FactorizedArray(levels=levels, indices=output)


def to_logical(selection: SelectionTypes, length: int) -> ndarray:
    """Convert a selection to a logical array.

    Args:
        selection (SelectionTypes): 
            List/array of integer indices, a range or a slice object.

            Alternatively, a list/array of booleans.

            An empty sequence is treated as a zero-length list of integers.

        length (int): 
            Length of the output array, i.e., the maximum possible index plus 1.

    Returns:
        ndarray: An array of unsigned 8-bit integers where selected
        entries are marked with 1 and all others are zero.

        If `selection` is an array of indices, the entries at the
        specified indices are set to 1.

        If `selection` is an array of booleans, the entries are
        converted directly to unsigned 8 bit integers.
    """
    output = zeros((length,), dtype=uint8)

    if isinstance(selection, range) or isinstance(selection, slice):
        output[selection] = 1
        return output

    if isinstance(selection, ndarray):
        if selection.dtype == bool_:
            if len(selection) != length:
                raise ValueError("length of 'selection' is not equal to 'length'.")
            output[selection] = 1
            return output
        elif selection.dtype == int_:
            output[selection] = 1
            return output
        else:
            raise TypeError(
                "'selection`s' dtype not supported, must be 'boolean' or 'int',"
                f"provided {selection.dtype}"
            )

    if len(selection) == 0:
        has_bool = False 
        has_number = True 
    else:
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

def match_lists(x, y, return_on_missing = True):
    mapping = {}
    counter = 0
    for y0 in y:
        mapping[y0] = counter
        counter += 1

    reordering = []
    for x0 in x:
        if x0 in mapping:
            reordering.append(mapping[x0])
        elif return_on_missing:
            return None
        else:
            reordering.append(None)

    return reordering

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
    outptrs = ndarray((columns,), dtype=uintp)
    outarrs = []
    for g in range(columns):
        curarr = ndarray((rows,), dtype=float64)
        outptrs[g] = curarr.ctypes.data
        outarrs.append(curarr)
    return NDOutputArrays(outarrs, outptrs)
