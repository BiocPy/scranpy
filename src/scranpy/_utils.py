from typing import Sequence, Callable, Any, Tuple, Union
from biocutils import factor, match
from mattress import TatamiNumericPointer, tatamize
from numpy import bool_, float64, int32, int_, ndarray, uint8, uintp, zeros, array

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


MatrixTypes = Union[TatamiNumericPointer]


def factorize(x: Sequence) -> Tuple[list, ndarray]:
    lev, ind = factor(x)
    return lev, array(ind, int32)


def is_list_of_type(x: Any, target_type: Callable) -> bool:
    return (isinstance(x, list) or isinstance(x, tuple)) and all(
        isinstance(item, target_type) for item in x
    )


def to_logical(selection: Sequence, length: int) -> ndarray:
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


def match_lists(x, y):
    reordering = match(x, y)
    for i in reordering:
        if i is None:
            return None
    return reordering


def tatamize_input(x) -> TatamiNumericPointer:
    return tatamize(x)


def create_pointer_array(arrs):
    num = len(arrs)
    output = ndarray((num,), dtype=uintp)

    if isinstance(arrs, list):
        for i in range(num):
            output[i] = arrs[i].ctypes.data
    else:
        i = 0
        for k in arrs:
            output[i] = arrs[k].ctypes.data
            i += 1

    return output


def process_block(block: Union[Sequence, None], number: int) -> Tuple:
    use_block = block is not None
    block_lev = None
    block_ind = None
    block_offset = 0
    num_blocks = 1

    if use_block:
        if len(block) != number:
            raise ValueError("length of 'block' should equal the number of cells")

        block_lev, block_ind = factorize(block)
        block_offset = block_ind.ctypes.data
        num_blocks = len(block_lev)

    return use_block, num_blocks, block_lev, block_ind, block_offset