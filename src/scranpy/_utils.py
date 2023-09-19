from typing import Sequence, Callable, Any, Tuple, Union
from biocutils import factor, match
from mattress import TatamiNumericPointer, tatamize
from numpy import bool_, int32, int_, ndarray, uint8, uintp, zeros, array
from summarizedexperiment import SummarizedExperiment

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


MatrixTypes = Union[TatamiNumericPointer, SummarizedExperiment]


def factorize(x: Sequence) -> Tuple[list, ndarray]:
    lev, ind = factor(x)
    return lev, array(ind, int32)


def to_logical(selection: Sequence, length: int, dtype=uint8) -> ndarray:
    output = zeros((length,), dtype=dtype)

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
        has_bool = False
        has_number = False
        for ss in selection:
            if isinstance(ss, bool):
                has_bool = True
            elif isinstance(ss, int):
                has_number = True

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


def tatamize_input(x: MatrixTypes, assay_type: Union[str, int]) -> TatamiNumericPointer:
    if isinstance(x, SummarizedExperiment):
        x = x.assay(assay_type)
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

    # num_blocks needs to be computed and returned separately from block_lev,
    # as the latter is None when block = None (i.e., all cells in one block).
    return use_block, num_blocks, block_lev, block_ind, block_offset
