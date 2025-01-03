from typing import Sequence, Tuple, Union

from biocutils import match

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def match_lists(x, y):
    reordering = match(x, y)
    for i in reordering:
        if i is None:
            return None
    return reordering


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
