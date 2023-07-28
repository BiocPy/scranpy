import numpy as np
from mattress import TatamiNumericPointer, tatamize

from .._logging import logger
from ..cpphelpers import lib
from ..types import MatrixTypes

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"

def log_norm_counts(x: MatrixTypes, block = None, size_factors = None, center = True, allow_zeros = False, allow_non_finite = False, num_threads = 1):
    if not isinstance(x, TatamiNumericPointer):
        raise ValueError("coming soon when DelayedArray support is implemented")

    use_sf = size_factors != None
    my_size_factors = None
    sf_offset = 0
    if use_sf:
        my_size_factors = size_factors.astype(np.float64)
        sf_offset = my_size_factors.ctypes.data

    use_block = block != None
    block_info = None
    block_offset = 0
    if use_block:
        block_info = factorize(block) # assumes that factorize is available somewhere.
        block_offset = block_info.indices.ctypes.data

    normed = lib.log_norm_counts(
        x.ptr, 
        use_block,
        block_offset,
        use_sf, 
        sf_offset,
        center,
        allow_zeros,
        allow_non_finite,
        num_threads
    )

    return TatamiNumericPointer(ptr = normed, obj = [x.obj, my_size_factors])
