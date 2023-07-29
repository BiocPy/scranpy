from typing import Optional, Sequence

import numpy as np
from mattress import TatamiNumericPointer, tatamize

from .._logging import logger
from ..cpphelpers import lib
from ..types import MatrixTypes, is_matrix_expected_type
from ..utils import factorize

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def log_norm_counts(
    x: MatrixTypes,
    block: Optional[Sequence] = None,
    size_factors: Optional[np.ndarray] = None,
    center: bool = True,
    allow_zeros: bool = False,
    allow_non_finite: bool = False,
    num_threads: int = 1,
    verbose: bool = False,
) -> TatamiNumericPointer:
    """Compute log normalization.

    This function expects the matrix (`x`) to be features (rows) by cells (columns).

    Args:
        x (MatrixTypes): Inpute matrix.
        block (Optional[Sequence], optional): Array containing the block/batch
            assignment for each cell. Defaults to None.
        size_factors (Optional[np.ndarray], optional): size factors for each cell.
            Defaults to None.
        center (bool, optional): center the size factors?. Defaults to True.
        allow_zeros (bool, optional): allow zeros?. Defaults to False.
        allow_non_finite (bool, optional): all nan or inifnite numbers?.
            Defaults to False.
        num_threads (int, optional): number of threads. Defaults to 1.
        verbose (bool, optional): display logs?. Defaults to False.

    Raises:
        TypeError, ValueError: if arguments don't meet expectations.

    Returns:
        TatamiNumericPointer: log normalized matrix.
    """
    if not is_matrix_expected_type(x):
        raise TypeError(
            f"Input must be a tatami, numpy or sparse matrix, provided {type(x)}."
        )

    if not isinstance(x, TatamiNumericPointer):
        raise ValueError("coming soon when DelayedArray support is implemented")

    NC = x.ncol()

    use_sf = size_factors is not None
    my_size_factors = None
    sf_offset = 0
    if use_sf:
        if size_factors.shape[0] != NC:
            raise ValueError(
                f"Must provide 'size_factors' (provided: {size_factors.shape[0]})"
                f" for all cells (expected: {NC})"
            )

        if not isinstance(size_factors, np.ndarray):
            raise TypeError("'size_factors' must be a numpy ndarray.")

        my_size_factors = size_factors.astype(np.float64)
        sf_offset = my_size_factors.ctypes.data

    use_block = block is not None
    block_info = None
    block_offset = 0
    if use_block:
        if len(block) != NC:
            raise ValueError(
                f"Must provide block assignments (provided: {len(block)})"
                f" for all cells (expected: {NC})."
            )

        block_info = factorize(block)  # assumes that factorize is available somewhere.
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
        num_threads,
    )

    return TatamiNumericPointer(ptr=normed, obj=[x.obj, my_size_factors])
