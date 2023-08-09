import numpy as np
from mattress import TatamiNumericPointer

from .. import cpphelpers as lib
from ..types import MatrixTypes, validate_matrix_types
from ..utils import factorize
from .argtypes import LogNormalizeCountsArgs

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def log_norm_counts(
    x: MatrixTypes, options: LogNormalizeCountsArgs = LogNormalizeCountsArgs()
) -> TatamiNumericPointer:
    """Compute log normalization.

    This function expects the matrix (`x`) to be features (rows) by cells (columns).

    Args:
        x (MatrixTypes): Inpute matrix.
        options (LogNormalizeCountsArgs): additional arguments defined
            by `LogNormalizeCountsArgs`.

    Raises:
        TypeError, ValueError: If arguments don't meet expectations.

    Returns:
        TatamiNumericPointer: Log normalized matrix.
    """
    validate_matrix_types(x)

    if not isinstance(x, TatamiNumericPointer):
        raise ValueError("coming soon when DelayedArray support is implemented")

    NC = x.ncol()

    use_sf = options.size_factors is not None
    my_size_factors = None
    sf_offset = 0
    if use_sf:
        if options.size_factors.shape[0] != NC:
            raise ValueError(
                f"Must provide 'size_factors' (provided: {options.size_factors.shape[0]})"
                f" for all cells (expected: {NC})"
            )

        if not isinstance(options.size_factors, np.ndarray):
            raise TypeError("'size_factors' must be a numpy ndarray.")

        my_size_factors = options.size_factors.astype(np.float64)
        sf_offset = my_size_factors.ctypes.data

    use_block = options.block is not None
    block_info = None
    block_offset = 0
    if use_block:
        if len(options.block) != NC:
            raise ValueError(
                f"Must provide block assignments (provided: {len(options.block)})"
                f" for all cells (expected: {NC})."
            )

        block_info = factorize(
            options.block
        )  # assumes that factorize is available somewhere.
        block_offset = block_info.indices.ctypes.data

    normed = lib.log_norm_counts(
        x.ptr,
        use_block,
        block_offset,
        use_sf,
        sf_offset,
        options.center,
        options.allow_zeros,
        options.allow_non_finite,
        options.num_threads,
    )

    return TatamiNumericPointer(ptr=normed, obj=[x.obj, my_size_factors])
