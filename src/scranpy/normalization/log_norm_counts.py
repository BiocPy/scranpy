from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from mattress import TatamiNumericPointer

from .. import cpphelpers as lib
from ..types import MatrixTypes, validate_matrix_types
from ..utils import factorize

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


@dataclass
class LogNormCountsOptions:
    """Optional arguments for 
    :py:meth:`~scranpy.normalization.log_norm_counts.log_norm_counts`.

    Attributes:
        block (Sequence, optional): 
            Block assignment for each cell.
            This is used to adjust the centering of size factors so that higher-coverage blocks are scaled down.

            If provided, this should have length equal to the number of cells, where cells have the same value if and only if they are in the same block.
            Defaults to None, indicating all cells are part of the same block.

        size_factors (np.ndarray, optional): Size factors for each cell.
            Defaults to None.

        center (bool, optional): Whether to center the size factors. Defaults to True.

        allow_zeros (bool, optional): Whether to gracefully handle zero size factors. 
            If True, zero size factors are automatically set to the smallest non-zero size factor.
            If False, an error is raised.
            Defaults to False.

        allow_non_finite (bool, optional): Whether to gracefully handle missing or infinite size factors.
            If True, infinite size factors are automatically set to the largest non-zero size factor,
            while missing values are automatically set to 1.
            If False, an error is raised.

        num_threads (int, optional): Number of threads to use to compute size factors,
            if none are provided in ``size_factors``. Defaults to 1.

        verbose (bool, optional): Whether to print logs. Defaults to False.
    """

    block: Optional[Sequence] = None
    size_factors: Optional[np.ndarray] = None
    center: bool = True
    allow_zeros: bool = False
    allow_non_finite: bool = False
    num_threads: int = 1
    verbose: bool = False


def log_norm_counts(
    input: MatrixTypes, options: LogNormCountsOptions = LogNormCountsOptions()
) -> TatamiNumericPointer:
    """Compute log-transformed normalized values.
    The normalization removes uninteresting per-cell differences due to sequencing efficiency and library size.
    The subsequent log-transformation ensures that any differences in the log-values represent log-fold changes in downstream analysis steps; 
    these relative changes in expression are more relevant than absolute changes.

    Args:
        input (MatrixTypes): 
            Matrix-like object containing cells in columns and features in rows, typically with count data.
            This should be a matrix class that can be converted into a :py:class:`~mattress.TatamiNumericPointer`.
            Developers may also provide the :py:class:`~mattress.TatamiNumericPointer` itself.
        options (LogNormCountsOptions): Optional parameters.

    Raises:
        TypeError, ValueError: If arguments don't meet expectations.

    Returns:
        TatamiNumericPointer: Log-normalized expression matrix.
    """
    validate_matrix_types(input)

    if not isinstance(input, TatamiNumericPointer):
        raise ValueError("Coming soon when DelayedArray support is implemented")

    NC = input.ncol()

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
        input.ptr,
        use_block,
        block_offset,
        use_sf,
        sf_offset,
        options.center,
        options.allow_zeros,
        options.allow_non_finite,
        options.num_threads,
    )

    return TatamiNumericPointer(ptr=normed, obj=[input.obj, my_size_factors])
