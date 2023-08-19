from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from numpy import float64, ndarray

from .. import cpphelpers as lib
from ..utils import factorize

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


@dataclass
class CenterSizeFactorsOptions:
    """Optional arguments for
    :py:meth:`~scranpy.normalization.center_size_factors.center_size_factors`.

    Attributes:
        block (Sequence, optional):
            Block assignment for each cell.
            This is used to adjust the centering of size factors so that higher-coverage blocks are scaled down.

            If provided, this should have length equal to the number of cells, where
            cells have the same value if and only if they are in the same block.
            Defaults to None, indicating all cells are part of the same block.

        in_place (bool, optional):
            Whether to modify the size factors in place.
            If False, a new array is returned.
            This argument is ignored if the input ``size_factors`` are not double-precision,
            in which case a new array is always returned.

        verbose (bool, optional): Whether to print logs. Defaults to False.
    """

    block: Optional[Sequence] = None
    in_place: bool = False
    verbose: bool = False


def center_size_factors(
    size_factors: ndarray, 
    options: CenterSizeFactorsOptions = CenterSizeFactorsOptions()
) -> ndarray:
    """Center size factors before computing normalized values from the count matrix.
    This ensures that the normalized values are on the same scale as the original counts for easier interpretation.

    Args:
        size_factors (ndarray):
            Floating-point array containing size factors for all cells.
            
        options (CenterSizeFactorsOptions): Optional parameters.

    Raises:
        TypeError, ValueError: If arguments don't meet expectations.

    Returns:
        Array containing centered size factors.
    """

    local_sf = size_factors.astype(float64, copy=not options.in_place)
    NC = local_sf.shape[0]

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

    lib.center_size_factors(
        NC,
        local_sf,
        use_block,
        block_offset
    )

    return local_sf
