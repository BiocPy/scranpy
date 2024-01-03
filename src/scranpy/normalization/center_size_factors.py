from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from numpy import float64, ndarray

from .. import _cpphelpers as lib
from .._utils import process_block

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


@dataclass
class CenterSizeFactorsOptions:
    """Optional arguments for :py:meth:`~scranpy.normalization.center_size_factors.center_size_factors`.

    Attributes:
        block:
            Block assignment for each cell.
            This is used to adjust the centering of size factors so that higher-coverage blocks are scaled down.

            If provided, this should have length equal to the number of cells, where
            cells have the same value if and only if they are in the same block.
            Defaults to None, indicating all cells are part of the same block.

        in_place:
            Whether to modify the size factors in place.
            If False, a new array is returned.
            This argument is ignored if the input ``size_factors`` are not double-precision,
            in which case a new array is always returned.

        allow_zeros:
            Whether to gracefully handle zero size factors.
            If True, zero size factors are automatically set to the smallest non-zero size factor.
            If False, an error is raised.
            Defaults to False.

        allow_non_finite:
            Whether to gracefully handle missing or infinite size factors.
            If True, infinite size factors are automatically set to the largest non-zero size factor,
            while missing values are automatically set to 1.
            If False, an error is raised.
    """

    block: Optional[Sequence] = None
    in_place: bool = False
    allow_zeros: bool = False
    allow_non_finite: bool = False


def center_size_factors(
    size_factors: ndarray,
    options: CenterSizeFactorsOptions = CenterSizeFactorsOptions(),
) -> ndarray:
    """Center size factors before computing normalized values from the count matrix. This ensures that the normalized
    values are on the same scale as the original counts for easier interpretation.

    Args:
        size_factors:
            Floating-point array containing size factors for all cells.

        options:
            Optional parameters.

    Raises:
        TypeError, ValueError:
            If arguments don't meet expectations.

    Returns:
        Array containing centered size factors.
    """

    local_sf = size_factors.astype(float64, copy=not options.in_place)
    NC = local_sf.shape[0]

    use_block, num_blocks, block_names, block_indices, block_offset = process_block(
        options.block, NC
    )

    lib.center_size_factors(
        NC,
        local_sf,
        options.allow_zeros,
        options.allow_non_finite,
        use_block,
        block_offset,
    )

    return local_sf
