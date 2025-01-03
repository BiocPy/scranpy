from typing import Optional, Sequence, Literal

import numpy
import biocutils

from . import lib_scranpy as lib

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def center_size_factors(
    size_factors: numpy.ndarray,
    block: Optional[Sequence] = None,
    mode: Literal["lowest", "per-block"] = "lowest",
    in_place: bool = False,
) -> numpy.ndarray:
    """Center size factors before computing normalized values from the count matrix.
    This ensures that the normalized values are on the same scale as the original counts for easier interpretation.

    Args:
        size_factors:
            Floating-point array containing size factors for all cells.

        block:
            Block assignment for each cell.
            If provided, this should have length equal to the number of cells, where cells have the same value if and only if they are in the same block.
            Defaults to ``None``, where all cells are treated as being part of the same block.

        mode: 
            How to scale size factors across blocks.
            ``lowest`` will scale all size factors by the lowest per-block average.
            ``per-block`` will center the size factors in each block separately.
            This argument is only used if ``block`` is provided.

        in_place:
            Whether to modify ``size_factors`` in place.
            If ``False``, a new array is returned.
            This argument only used if ``size_factors`` is double-precision, otherwise a new array is always returned.

    Returns:
        Array containing centered size factors.
        If ``in_place = True``, this is a reference to ``size_factors``.

    References:
        The ``center_size_factors`` function in the `scran_norm <https://github.com/libscran/scran_norm>`_ C++ library, which describes the rationale behind centering.
    """
    if in_place:
        do_copy = None
    else:
        do_copy = True
    local_sf = numpy.array(size_factors, dtype=numpy.float64, copy=do_copy)

    if block is not None:
        _, blockind = biocutils.factorize(block, sort_levels=True, dtype=numpy.uint32, fail_missing=True)
    else:
        blockind = None

    lib.center_size_factors(
        local_sf, 
        blockind,
        mode == "lowest"
    )

    return local_sf
