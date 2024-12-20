from typing import Optional, Sequence, Literal

import numpy

from . import lib_scranpy as lib


def sanitize_size_factors(
    size_factors: numpy.ndarray,
    replace_zero: bool = True,
    replace_negative: bool = True,
    replace_infinite: bool = True,
    replace_nan: bool = True,
    in_place: bool = False
) -> numpy.ndarray:
    """Replace invalid size factors, i.e., zero, negative, infinite or NaNs.

    Args:
        size_factors:
            Floating-point array containing size factors for all cells.

        replace_zero:
            Whether to replace size factors of zero with the lowest positive
            factor. If False, zeros are retained.

        replace_negative:
            Whether to replace negative size factors with the lowest positive
            factor. If False, negative values are retained.

        replace_infinite:
            Whether to replace infinite size factors with the largest positive
            factor. If False, infinite values are retained.

        replace_nan:
            Whether to replace NaN size factors with unity. If False, NaN
            values are retained.

        in_place:
            Whether to modify ``size_factors`` in place. If False, a new array
            is returned. This argument only used if ``size_factors`` is
            double-precision, otherwise a new array is always returned.

    Returns:
        Array containing sanitized size factors. If ``in_place = True``, this
        is a reference to ``size_factors``.
    """
    local_sf = numpy.array(size_factors, dtype=numpy.float64, copy=not in_place)
    lib.sanitize_size_factors(
        local_sf,
        replace_zero,
        replace_negative,
        replace_infinite,
        replace_nan
    )
    return local_sf
