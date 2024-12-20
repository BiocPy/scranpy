from typing import Any

import mattress
import numpy

from . import lib_scranpy as lib


def compute_clrm1_factors(x: Any, num_threads: int = 1) -> numpy.ndarray:
    """Compute size factors from an ADT count matrix using the CLRm1 method.

    Args:
        x:
            A matrix-like object containing ADT count data. Rows correspond to
            tags and columns correspond to cells.

        num_threads:
            Number of threads to use.

    Returns:
        Array containing the CLRm1 size factor for each cell.
    """
    ptr = mattress.initialize(x)
    return lib.compute_clrm1_factors(ptr.ptr, num_threads)
