from typing import Sequence, Any, Union

import delayedarray
import mattress
import numpy

from . import lib_scranpy as lib

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def normalize_counts(
    x: Any,
    size_factors: Sequence,
    log: bool = True,
    pseudo_count: float = 1,
    log_base: float = 2,
    preserve_sparsity: bool = False 
) -> Union[delayedarray.DelayedArray, mattress.InitializedMatrix]:
    """Create a matrix of (log-transformed) normalized expression values.
    The normalization removes uninteresting per-cell differences due to sequencing efficiency and library size.
    The log-transformation ensures that any differences represent log-fold changes in downstream analysis steps; such relative changes in expression are more relevant than absolute changes.

    Args:
        x:
            Matrix-like object containing cells in columns and features in rows, typically with count data.

            Alternatively, a :py:class:`~mattress.InitializedMatrix.InitializedMatrix` representing a count matrix, typically created by :py:class:`~mattress.initialize.initialize`.

        size_factors:
            Size factor for each cell. This should have length equal to the number of columns in ``x``.
    
        log: 
            Whether log-transformation should be performed.

        pseudo_count:
            Positive pseudo-count to add before log-transformation.
            Ignored if ``log = False``.

        log_base:
            Base of the log-transformation, ignored if ``log = False``.

        preserve_sparsity:
            Whether to preserve sparsity when ``pseudo_count != 1``.
            If ``True``, users should manually add ``log(pseudo_count, log_base)`` to the returned matrix to obtain the desired log-transformed expression values.
            Ignored if ``log = False`` or ``pseudo_count = 1``.

    Returns:
        If ``x`` is a matrix-like object, a :py:class:`~delayedarray.DelayedArray.DelayedArray` is returned containing the (log-transformed) normalized expression matrix.

        If ``x`` is an ``InitializedMatrix``, a new ``InitializedMatrix`` is returned containing the normalized expression matrix.

    References:
        The ``normalize_counts`` function in the `scran_norm <https://github.com/libscran/scran_norm>`_ C++ library, which provides the reference implementation.
    """
    size_factors = numpy.array(size_factors, dtype=numpy.float64, copy=None)

    if isinstance(x, mattress.InitializedMatrix):
        return mattress.InitializedMatrix(
            lib.normalize_counts(
                x.ptr,
                size_factors,
                log,
                pseudo_count,
                log_base,
                preserve_sparsity
            )
        )

    if log and pseudo_count != 1 and preserve_sparsity:
        size_factors = size_factors * pseudo_count # don't use *= as this might modify in place.
        pseudo_count = 1

    x = delayedarray.DelayedArray(x)
    normalized = x / size_factors
    if not log:
        return normalized

    if pseudo_count == 1:
        return numpy.log1p(normalized) / numpy.log(log_base)

    return numpy.log(normalized + pseudo_count) / numpy.log(log_base)
