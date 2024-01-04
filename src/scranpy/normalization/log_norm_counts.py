from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

from delayedarray import DelayedArray
from mattress import TatamiNumericPointer, tatamize
from numpy import array, float64, log, log1p, ndarray

from .. import _cpphelpers as lib
from .center_size_factors import CenterSizeFactorsOptions, center_size_factors

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


@dataclass
class LogNormCountsOptions:
    """Optional arguments for :py:meth:`~scranpy.normalization.log_norm_counts.log_norm_counts`.

    Attributes:
        size_factors:
            Size factors for each cell.
            Defaults to None, in which case the library sizes are used.

        delayed:
            Whether to force the log-normalization to be
            delayed. This reduces memory usage by avoiding unnecessary
            copies of the count matrix.

        center:
            Whether to center the size factors. Defaults to True.

        center_size_factors_options:
            Optional arguments to pass to :py:meth:`~scranpy.normalization.center_size_factors.center_size_factors`
            if ``center = True``.

        with_size_factors:
            Whether to return the (possibly centered) size factors in the output.

        assay_type:
            Assay to use from ``input`` if it is a
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.

        num_threads:
            Number of threads to use to compute size factors,
            if none are provided in ``size_factors``. Defaults to 1.
    """

    block: Optional[Sequence] = None
    size_factors: Optional[ndarray] = None
    center: bool = True
    center_size_factors_options: CenterSizeFactorsOptions = field(
        default_factory=CenterSizeFactorsOptions
    )
    delayed: bool = True
    with_size_factors: bool = False
    assay_type: Union[str, int] = 0
    num_threads: int = 1


def log_norm_counts(input, options: LogNormCountsOptions = LogNormCountsOptions()):
    """Compute log-transformed normalized values. The normalization removes uninteresting per-cell differences due to
    sequencing efficiency and library size. The subsequent log-transformation ensures that any differences in the log-
    values represent log-fold changes in downstream analysis steps; these relative changes in expression are more
    relevant than absolute changes.

    Args:
        input:
            Matrix-like object containing cells in columns and features in rows, typically with count data.
            This should be a matrix class that can be converted into a :py:class:`~mattress.TatamiNumericPointer`.
            Developers may also provide the :py:class:`~mattress.TatamiNumericPointer` itself.

            Alternatively, a :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`
            containing such a matrix in its assays.

            Developers may also provide a :py:class:`~mattress.TatamiNumericPointer.TatamiNumericPointer` directly.

        options:
            Optional parameters.

    Raises:
        TypeError, ValueError:
            If arguments don't meet expectations.

    Returns:
        If `options.with_size_factors = False`, the log-normalized matrix is
        directly returned. This is either a :py:class:`~mattress.TatamiNumericPointer`,
        if ``input`` is also a :py:class:`~mattress.TatamiNumericPointer`; as a
        :py:class:`~delayedarray.DelayedArray`, if ``input`` is array-like and
        ``delayed = True``; or otherwise, an object of the same type as ``input``.

        If `options.with_size_factors = True`, a 2-tuple is returned containing
        the log-normalized matrix and an array of (possibly centered) size factors.
    """

    is_ptr = isinstance(input, TatamiNumericPointer)

    my_size_factors = options.size_factors
    if my_size_factors is None:
        ptr = input
        if not is_ptr:
            ptr = tatamize(input)
        my_size_factors = ptr.column_sums(num_threads=options.num_threads)
    elif isinstance(my_size_factors, ndarray):
        my_size_factors = my_size_factors.astype(
            float64, copy=True
        )  # just make a copy and avoid problems.
    else:
        my_size_factors = array(my_size_factors, dtype=float64)

    if options.center:
        optcopy = copy(options.center_size_factors_options)
        optcopy.in_place = True
        center_size_factors(my_size_factors, optcopy)

    mat = None
    if is_ptr:
        normed = lib.log_norm_counts(input.ptr, my_size_factors)
        mat = TatamiNumericPointer(ptr=normed, obj=[input.obj, my_size_factors])
    else:
        if not isinstance(input, DelayedArray) and options.delayed:
            input = DelayedArray(input)
        mat = log1p(input / my_size_factors) / log(2)

    if options.with_size_factors:
        return mat, my_size_factors
    else:
        return mat
