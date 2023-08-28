from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from mattress import TatamiNumericPointer, tatamize
from numpy import float64, ndarray, log1p, log
from delayedarray import DelayedArray
from copy import copy

from .. import cpphelpers as lib
from ..types import validate_matrix_types
from ..utils import factorize
from .center_size_factors import center_size_factors, CenterSizeFactorsOptions

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


@dataclass
class LogNormCountsOptions:
    """Optional arguments for
    :py:meth:`~scranpy.normalization.log_norm_counts.log_norm_counts`.

    Attributes:
        size_factors (ndarray, optional): Size factors for each cell.
            Defaults to None, in which case the library sizes are used.

        delayed (bool): Whether to force the log-normalization to be
            delayed. This reduces memory usage by avoiding unnecessary
            copies of the count matrix.

        center (bool, optional): Whether to center the size factors. Defaults to True.

        center_size_factors_options (CenterSizeFactorsOptions, optional):
            Optional arguments to pass to :py:meth:`~scranpy.normalization.center_size_factors.center_size_factors`
            if ``center = True``.

        num_threads (int, optional): Number of threads to use to compute size factors,
            if none are provided in ``size_factors``. Defaults to 1.

        verbose (bool, optional): Whether to print logs. Defaults to False.
    """

    block: Optional[Sequence] = None
    size_factors: Optional[ndarray] = None
    center: bool = True
    center_size_factors_options: CenterSizeFactorsOptions = field(
        default_factory=CenterSizeFactorsOptions
    )
    num_threads: int = 1
    verbose: bool = False


def log_norm_counts(
    input, options: LogNormCountsOptions = LogNormCountsOptions()
): 
    """Compute log-transformed normalized values.
    The normalization removes uninteresting per-cell differences due to sequencing efficiency and library size.
    The subsequent log-transformation ensures that any differences in the log-values represent log-fold changes in downstream analysis steps;
    these relative changes in expression are more relevant than absolute changes.

    Args:
        input: 
            Matrix-like object containing cells in columns and features in rows, typically with count data.
            This should be a matrix class that can be converted into a :py:class:`~mattress.TatamiNumericPointer`.
            Developers may also provide the :py:class:`~mattress.TatamiNumericPointer` itself.

        options (LogNormCountsOptions): Optional parameters.

    Raises:
        TypeError, ValueError: If arguments don't meet expectations.

    Returns:
        The log-normalized matrix, either as a :py:class:`~mattress.TatamiNumericPointer`
        if ``input`` is also a :py:class:`~mattress.TatamiNumericPointer`; as a
        :py:class:`~delayedarray.DelayedArray`, if ``input`` is array-like and
        ``delayed = True``; or an object of the same type as ``input`` otherwise.
    """

    is_ptr = isinstance(input, TatamiNumericPointer)

    my_size_factors = options.size_factors
    if my_size_factors is not None:
        ptr = input
        if not is_ptr:
            ptr = tatamize(input)
        my_size_factors = ptr.column_sums(num_threads = options.num_threads)
    else:
        my_size_factors = my_size_factors.astype(float64, copy=True) # just make a copy and avoid problems.

    if center:
        optcopy = copy(center_size_factors_options)
        optcopy.in_place = True
        center_size_factors(my_size_factors, center_size_factors_options)

    if is_ptr:
        if not isinstance(input, DelayedArray) and options.delayed:
            input = DelayedArray(input)
        return log1p(input / my_size_factors) / log(2) 
    else:
        normed = lib.log_norm_counts(input.ptr, my_size_factors)
        return TatamiNumericPointer(ptr=normed, obj=[input.obj, my_size_factors])
