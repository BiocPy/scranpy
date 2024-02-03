from dataclasses import dataclass
from typing import Sequence, Union

from numpy import array, float64, int32, ndarray, zeros

from .. import _cpphelpers as lib


@dataclass
class HypergeometricTestOptions:
    """Options for :py:meth:`~scranpy.feature_set_enrichment.hypergeometric_tail.hypergeometric_tail`.

    Attributes:
        log:
            Whether to report log-transformed p-values.

        upper_tail:
            Whether to compute the upper tail of the hypergeometric distribution,
            i.e., test for overrepresentation.

        num_threads:
            Number of threads to use.
    """

    log: bool = False
    upper_tail: bool = True
    num_threads: int = 1


def _recycle_vector(x):
    if isinstance(x, ndarray):
        return x.astype(int32, copy=False)
    elif not isinstance(x, Sequence):
        x = [x]
    return array(x, dtype=int32)


def hypergeometric_test(
    markers_in_set: Union[int, Sequence[int]],
    set_size: Union[int, Sequence[int]],
    total_markers: Union[int, Sequence[int]],
    total_genes: Union[int, Sequence[int]],
    options=HypergeometricTestOptions(),
) -> ndarray:
    """Run the hypergeometric test to identify enrichment of interesting (usually marker) genes in a feature set or
    pathway.

    Args:
        markers_in_set:
            Array containing the number of markers inside the feature sets.
            Alternatively, a single integer containing this number.

        set_size:
            Array containing the sizes of the feature sets.
            Alternatively, a single integer containing this number.

        total_markers:
            Array containing the total number of markers.
            Alternatively, a single integer containing this number.

        total_genes:
            Array containing the total number of genes in the analysis.
            Alternatively, a single integer containing this number.

        options:
            Further options.

    Returns:
        Array of p-values of length equal to the length
        of the input arrays (or 1, if all inputs were scalars).

    Each array input is expected to be 1-dimensional and of the same length,
    where a hypergeometric test is applied on the corresponding values across
    all arrays. However, any of the arguments may be integers, in which case
    they are recycled to the length of the arrays for testing.
    """
    markers_in_set = _recycle_vector(markers_in_set)
    set_size = _recycle_vector(set_size)
    total_markers = _recycle_vector(total_markers)
    total_genes = _recycle_vector(total_genes)

    num_genes = set(
        [len(markers_in_set), len(set_size), len(total_markers), len(total_genes)]
    )
    if len(num_genes) > 1:
        num_genes.remove(1)
    if len(num_genes) == 0:
        num_genes = 1
    elif len(num_genes) == 1:
        num_genes = list(num_genes)[0]
    else:
        raise ValueError("arguments should be of length 1 or the same length")

    output = zeros(num_genes, dtype=float64)
    lib.hypergeometric_test(
        num_genes,
        len(markers_in_set),
        markers_in_set.ctypes.data,
        len(set_size),
        set_size.ctypes.data,
        len(total_markers),
        total_markers.ctypes.data,
        len(total_genes),
        total_genes.ctypes.data,
        options.log,
        options.upper_tail,
        options.num_threads,
        output,
    )

    return output
