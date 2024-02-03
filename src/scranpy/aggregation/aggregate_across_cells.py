from numpy import zeros, float64, int32, uintp
from typing import Sequence, Tuple, Union
from dataclasses import dataclass
from biocframe import BiocFrame
from summarizedexperiment import SummarizedExperiment

from .._utils import factorize, tatamize_input, MatrixTypes
from .. import _cpphelpers as lib


class _CombinedFactors:
    def __init__(self, ptr):
        self._ptr = ptr
        self._n = lib.get_combined_factors_size(ptr)

    def __del__(self):
        lib.free_combined_factors(self._ptr)

    def size(self):
        return self._n

    def levels(self, i):
        output = zeros((self._n,), dtype=int32)
        lib.get_combined_factors_level(self._ptr, i, output)
        return output

    def counts(self):
        output = zeros((self._n,), dtype=int32)
        lib.get_combined_factors_count(self._ptr, output)
        return output


@dataclass
class AggregateAcrossCellsOptions:
    """Options to pass to :py:meth:`~scranpy.aggregation.aggregate_across_cells.aggregate_across_cells`.

    Attributes:
        compute_sums:
            Whether to compute the sum of each group.

        compute_detected:
            Whether to compute the number of detected cells in each group.

        assay_type:
            Assay to use from ``input`` if it is a
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.

        num_threads:
            Number of threads.
    """

    compute_sums: bool = True
    compute_detected: bool = True
    assay_type: Union[int, str] = 0
    num_threads: int = 1


def aggregate_across_cells(
    input: MatrixTypes,
    groups: Union[Sequence, Tuple[Sequence], dict, BiocFrame],
    options: AggregateAcrossCellsOptions = AggregateAcrossCellsOptions(),
) -> SummarizedExperiment:
    """Aggregate expression values for groups of cells.

    Args:
        input: Matrix-like object where rows are features and columns are cells, typically containing
            expression values of some kind. This should be a matrix class that can be converted into a
            :py:class:`~mattress.TatamiNumericPointer.TatamiNumericPointer`.

            Alternatively, a :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`
            containing such a matrix in its assays.

            Developers may also provide a :py:class:`~mattress.TatamiNumericPointer.TatamiNumericPointer` directly.

        groups:
            A sequence of length equal to the number of columns of ``input``, specifying the group
            to which each column is assigned. Alternatively, a tuple, dictionary, or
            :py:class:`~biocframe.BiocFrame` of one or more such sequences, in which case
            each unique combination of levels across all sequences is defined as a "group".

        options:
            Further options.

    Returns:
        A SummarizedExperiment where each row corresponds to a row in ``input`` and each
        column corresponds to a group. Assays contain the sum of expression values (if
        ``options.compute_sums = True``) and the number of cells with detected expression (if
        ``options.compute_detected = True``) for each group. Column data contains the identity
        of each group; for ``groups`` containing multiple sequences, the identity of each
        group is defined as a unique combination of levels from each sequence.
    """
    x = tatamize_input(input, options.assay_type)
    NR = x.nrow()
    NC = x.ncol()

    factors = []
    factor_names = None
    if isinstance(groups, tuple):
        factors = [*groups]
    elif isinstance(groups, BiocFrame):
        factor_names = groups.column_names
        for gr in factor_names:
            factors.append(groups.column(gr))
    elif isinstance(groups, dict):
        factor_names = list(groups.keys())
        for gr in factor_names:
            factors.append(groups[gr])
    else:
        factors.append(groups)

    stored_levels = []
    stored_indices = []
    for i, si in enumerate(factors):
        if len(si) != NC:
            raise ValueError(
                "length of grouping vectors should be equal to the number of columns of 'input'"
            )
        lev, ind = factorize(si)
        stored_levels.append(lev)
        stored_indices.append(ind)

    # Possibly combining factors here.
    nstored = len(stored_indices)
    combined = None
    levels = []
    counts = None

    if nstored == 1:
        levels = [stored_levels[0]]
        combined = stored_indices[0]
        counts = zeros((len(levels[0]),), dtype=int32)
        for il in combined:
            counts[il] += 1
    else:
        indptrs = zeros((nstored,), dtype=uintp)
        for i, si in enumerate(stored_indices):
            indptrs[i] = si.ctypes.data

        combined = zeros((NC,), dtype=int32)
        ptr = _CombinedFactors(
            lib.combine_factors(NC, nstored, indptrs.ctypes.data, combined)
        )
        for i, sl in enumerate(stored_levels):
            outlev = ptr.levels(i)
            levels.append([sl[j] for j in outlev])
        counts = ptr.counts()

    ngroups = len(counts)

    # Actually doing the aggregation.
    sums_out = None
    sums_out_ptr = 0
    if options.compute_sums:
        sums_out = zeros((ngroups, NR), dtype=float64)
        sums_out_ptr = sums_out.ctypes.data

    detected_out = None
    detected_out_ptr = 0
    if options.compute_detected:
        detected_out = zeros((ngroups, NR), dtype=int32)
        detected_out_ptr = detected_out.ctypes.data

    lib.aggregate_across_cells(
        x.ptr,
        groups=combined,
        ngroups=ngroups,
        do_sums=options.compute_sums,
        output_sums=sums_out_ptr,
        do_detected=options.compute_detected,
        output_detected=detected_out_ptr,
        nthreads=options.num_threads,
    )

    # Formatting the output.
    output_assays = {}
    if options.compute_sums:
        output_assays["sums"] = sums_out.T
    if options.compute_detected:
        output_assays["detected"] = detected_out.T

    output = SummarizedExperiment(output_assays)
    if nstored == 1:
        output.column_names = levels[0]

    if factor_names is None:
        factor_names = ["factor_" + str(i + 1) for i in range(nstored)]

    reported_factors = {}
    for i, x in enumerate(levels):
        reported_factors[factor_names[i]] = x
    reported_factors["counts"] = counts
    output.column_data = BiocFrame(reported_factors)

    return output
