from dataclasses import dataclass
from typing import Optional, Sequence, Union

from numpy import array, float64, ndarray, zeros

from .. import _cpphelpers as lib
from .._utils import MatrixTypes, factorize, process_block, tatamize_input


@dataclass
class GroupedSizeFactorsOptions:
    """Options to pass to :py:meth:`~scranpy.grouped_size_factors.grouped_size_factors`.

    Attributes:
        groups:
            Sequence of group assignments, of length equal to the number of cells.

        rank:
            Number of principal components to obtain in the low-dimensional
            representation prior to clustering. Only used if ``clusters`` is None.

        block:
            Sequence of block assignments, where PCA and clustering is
            performed within each block. Only used if ``clusters`` is None.

        initial_factors:
            Array of initial size factors to obtain a log-normalized matrix prior
            to PCA and clustering. Only used if ``clusters`` is None.

        assay_type:
            Assay containing the count matrix, if ``input`` is a SummarizedExperiment.

        num_threads:
            Number of threads to use for the various calculations.
    """

    rank: int = 25
    groups: Optional[Sequence] = None
    block: Optional[Sequence] = None
    initial_factors: Optional[Sequence] = None
    assay_type: Union[int, str] = 0
    num_threads: int = 1


def grouped_size_factors(
    input: MatrixTypes, options: GroupedSizeFactorsOptions = GroupedSizeFactorsOptions()
) -> ndarray:
    """Compute grouped size factors to remove composition biases between groups of cells.
    This sums all cells from the same group into a pseudo-cell, applies median-based normalization between pseudo-cells,
    and propagates the pseudo-cell size factors back to each cell via library size scaling.

    Args:
        input:
            Matrix-like object where rows are features and columns are cells, typically containing
            expression values of some kind. This should be a matrix class that can be converted into a
            :py:class:`~mattress.TatamiNumericPointer.TatamiNumericPointer`.

            Alternatively, a :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`
            containing such a matrix in its assays.

            Developers may also provide a :py:class:`~mattress.TatamiNumericPointer.TatamiNumericPointer` directly.

        options:
            Further options.

    Returns:
        Array of size factors for each cell in ``input``.
    """
    ptr = tatamize_input(input, options.assay_type)
    output = zeros(ptr.ncol(), dtype=float64)

    if options.groups is not None:
        group_levels, group_indices = factorize(options.groups)
        if len(group_indices) != ptr.ncol():
            raise ValueError(
                "length of 'options.groups' should be equal to number of cells in 'input'"
            )
        lib.grouped_size_factors_with_clusters(
            ptr.ptr, group_indices, output, options.num_threads
        )
    else:
        use_block, num_blocks, block_names, block_info, block_offset = process_block(
            options.block, ptr.ncol()
        )

        use_init_sf = options.initial_factors is not None
        init_sf_info = None
        init_sf_offset = 0
        if use_init_sf:
            if len(options.initial_factors) != ptr.ncol():
                raise ValueError(
                    "length of 'options.initial_factors' should be equal to the number of cells in 'input'"
                )
            init_sf_info = array(options.initial_factors, copy=False, dtype=float64)
            init_sf_offset = init_sf_info.ctypes.data

        lib.grouped_size_factors_without_clusters(
            ptr.ptr,
            use_block,
            block_offset,
            use_init_sf,
            init_sf_offset,
            options.rank,
            output,
            options.num_threads,
        )

    return output
