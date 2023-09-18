from dataclasses import dataclass

from .._utils import MatrixTypes, tatamize_input, factorize, process_block


@dataclass
class GroupedSizeFactorsOptions:
    """Options to pass to :py:meth:`~scranpy.grouped_size_factors.grouped_size_factors`.

    Attributes:
        clusters (Sequence, optional):
            Sequence of cluster assignments of length equal to the number of cells.

        rank (int): 
            Number of principal components to obtain in the low-dimensional
            representation prior to clustering. Only used if ``clusters`` is None.

        block (Sequence, optional): 
            Sequence of block assignments, where PCA and clustering is
            performed within each block. Only used if ``clusters`` is None.

        num_threads (int):
            Number of threads to use for the various calculations.
    """
    rank: int = 25
    clusters: Optional[Sequence] = None
    block: Optional[Sequence] = None
    assay_type: Union[int, str] = 0
    num_threads: int = 1


def grouped_size_factors(input: MatrixTypes, options: GroupedSizeFactorsOptions = GroupedSizeFactorsOptions()):
    ptr = tatamize_input(input, options.assay_type)
    output = ndarray(ptr.ncol(), dtype=float64)

    if options.clusters is None:
        cluster_levels, cluster_indices = factorize(options.clusters)
        if len(cluster_indices) != ptr.ncol():
            raise ValueError("length of 'options.clusters' should be equal to number of cells in 'input'")
        lib.grouped_size_factors_with_clusters(ptr.ptr, output, options.num_threads)
    else:
        use_block, num_blocks, block_names, block_info, block_offset = process_block(options.block, NC)
        lib.grouped_size_factors_without_cluster(ptr.ptr, use_block, block_offset, options.rank, output, options.num_threads)

    return output
