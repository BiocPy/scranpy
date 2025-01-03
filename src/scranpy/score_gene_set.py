from typing import Any, Sequence, Optional, Literal, Tuple
from dataclasses import dataclass

import mattress
import biocutils
import numpy
import delayedarray

from . import lib_scranpy as lib


@dataclass
class ScoreGeneSetResults:
    """Results of :py:func:`~scranpy.score_gene_set.score_gene_set`."""

    scores: numpy.ndarray
    """Floating-point array of length equal to the number of cells, containing the gene set score for each cell."""

    weights: numpy.ndarray
    """Floating-point array of length equal to the number of genes in the set, containing the weight of each gene in terms of its contribution to the score."""


def score_gene_set(
    x: Any,
    set: Sequence,
    rank: int = 1, 
    scale: bool = False,
    block: Optional[Sequence] = None, 
    block_weight_policy: Literal["variable", "equal", "none"] = "variable",
    variable_block_weight: Tuple = (0, 1000),
    extra_work: int = 7,
    iterations: int = 1000,
    seed: int = 5489,
    realized: bool = True,
    num_threads: int =1
) -> ScoreGeneSetResults:
    """Compute per-cell scores for a gene set, defined as the column sums of a rank-1 approximation to the submatrix for the feature set.
    This uses the same approach implemented in the GSDecon package by Jason Hackney.

    Args:
        x: 
            A matrix-like object where rows correspond to genes or genomic features and columns correspond to cells.
            The matrix is expected to contain log-expression values.

        set:
            Array of integer indices specifying the rows of ``x`` belonging to the gene set.
            Alternatively, a sequence of boolean values of length equal to the number of rows, where truthy elements indicate that the corresponding row belongs to the gene set.
        
        rank:
            Rank of the approximation.

        scale:
            Whether to scale all genes to have the same variance.

        block:
           Array of length equal to the number of columns of ``x``, containing the block of origin (e.g., batch, sample) for each cell.
           Alternatively ``None``, if all cells are from the same block.

        block_weight_policy:
            Policy to use for weighting different blocks when computing the average for each statistic.
            Only used if ``block`` is provided.

        variable_block_weight:
            Parameters for variable block weighting.
            This should be a tuple of length 2 where the first and second values are used as the lower and upper bounds, respectively, for the variable weight calculation.
            Only used if ``block`` is provided and ``block_weight_policy = "variable"``.

        extra_work:
            Number of extra dimensions for the IRLBA workspace.

        iterations:
            Maximum number of restart iterations for IRLBA.

        seed:
            Seed for the initial random vector in IRLBA.

        realized:
            Whether to realize ``x`` into an optimal memory layout for IRLBA.
            This speeds up computation at the cost of increased memory usage.

        num_threads:
            Number of threads to use.

    Returns:
        Array of per-cell scores and per-gene weights.

    References:
        https://github.com/libscran/gsdecon, which describes the approach in more detail.
        In particular, see the documentation for the ``compute_blocked`` function for an explanation of the blocking strategy.
    """
    if block is not None:
        blocklev, blockind = biocutils.factorize(block, sort_levels=True, dtype=numpy.uint32, fail_missing=True)
    else:
        blocklev = None
        blockind = None

    x = delayedarray.DelayedArray(x)[set,:]
    mat = mattress.initialize(x)

    scores, weights = lib.score_gene_set(
        mat.ptr,
        rank,
        blockind,
        block_weight_policy,
        variable_block_weight,
        scale,
        realized,
        extra_work,
        iterations,
        seed,
        num_threads
    )

    return ScoreGeneSetResults(scores, weights)
