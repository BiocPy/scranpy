from typing import Optional, Sequence, Literal, Tuple, Any
from dataclasses import dataclass

import numpy
import biocutils
import mattress

from . import lib_scranpy as lib


@dataclass
class RunPcaResults:
    """Results of :py:func:`~run_pca`."""

    components: numpy.ndarray
    """Floating-point matrix of principal component (PC) scores.
    Rows are dimensions (i.e., PCs) and columns are cells."""

    rotation: numpy.ndarray
    """Floating-point rotation matrix.
    Rows are genes and columns are dimensions (i.e., PCs)."""

    variance_explained: numpy.ndarray
    """Floating-point array of length equal to the number of PCs, containing the variance explained by each successive PC."""

    total_variance: float
    """Total variance in the dataset."""

    center: numpy.ndarray
    """If ``block`` was used in :py:func:`~run_pca`, this is a floating-point matrix containing the mean for each gene (column) in each block of cells (row).
    Otherwise, this is a floating-point array of length equal to the number of genes, containing the mean for each gene across all cells."""

    scale: Optional[numpy.ndarray]
    """Floating-point array containing the scaling factor applied to each gene. 
    Only reported if ``scale = True``, otherwise it is ``None``."""

    block: Optional[list]
    """Levels of the blocking factor, corresponding to each row of ``center``.
    This is ``None`` if no blocking was performed."""


def run_pca(
    x: Any,
    number: int = 25,
    scale: bool = False,
    block: Optional[Sequence] = None, 
    block_weight_policy: Literal["variable", "equal", "none"] = "variable",
    variable_block_weight: Tuple = (0, 1000),
    components_from_residuals: bool = False,
    extra_work: int = 7,
    iterations: int = 1000,
    seed: int = 5489,
    realized: bool = True,
    num_threads: int =1
) -> RunPcaResults:
    """Run a PCA on the gene-by-cell log-expression matrix to obtain a low-dimensional representation for downstream analyses.

    Args:
        x:
            A matrix-like object where rows correspond to genes or genomic features and columns correspond to cells.
            Typically, the matrix is expected to contain log-expression values, and the rows should be filtered to relevant (e.g., highly variable) genes.

        number:
            Number of PCs to retain.

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

        components_from_residuals:
            Whether to compute the PC scores from the residuals in the presence of a blocking factor.
            If ``False``, the residuals are only used to compute the rotation matrix, and the original expression values of the cells are projected onto this new space.
            Only used if ``block`` is provided.

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
        The results of the PCA.

    References:
        https://github.com/libscran/scran_pca, which describes the approach in more detail.
        In particular, see the documentation for the ``blocked_pca`` function for an explanation of the blocking strategy.
    """
    if block is not None:
        blocklev, blockind = biocutils.factorize(block, sort_levels=True, dtype=numpy.uint32, fail_missing=True)
    else:
        blocklev = None
        blockind = None

    mat = mattress.initialize(x)
    pcs, rotation, varexp, total_var, center, out_scale = lib.run_pca(
        mat.ptr,
        number,
        blockind,
        block_weight_policy,
        variable_block_weight,
        components_from_residuals,
        scale,
        realized,
        extra_work,
        iterations,
        seed,
        num_threads
    )

    if not scale:
        out_scale = None
    return RunPcaResults(pcs, rotation, varexp, total_var, center, out_scale, blocklev)
