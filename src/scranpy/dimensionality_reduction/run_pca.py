import ctypes as ct
from collections import namedtuple
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

from numpy import ctypeslib, ndarray, copy

from .. import cpphelpers as lib
from .._logging import logger
from .._utils import to_logical, tatamize_input, factorize, MatrixTypes

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"

PcaResult = namedtuple("PcaResult", ["principal_components", "variance_explained"])
PcaResult.__doc__ = """Named tuple of results from :py:meth:`~scranpy.dimensionality_reduction.run_pca.run_pca`.

principal_components (ndarray): Matrix of principal component (PC) coordinates,
    where the rows are cells and columns are PCs.
variance_explained (ndarray): Array of length equal to the number of PCs,
    containing the percentage of variance explained by each PC.
"""


def _extract_pca_results(pptr: ct.c_void_p, nc: int) -> PcaResult:
    actual_rank = lib.fetch_simple_pca_num_dims(pptr)

    pc_pointer = lib.fetch_simple_pca_coordinates(pptr)

    # In C++, the PCs are stored as a column-major dim*cell matrix. As NumPy
    # typically stores things in row-major format, we switch the dimensions so
    # that it's in a more conventional format when we copy it to Python.
    pc_array = copy(ctypeslib.as_array(pc_pointer, shape=(nc, actual_rank)), order="C")
    principal_components = pc_array

    var_pointer = lib.fetch_simple_pca_variance_explained(pptr)
    var_array = copy(ctypeslib.as_array(var_pointer, shape=(actual_rank,)))
    total = lib.fetch_simple_pca_total_variance(pptr)
    variance_explained = var_array / total

    return PcaResult(principal_components, variance_explained)


@dataclass
class RunPcaOptions:
    """Optional arguments for :py:meth:`~scranpy.dimensionality_reduction.run_pca.run_pca`.

    Attributes:
        rank (int): Number of top PCs to compute.
            Larger values capture more biological structure at the cost of increasing
            computational work and absorbing more random noise.
            Defaults to 25.

        subset (ndarray, optional): Array specifying which features should be
            used in the PCA (e.g., highly variable genes from
            :py:meth:`~scranpy.feature_selection.choose_hvgs.choose_hvgs`).
            This may contain integer indices or booleans.
            Defaults to None, in which all features are used.

        block (Sequence, optional):
            Block assignment for each cell.
            This can be used to reduce the effect of inter-block differences on the PCA
            (see ``block_method`` for more details).

            If provided, this should have length equal to the number of cells, where
            cells have the same value if and only if they are in the same block.
            Defaults to None, indicating all cells are part of the same block.

        scale (bool, optional):
            Whether to scale each feature to unit variance.
            This improves robustness (i.e., reduces sensitivity) to a small number of
            highly variable features. Defaults to False.

        block_method (Literal["none", "project", "regress"], optional): How to adjust
            the PCA for the blocking factor.

            - ``"regress"`` will regress out the factor, effectively performing a PCA on
                the residuals. This only makes sense in limited cases, e.g., inter-block
                differences are linear and the composition of each block is the same.
            - ``"project"`` will compute the rotation vectors from the residuals but
                will project the cells onto the PC space. This focuses the PCA on
                within-block variance while avoiding any assumptions about the
                nature of the inter-block differences. Any removal of block effects
                should be performed separately.
            - ``"none"`` will ignore any blocking factor, i.e., as if ``block = null``.
                Any inter-block differences will both contribute to the determination of
                the rotation vectors and also be preserved in the PC space. Any removal of block effects
                should be performed separately.

            This option is only used if ``block`` is not `null`.
            Defaults to "project".

        block_weights (bool, optional):
            Whether to weight each block so that it contributes the same number of effective observations to the
            covariance matrix. Defaults to True.

        num_threads (int, optional):  Number of threads to use. Defaults to 1.

        verbose (bool): Whether to print logs. Defaults to False.

    Raises:
        ValueError: If ``block_method`` is not an expected value.
    """

    rank: int = 25
    subset: Optional[ndarray] = None
    block: Optional[Sequence] = None
    scale: bool = False
    block_method: Literal["none", "project", "regress"] = "project"
    block_weights: bool = True
    num_threads: int = 1
    verbose: bool = False

    def __post_init__(self):
        if self.block_method not in ["none", "project", "regress"]:
            raise ValueError(
                '\'block_method\' must be one of "none", "project", "regress"'
                f"provided {self.block_method}"
            )


def run_pca(input: MatrixTypes, options: RunPcaOptions = RunPcaOptions()) -> PcaResult:
    """Perform a principal component analysis (PCA) to retain the top PCs. This is used to denoise and compact a dataset
    by removing later PCs associated with random noise, under the assumption that interesting biological heterogeneity
    is the major source of variation in the dataset.

    Args:
        input (MatrixTypes): Matrix-like object where rows are features and columns are cells, typically containing
            log-normalized values. This should be a matrix class that can be converted into a
            :py:class:`~mattress.TatamiNumericPointer`. Developers may also provide the
            :py:class:`~mattress.TatamiNumericPointer` itself.

        options (RunPcaOptions): Optional parameters.

    Raises:
        TypeError: If ``input`` is not an expected type.
        ValueError: if ``options.block`` does not match the number of cells.

    Returns:
        PcaResult: Object containing the PC coordinates and the variance
            explained by each PC.
    """
    x = tatamize_input(input)

    nr = x.nrow()
    nc = x.ncol()

    use_subset = options.subset is not None
    temp_subset = None
    subset_offset = 0
    if use_subset:
        temp_subset = to_logical(options.subset, nr)
        subset_offset = temp_subset.ctypes.data

    result = None
    if options.block is None or (
        options.block_method == "none" and not options.block_weights
    ):
        if options.verbose is True:
            logger.info("No block information is provided, running simple_pca...")

        pptr = lib.run_simple_pca(
            x.ptr,
            options.rank,
            use_subset,
            subset_offset,
            options.scale,
            options.num_threads,
        )
        try:
            result = _extract_pca_results(pptr, nc)

        finally:
            lib.free_simple_pca(pptr)

    else:
        if len(options.block) != nc:
            raise ValueError(
                f"Must provide block assignments (provided: {len(options.block)})"
                f" for all cells (expected: {nc})."
            )

        block_levels, block_indices = factorize(options.block)

        if options.block_method == "regress":
            if options.verbose is True:
                logger.info("Block information is provided, running residual_pca...")

            pptr = lib.run_residual_pca(
                x.ptr,
                block_indices,
                options.block_weights,
                options.rank,
                use_subset,
                subset_offset,
                options.scale,
                options.num_threads,
            )
            try:
                result = _extract_pca_results(pptr, nc)

            finally:
                lib.free_residual_pca(pptr)

        elif options.block_method == "project" or options.block_method == "none":
            if options.verbose is True:
                logger.info("Block information is provided, running multibatch_pca...")

            pptr = lib.run_multibatch_pca(
                x.ptr,
                block_indices,
                (options.block_method == "project"),
                options.block_weights,
                options.rank,
                use_subset,
                subset_offset,
                options.scale,
                options.num_threads,
            )
            try:
                result = _extract_pca_results(pptr, nc)

            finally:
                lib.free_multibatch_pca(pptr)

    if result is None:
        raise RuntimeError("PCA result cannot be empty.")

    return result
