import ctypes as ct
from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np

from .. import cpphelpers as lib
from .._logging import logger
from ..types import MatrixTypes
from ..utils import factorize, to_logical, validate_and_tatamize_input

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"

PCAResult = namedtuple("PCAResult", ["principal_components", "variance_explained"])
PCAResult.__doc__ = """Named tuple of results from run pca step.

principal_components (np.ndarray): principal components.
variance_explained (np.ndarray): variance explained by each PC.
"""


def _extract_pca_results(pptr: ct.c_void_p, nc: int) -> PCAResult:
    """Extract principal components and variance explained from scran results.

    Args:
        pptr (ct.c_void_p): Pointer to scran::MultiBatchPca::Results.

    Returns:
        PCAResult: Results with principal components and variance explained.
    """
    actual_rank = lib.fetch_simple_pca_num_dims(pptr)

    pc_pointer = lib.fetch_simple_pca_coordinates(pptr)
    pc_array = deepcopy(np.ctypeslib.as_array(pc_pointer, shape=(actual_rank, nc)))
    principal_components = pc_array

    var_pointer = lib.fetch_simple_pca_variance_explained(pptr)
    var_array = deepcopy(np.ctypeslib.as_array(var_pointer, shape=(actual_rank,)))
    total = lib.fetch_simple_pca_total_variance(pptr)
    variance_explained = var_array / total

    return PCAResult(principal_components, variance_explained)


@dataclass
class RunPcaArgs:
    """Arguments for principal components analysis -
    :py:meth:`~scranpy.dimensionality_reduction.run_pca.run_pca`.

    Attributes:
        rank (int): Number of top PC's to compute.
        subset (Mapping, optional): Array specifying which features should be
            retained (e.g., HVGs). This may contain integer indices or booleans.
            Defaults to None, then all features are retained.
        block (Sequence, optional): Block assignment for each cell.
            This is used to segregate cells in order to perform comparisons within
            each block. Defaults to None, indicating all cells are part of the same
            block.
        scale (bool, optional): Whether to scale each feature to unit variance.
            Defaults to False.
        block_method (Literal["none", "project", "regress"], optional): How to adjust
            the PCA for the blocking factor.

            - ``"regress"`` will regress out the factor, effectively performing a PCA on
                the residuals. This only makes sense in limited cases, e.g., inter-block
                differences are linear and the composition of each block is the same.
            - ``"project"`` will compute the rotation vectors from the residuals but
                will project the cells onto the PC space. This focuses the PCA on
                within-block variance while avoiding any assumptions about the
                nature of the inter-block differences.
            - ``"none"`` will ignore any blocking factor, i.e., as if ``block = null``.
                Any inter-block differences will both contribute to the determination of
                the rotation vectors and also be preserved in the PC space.
                This option is only used if ``block`` is not `null`.
            Defaults to "project".
        block_weights (bool, optional): Whether to weight each block so that it
            contributes the same number of effective observations to the covariance
            matrix. Defaults to True.
        num_threads (int, optional):  Number of threads to use. Defaults to 1.
        verbose (bool): Display logs? Defaults to False.

    Raises:
        ValueError: If ``block_method`` is not an expected value.
    """

    rank: int = 50
    subset: Optional[Sequence] = None
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


def run_pca(input: MatrixTypes, options: RunPcaArgs = RunPcaArgs()) -> PCAResult:
    """Run Prinicpal Component Analysis (PCA).

    Args:
        input (MatrixTypes): Input Matrix.
        options (PcaArgs): Optional parameters.

    Raises:
        TypeError: If ``input`` is not an expected type.
        ValueError: if ``options.block`` does not match the number of cells.

    Returns:
        PCAResult: Principal components and variable explained metrics.
    """
    x = validate_and_tatamize_input(input)

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

        block_info = factorize(options.block)

        if options.block_method == "regress":
            if options.verbose is True:
                logger.info("Block information is provided, running residual_pca...")

            pptr = lib.run_residual_pca(
                x.ptr,
                block_info.indices,
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
                block_info.indices,
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
