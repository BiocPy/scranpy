import ctypes as ct
from collections import namedtuple
from copy import deepcopy
from typing import Literal, Optional, Sequence

import numpy as np

from ..cpphelpers import lib
from ..types import MatrixTypes
from ..utils import factorize, to_logical, validate_and_tatamize_input

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"

PCAResult = namedtuple("PCAResult", ["principal_components", "variance_explained"])


def _extract_pca_results(pptr: ct.c_void_p, nc: int) -> PCAResult:
    """Extract principal components and variance explained from scran results.

    Args:
        pptr (ct.c_void_p): a pointer to scran::MultiBatchPca::Results.

    Returns:
        PCAResult: with principal components and variance explained.
    """
    actual_rank = lib.fetch_simple_pca_num_dims(pptr)

    pc_pointer = ct.cast(
        lib.fetch_simple_pca_coordinates(pptr), ct.POINTER(ct.c_double)
    )
    pc_array = deepcopy(np.ctypeslib.as_array(pc_pointer, shape=(actual_rank, nc)))
    principal_components = pc_array

    var_pointer = ct.cast(
        lib.fetch_simple_pca_variance_explained(pptr), ct.POINTER(ct.c_double)
    )
    var_array = deepcopy(np.ctypeslib.as_array(var_pointer, shape=(actual_rank,)))
    total = lib.fetch_simple_pca_total_variance(pptr)
    variance_explained = var_array / total

    return PCAResult(principal_components, variance_explained)


def run_pca(
    x: MatrixTypes,
    rank: int,
    subset: Optional[Sequence] = None,
    block: Optional[Sequence] = None,
    scale: bool = False,
    block_method: Literal["none", "project", "regress"] = "project",
    block_weights: bool = True,
    num_threads: int = 1,
) -> PCAResult:
    """Run Prinicpal Component Analysis (PCA).

    Args:
        x (MatrixTypes): Inpute Matrix.
        rank (int): Number of top PC's to compute.
        subset (Mapping, optional): Array specifying which features should be
            retained (e.g., HVGs). This should be of length equal to the number of
            rows in `x`; elements should be `true` to retain each row.
            Defaults to None, then all features are retained.
        block (Sequence, optional): Array containing the block/batch
            assignment for each cell. Defaults to None.
        scale (bool, optional): Whether to scale each feature to unit variance.
            Defaults to False.
        block_method (Literal["none", "project", "regress"], optional): How to adjust
            the PCA for the blocking factor.
            - `"regress"` will regress out the factor, effectively performing a PCA on
                the residuals. This only makes sense in limited cases, e.g., inter-block
                differences are linear and the composition of each block is the same.
            - `"project"` will compute the rotation vectors from the residuals but will
                project the cells onto the PC space. This focuses the PCA on
                within-block variance while avoiding any assumptions about the
                nature of the inter-block differences.
            - `"none"` will ignore any blocking factor, i.e., as if `block = null`. Any
                inter-block differences will both contribute to the determination of
                the rotation vectors and also be preserved in the PC space.
                This option is only used if `block` is not `null`.
            Defaults to "project".
        block_weights (bool, optional): Whether to weight each block so that it
            contributes the same number of effective observations to the covariance
            matrix. Defaults to True.
        num_threads (int, optional):  Number of threads to use. Defaults to 1.

    Raises:
        ValueError: if inputs do not match with expectations.

    Returns:
        PCAResult: principal components and variable explained metrics.
    """
    x = validate_and_tatamize_input(x)

    if block_method not in ["none", "project", "regress"]:
        raise ValueError(
            '\'block_method\' must be one of "none", "residual" or "project"'
            f"provided {block_method}"
        )

    nr = x.nrow()
    nc = x.ncol()

    use_subset = subset is not None
    temp_subset = None
    subset_offset = 0
    if use_subset:
        if len(subset) != nr:
            raise ValueError(
                f"'subsets (provided: {len(subset)}) must be the same as number "
                f"of features (expected: {nr})."
            )
        temp_subset = to_logical(subset, nr)
        subset_offset = temp_subset.ctypes.data

    result = None
    if block is None or (block_method == "none" and not block_weights):
        pptr = lib.run_simple_pca(
            x.ptr, rank, use_subset, subset_offset, scale, num_threads
        )
        try:
            result = _extract_pca_results(pptr, nc)

        finally:
            lib.free_simple_pca(pptr)

    else:
        if len(block) != nc:
            raise ValueError(
                f"Must provide block assignments (provided: {len(block)})"
                f" for all cells (expected: {nc})."
            )

        block_info = factorize(block)
        block_offset = block_info.indices.ctypes.data

        if block_method == "regress":
            pptr = lib.run_residual_pca(
                x.ptr,
                block_offset,
                block_weights,
                rank,
                use_subset,
                subset_offset,
                scale,
                num_threads,
            )
            try:
                result = _extract_pca_results(pptr, nc)

            finally:
                lib.free_residual_pca(pptr)

        elif block_method == "project" or block_method == "none":
            pptr = lib.run_multibatch_pca(
                x.ptr,
                block_offset,
                (block_method == "project"),
                block_weights,
                rank,
                use_subset,
                subset_offset,
                scale,
                num_threads,
            )
            try:
                result = _extract_pca_results(pptr, nc)

            finally:
                lib.free_multibatch_pca(pptr)

    if result is None:
        raise RuntimeError("PCA result cannot be empty.")

    return result
