import ctypes as ct
from collections import namedtuple
from copy import deepcopy

import numpy as np

from .. import cpphelpers as lib
from .._logging import logger
from ..types import MatrixTypes
from ..utils import factorize, to_logical, validate_and_tatamize_input
from .argtypes import RunPcaArgs

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


def run_pca(x: MatrixTypes, options: RunPcaArgs = RunPcaArgs()) -> PCAResult:
    """Run Prinicpal Component Analysis (PCA).

    Args:
        x (MatrixTypes): Input Matrix.
        options (PcaArgs): Optional arguments specified by
            `PcaArgs`.

    Raises:
        ValueError: If inputs do not match with expectations.

    Returns:
        PCAResult: Principal components and variable explained metrics.
    """
    x = validate_and_tatamize_input(x)

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
            logger.info("no block information is provided, running simple_pca...")

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
                logger.info("block information is provided, running residual_pca...")

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
                logger.info("block information is provided, running multibatch_pca...")

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
