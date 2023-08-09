from typing import Mapping

import numpy as np
from biocframe import BiocFrame

from .. import cpphelpers as lib
from .._logging import logger
from ..types import MatrixTypes, NDOutputArrays
from ..utils import create_output_arrays, factorize, validate_and_tatamize_input
from .argtypes import ScoreMarkersArgs

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def create_output_summary_arrays(rows, groups) -> NDOutputArrays:
    """Create a list of ndarrays of shape (rows, groups) for marker detection results.

    Args:
        rows (int): Number of rows.
        groups (int): Number of groups.

    Returns:
        NDOutputArrays: A tuple with list of
        ndarrays and their references.
    """
    output = {
        "min": create_output_arrays(rows, groups),
        "mean": create_output_arrays(rows, groups),
        "min_rank": create_output_arrays(rows, groups),
    }

    outptrs = np.ndarray((3,), dtype=np.uintp)
    outptrs[0] = output["min"].references.ctypes.data
    outptrs[1] = output["mean"].references.ctypes.data
    outptrs[2] = output["min_rank"].references.ctypes.data
    return NDOutputArrays(output, outptrs)


def create_summary_biocframe(summary: NDOutputArrays, group: int) -> BiocFrame:
    """Create a BiocFrame object for score markers results.

    Args:
        summary (NDOutputArrays): Summary result from score markers.
        group (int): Group or cluster to access.

    Returns:
        BiocFrame: A `BiocFrame` with "min", "mean" and "min_rank" scores.
    """
    return BiocFrame(
        {
            "min": summary.arrays["min"].arrays[group],
            "mean": summary.arrays["mean"].arrays[group],
            "min_rank": summary.arrays["min_rank"].arrays[group],
        }
    )


def score_markers(
    input: MatrixTypes, options: ScoreMarkersArgs = ScoreMarkersArgs()
) -> Mapping:
    """Score genes as potential markers for each group of cells.

    This function expects the matrix (`input`) to be features (rows) by cells (columns).

    Args:
        input (MatrixTypes): Input matrix.
        options (ScoreMarkersArgs): additional arguments defined by `ScoreMarkersArgs`.

    Raises:
        ValueError: If inputs do not match expectations.

    Returns:
        Mapping: Dictionary with computed metrics for each group.
    """
    x = validate_and_tatamize_input(input)

    nr = x.nrow()
    nc = x.ncol()

    # TODO: should we do this?
    if options.grouping is None:
        if options.verbose is True:
            logger.info("grouping is none, assigning all cells to the same cluster...")

        options.grouping = [0] * nc

    if len(options.grouping) != nc:
        raise ValueError(
            "length of 'grouping' should be equal to the number of columns in 'x'"
        )

    grouping = factorize(options.grouping)
    num_groups = len(grouping.levels)

    block_offset = 0
    num_blocks = 1
    if options.block is not None:
        if len(options.block) != nc:
            raise ValueError(
                "length of 'block' should be equal to the number of columns in 'x'"
            )
        block = factorize(options.block)
        num_blocks = len(block.levels)
        block_offset = block.indices.ctypes.data

    means = create_output_arrays(nr, num_groups)
    detected = create_output_arrays(nr, num_groups)
    cohen = create_output_summary_arrays(nr, num_groups)
    lfc = create_output_summary_arrays(nr, num_groups)
    delta_detected = create_output_summary_arrays(nr, num_groups)

    auc = None
    auc_offset = 0
    if options.compute_auc is True:
        auc = create_output_summary_arrays(nr, num_groups)
        auc_offset = auc.references.ctypes.data

    if options.verbose is True:
        logger.info("scoring markers for each cluster...")

    lib.score_markers(
        x.ptr,
        num_groups,
        grouping.indices,
        num_blocks,
        block_offset,
        options.compute_auc,
        options.threshold,
        means.references.ctypes.data,
        detected.references.ctypes.data,
        cohen.references.ctypes.data,
        auc_offset,
        lfc.references.ctypes.data,
        delta_detected.references.ctypes.data,
        options.num_threads,
    )

    output = {}
    for g in range(num_groups):
        current = {
            "means": means.arrays[g],
            "detected": detected.arrays[g],
            "cohen": create_summary_biocframe(cohen, g),
            "lfc": create_summary_biocframe(lfc, g),
            "delta_detected": create_summary_biocframe(delta_detected, g),
        }

        if options.compute_auc is True:
            current["auc"] = create_summary_biocframe(auc, g)

        output[grouping.levels[g]] = BiocFrame(current)

    return output
