from dataclasses import dataclass
from collections import namedtuple
from typing import Mapping, Optional, Sequence, Union

from biocframe import BiocFrame
from numpy import ndarray, uintp, float64

from .. import cpphelpers as lib
from .._logging import logger
from .._utils import process_block, factorize, tatamize_input, MatrixTypes

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


_NDOutputArrays = namedtuple("_NDOutputArrays", ["arrays", "references"])


def create_output_arrays(length: int, number: int) -> _NDOutputArrays:
    outptrs = ndarray((number,), dtype=uintp)
    outarrs = []
    for g in range(number):
        curarr = ndarray((length,), dtype=float64)
        outptrs[g] = curarr.ctypes.data
        outarrs.append(curarr)
    return _NDOutputArrays(outarrs, outptrs)


def create_output_summary_arrays(rows: int, groups: int) -> _NDOutputArrays:
    output = {
        "min": create_output_arrays(rows, groups),
        "mean": create_output_arrays(rows, groups),
        "min_rank": create_output_arrays(rows, groups),
    }

    outptrs = ndarray((3,), dtype=uintp)
    outptrs[0] = output["min"].references.ctypes.data
    outptrs[1] = output["mean"].references.ctypes.data
    outptrs[2] = output["min_rank"].references.ctypes.data
    return _NDOutputArrays(output, outptrs)


def create_summary_biocframe(summary: _NDOutputArrays, group: int) -> BiocFrame:
    return BiocFrame(
        {
            "min": summary.arrays["min"].arrays[group],
            "mean": summary.arrays["mean"].arrays[group],
            "min_rank": summary.arrays["min_rank"].arrays[group],
        }
    )


@dataclass
class ScoreMarkersOptions:
    """Optional arguments for :py:meth:`~scranpy.marker_detection.score_markers.score_markers`.

    Attributes:
        block (Sequence, optional):
            Block assignment for each cell.
            Comparisons are only performed within each block to avoid interference from
            inter-block differences, e.g., batch effects.

            If provided, this should have length equal to the number of cells, where
            cells have the same value if and only if they are in the same block.
            Defaults to None, indicating all cells are part of the same block.

        threshold (float, optional):
            Log-fold change threshold to use for computing Cohen's d and the AUC.
            Large positive values favor markers with large log-fold changes over those
            with low variance. Defaults to 0.

        compute_auc (bool, optional):
            Whether to compute the AUCs.
            This can be set to ``False`` for greater speed and memory efficiency.
            Defaults to True.

        assay_type (Union[int, str]):
            Assay to use from ``input`` if it is a
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.

        num_threads (int, optional): Number of threads to use. Defaults to 1.

        verbose (bool, optional): Whether to print logs. Defaults to False.
    """

    block: Optional[Sequence] = None
    threshold: float = 0
    compute_auc: bool = True
    assay_type: Union[str, int] = "logcounts"
    num_threads: int = 1
    verbose: bool = False


def score_markers(
    input: MatrixTypes,
    grouping: Sequence,
    options: ScoreMarkersOptions = ScoreMarkersOptions(),
) -> Mapping:
    """Score genes as potential markers for groups of cells. Markers are genes that are strongly up-regulated within
    each group, allowing users to associate each group with some known (or novel) cell type or state. The groups
    themselves are typically constructed from the data, e.g., with
    :py:meth:`~scranpy.clustering.build_snn_graph.build_snn_graph`.

    Args:
        input (MatrixTypes): Matrix-like object where rows are features and columns are cells, typically containing
            expression values of some kind. This should be a matrix class that can be converted into a
            :py:class:`~mattress.TatamiNumericPointer.TatamiNumericPointer`.

            Alternatively, a :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`
            containing such a matrix in its assays.

            Developers may also provide a :py:class:`~mattress.TatamiNumericPointer.TatamiNumericPointer` directly.

        grouping (Sequence, optional):
            Group assignment for each cell.
            This should have length equal to the number of cells,
            where the entry for each cell specifies the assigned group for that cell.

        options (ScoreMarkersOptions): Optional parameters.

    Raises:
        ValueError: If ``input`` is not an expected type.

    Returns:
        Mapping: Dictionary with computed metrics for each group.
    """
    x = tatamize_input(input, options.assay_type)

    nr = x.nrow()
    nc = x.ncol()

    if len(grouping) != nc:
        raise ValueError(
            "Length of 'grouping' should be equal to the number of columns in 'x'"
        )

    group_levels, group_indices = factorize(grouping)
    num_groups = len(group_levels)

    use_block, num_blocks, block_names, block_info, block_offset = process_block(
        options.block, nc
    )

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
        logger.info("Scoring markers for each cluster...")

    lib.score_markers(
        x.ptr,
        num_groups,
        group_indices,
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

        output[group_levels[g]] = BiocFrame(current)

    return output
