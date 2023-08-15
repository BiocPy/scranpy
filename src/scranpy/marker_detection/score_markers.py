from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from biocframe import BiocFrame
from numpy import ndarray, uintp

from .. import cpphelpers as lib
from .._logging import logger
from ..types import MatrixTypes, NDOutputArrays
from ..utils import create_output_arrays, factorize, validate_and_tatamize_input

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def create_output_summary_arrays(rows: int, groups: int) -> NDOutputArrays:
    output = {
        "min": create_output_arrays(rows, groups),
        "mean": create_output_arrays(rows, groups),
        "min_rank": create_output_arrays(rows, groups),
    }

    outptrs = ndarray((3,), dtype=uintp)
    outptrs[0] = output["min"].references.ctypes.data
    outptrs[1] = output["mean"].references.ctypes.data
    outptrs[2] = output["min_rank"].references.ctypes.data
    return NDOutputArrays(output, outptrs)


def create_summary_biocframe(summary: NDOutputArrays, group: int) -> BiocFrame:
    return BiocFrame(
        {
            "min": summary.arrays["min"].arrays[group],
            "mean": summary.arrays["mean"].arrays[group],
            "min_rank": summary.arrays["min_rank"].arrays[group],
        }
    )


@dataclass
class ScoreMarkersOptions:
    """Optional arguments for
    :py:meth:`~scranpy.marker_detection.score_markers.score_markers`.

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

        num_threads (int, optional): Number of threads to use. Defaults to 1.
        verbose (bool, optional): Whether to print logs. Defaults to False.
    """

    block: Optional[Sequence] = None
    threshold: float = 0
    compute_auc: bool = True
    num_threads: int = 1
    verbose: bool = False


def score_markers(
    input: MatrixTypes,
    grouping: Sequence,
    options: ScoreMarkersOptions = ScoreMarkersOptions(),
) -> Mapping:
    """Score genes as potential markers for groups of cells.
    Markers are genes that are strongly up-regulated within each group,
    allowing users to associate each group with some known (or novel) cell type or state.
    The groups themselves are typically constructed from the data, e.g., with
    :py:meth:`~scranpy.clustering.build_snn_graph.build_snn_graph`.

    Args:
        input (MatrixTypes):
            Matrix-like object where rows are features and columns are cells, typically containing log-normalized values.
            This should be a matrix class that can be converted into a :py:class:`~mattress.TatamiNumericPointer`.
            Developers may also provide the :py:class:`~mattress.TatamiNumericPointer` itself.

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
    x = validate_and_tatamize_input(input)

    nr = x.nrow()
    nc = x.ncol()

    if len(grouping) != nc:
        raise ValueError(
            "Length of 'grouping' should be equal to the number of columns in 'x'"
        )

    grouping = factorize(grouping)
    num_groups = len(grouping.levels)

    block_offset = 0
    num_blocks = 1
    if options.block is not None:
        if len(options.block) != nc:
            raise ValueError(
                "Length of 'block' should be equal to the number of columns in 'x'"
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
        logger.info("Scoring markers for each cluster...")

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
