from typing import Union, Sequence, Any, Optional
from dataclasses import dataclass
from collections.abc import Mapping

import numpy
import biocutils
import mattress

from ._utils_qc import _sanitize_subsets
from . import lib_scranpy as lib


@dataclass
class ComputeCrisprQcMetricsResults:
    """Results of :py:func:`~compute_crispr_qc_metrics`."""

    sum: numpy.array 
    """Sum of counts across all CRISPR guides for each cell."""

    detected: numpy.array 
    """Number of detected CRISPR guides in each cell."""

    max_value: numpy.array 
    """Maximum count for any single guide in each cell."""

    max_index: numpy.array 
    """Row index of the guide with the highest count in each cell."""


def compute_crispr_qc_metrics(x: Any, num_threads: int = 1):
    """Compute quality control metrics from CRISPR count data.

    Args: 
        x:
            A matrix-like object containing CRISPR counts.

        num_threads:
            Number of threads to use.

    Returns:
        QC metrics computed from the count matrix for each cell.
    """
    ptr = mattress.initialize(x)
    osum, odetected, omaxval, omaxind = lib.compute_crispr_qc_metrics(ptr.ptr, num_threads)
    return ComputeCrisprQcMetricsResults(osum, odetected, omaxval, omaxind)


@dataclass
class SuggestCrisprQcThresholdsResults:
    """Results of :py:func:`~suggest_crispr_qc_metrics`."""

    max_value: Union[biocutils.NamedList, float]
    """Threshold on the maximum count in each cell. Cells with lower maxima are
    considered to be of low quality. If a blocking factor is provided, a
    separate threshold is computed for each level."""

    block: Optional[list]
    """Levels of the blocking factor. ``None`` if no blocking was performed."""


def suggest_crispr_qc_thresholds(
    metrics: ComputeCrisprQcMetricsResults,
    block: Optional[Sequence] = None,
    num_mads: float = 3.0,
) -> SuggestCrisprQcThresholdsResults:
    """Suggest filter thresholds for the CRISPR-derived QC metrics, typically
    generated from :py:func:`~compute_crispr_qc_metrics`.

    Args:
        metrics:
            CRISPR-derived QC metrics from :py:func:`~compute_crispr_qc_metrics`.

        block:
            Factor specifying the block of origin (e.g., batch, sample) for
            each cell in ``metrics``. If supplied, a separate threshold is
            computed from the cells in each block. Alternatively ``None``, if
            all cells are from the same block.

        num_mads:
            Number of median from the median, used to define the threshold for
            outliers in each metric.

    Returns:
        Suggested filters on the relevant QC metrics.
    """
    if not block is None:
        blocklev, blockind = biocutils.factorize(block, sort_levels=True, dtype=numpy.uint32, fail_missing=True)
    else:
        blocklev = None
        blockind = None

    max_value, = lib.suggest_crispr_qc_thresholds(
        (metrics.sum, metrics.detected, metrics.max_value, metrics.max_index),
        blockind,
        num_mads
    )

    if not blockind is None:
        max_value = biocutils.NamedList(max_value, blocklev)
    return SuggestCrisprQcThresholdsResults(max_value, blocklev)


def filter_crispr_qc_metrics(
    thresholds: SuggestCrisprQcThresholdsResults,
    metrics: ComputeCrisprQcMetricsResults,
    block: Optional[Sequence] = None
) -> numpy.ndarray:  
    """Filter for high-quality cells based on CRISPR-derived QC metrics.

    Args:
        thresholds:
            Filter thresholds on the QC metrics, typically computed with
            :py:func:`~suggest_crispr_qc_thresholds`.

        metrics:
            CRISPR-derived QC metrics, typically computed with
            :py:func:`~compute_crispr_qc_metrics`.

        block:
            Factor specifying the block of origin (e.g., batch, sample) for
            each cell in ``metrics``. This should be the same as that used
            in :py:func:`~suggest_crispr_qc_thresholds`.

    Returns:
        A NumPy vector of length equal to the number of cells in ``metrics``,
        containing truthy values for putative high-quality cells.
    """
    if not thresholds.block is None:
        if block is None:
            raise ValueError("'block' must be supplied if it was used in 'suggest_crispr_qc_thresholds'")
        blockind = biocutils.match(block, thresholds.block, dtype=numpy.uint32, fail_missing=True)
        max_value = numpy.array(thresholds.max_value.as_list(), dtype=numpy.float64)
    else:
        if not block is None:
            raise ValueError("'block' cannot be supplied if it was not used in 'suggest_crispr_qc_thresholds'")
        blockind = None
        max_value = thresholds.max_value

    return lib.filter_crispr_qc_metrics(
        (max_value,),
        (metrics.sum, metrics.detected, metrics.max_value, metrics.max_index),
        blockind
    )
