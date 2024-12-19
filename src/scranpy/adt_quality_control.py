from typing import Union, Sequence, Any, Optional
from dataclasses import dataclass
from collections.abc import Mapping

import numpy
import biocutils
import mattress

from ._utils_qc import _sanitize_subsets
from . import lib_scranpy as lib


@dataclass
class ComputeAdtQcMetricsResults:
    """Results of :py:func:`~compute_adt_qc_metrics`."""

    sum: numpy.array 
    """Sum of counts across all ADTs for each cell."""

    detected: numpy.array 
    """Number of detected ADTs in each cell."""

    subset_sum: biocutils.NamedList
    """Sum of counts across each ADT subset in each cell. Each list element
    corresponds to an ADT subset and is a NumPy array that contains the sum of
    counts for that subset in each cell.
    """


def compute_adt_qc_metrics(x: Any, subsets: Union[Mapping, Sequence], num_threads: int = 1):
    """Compute quality control metrics from ADT count data.

    Args: 
        x:
            A matrix-like object containing ADT counts.

        subsets:
            A specification of the ADT subsets, typically used to
            identify control features like IgGs. This may be either:

            - A list of arrays. Each array corresponds to an ADT subset
              and can either contain boolean or integer values. For booleans,
              the array should be of length equal to the number of rows
              and be truthy for rows that belong in the subset. Integers
              are treated as indices of rows that belong in the subset.
            - A dictionary where keys are the names of each ADT subset and
              the values are arrays as described above.
            - A :py:class:`~biocutils.NamedList.NamedList` where each element
              is an array as described above, possibly with names.

        num_threads:
            Number of threads to use.

    Returns:
        QC metrics computed from the count matrix for each cell.
    """
    ptr = mattress.initialize(x)
    subkeys, subvals = _sanitize_subsets(subsets, x.shape[0])
    osum, odetected, osubset_sum = lib.compute_adt_qc_metrics(ptr.ptr, subvals, num_threads)
    osubset_sum = biocutils.NamedList(osubset_sum, subkeys)
    return ComputeAdtQcMetricsResults(osum, odetected, osubset_sum)


@dataclass
class SuggestAdtQcThresholdsResults:
    """Results of :py:func:`~suggest_adt_qc_metrics`."""

    detected: Union[biocutils.NamedList, float]
    """Threshold on the number of detected ADTs. Cells with lower numbers of
    detected ADTs are considered to be of low quality. If a blocking factor is
    provided, a separate threshold is computed for each level."""

    subset_sum: biocutils.NamedList
    """Thresholds on the sum of counts in each ADT subset. Each element of the
    list corresponds to a ADT subset. Cells with higher totals than the
    threshold are considered to be of low quality. If a blocking factor is
    provided, a separate threshold is computed for each level."""

    block: Optional[list]
    """Levels of the blocking factor. ``None`` if no blocking was performed."""


def suggest_adt_qc_thresholds(
    metrics: ComputeAdtQcMetricsResults,
    block: Optional[Sequence] = None,
    min_detected_drop: float = 0.1,
    num_mads: float = 3.0,
) -> SuggestAdtQcThresholdsResults:
    """Suggest filter thresholds for the ADT-derived QC metrics, typically
    generated from :py:func:`~compute_adt_qc_metrics`.

    Args:
        metrics:
            ADT-derived QC metrics from :py:func:`~compute_adt_qc_metrics`.

        block:
            Factor specifying the block of origin (e.g., batch, sample) for
            each cell in ``metrics``. If supplied, a separate threshold is
            computed from the cells in each block. Alternatively ``None``, if
            all cells are from the same block.

        min_detected_drop:
            Minimum drop in the number of detected ADTs from the median, in
            order to consider a cell to be of low quality.

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

    detected, subset_sums = lib.suggest_adt_qc_thresholds(
        (metrics.sum, metrics.detected, metrics.subset_sum.as_list()),
        blockind,
        min_detected_drop,
        num_mads
    )

    if not blockind is None:
        detected = biocutils.NamedList(detected, blocklev)
        for i, s in enumerate(subset_sums):
            subset_sums[i] = biocutils.NamedList(s, blocklev)

    subset_sums = biocutils.NamedList(subset_sums, metrics.subset_sum.get_names())
    return SuggestAdtQcThresholdsResults(detected, subset_sums, blocklev)


def filter_adt_qc_metrics(
    thresholds: SuggestAdtQcThresholdsResults,
    metrics: ComputeAdtQcMetricsResults,
    block: Optional[Sequence] = None
) -> numpy.ndarray:
    """Filter for high-quality cells based on ADT-derived QC metrics.

    Args:
        thresholds:
            Filter thresholds on the QC metrics, typically computed with
            :py:func:`~suggest_adt_qc_thresholds`.

        metrics:
            ADT-derived QC metrics, typically computed with
            :py:func:`~compute_adt_qc_metrics`.

        block:
            Factor specifying the block of origin (e.g., batch, sample) for
            each cell in ``metrics``. This should be the same as that used
            in :py:func:`~suggest_adt_qc_thresholds`.

    Returns:
        A NumPy vector of length equal to the number of cells in ``metrics``,
        containing truthy values for putative high-quality cells.
    """
    if not thresholds.block is None:
        if block is None:
            raise ValueError("'block' must be supplied if it was used in 'suggest_adt_qc_thresholds'")
        blockind = biocutils.match(block, thresholds.block, dtype=numpy.uint32, fail_missing=True)
        detected = numpy.array(thresholds.detected.as_list(), dtype=numpy.float64)
        subset_sum = [numpy.array(s.as_list(), dtype=numpy.float64) for s in thresholds.subset_sum.as_list()]
    else:
        if not block is None:
            raise ValueError("'block' cannot be supplied if it was not used in 'suggest_adt_qc_thresholds'")
        blockind = None
        detected = thresholds.detected
        subset_sum = numpy.array(thresholds.subset_sum.as_list(), dtype=numpy.float64)

    return lib.filter_adt_qc_metrics(
        (detected, subset_sum),
        (metrics.sum, metrics.detected, metrics.subset_sum.as_list()),
        blockind
    )
