from typing import Union, Sequence, Any, Optional
from dataclasses import dataclass
from collections.abc import Mapping

import numpy
import biocutils
import mattress

from ._utils_qc import _sanitize_subsets
from . import lib_scranpy as lib


@dataclass
class ComputeRnaQcMetricsResults:
    """Results of :py:func:`~compute_rna_qc_metrics`."""

    sum: numpy.array 
    """Sum of counts across all genes for each cell."""

    detected: numpy.array 
    """Number of detected genes in each cell."""

    subset_proportion: biocutils.NamedList
    """Proportion of counts in each gene subset in each cell. Each list element
    corresponds to a gene subset and is a NumPy array that contains the
    proportion of counts in that subset in each cell."""


def compute_rna_qc_metrics(x: Any, subsets: Union[Mapping, Sequence], num_threads: int = 1):
    """Compute quality control metrics from RNA count data.

    Args: 
        x:
            A matrix-like object containing RNA counts.

        subsets:
            A specification of the gene subsets, typically used to identify
            control features like mitochondrial genes. This may be either:

            - A list of arrays. Each array corresponds to a feature subset
              and can either contain boolean or integer values. For booleans,
              the array should be of length equal to the number of rows
              and be truthy for rows that belong in the subset. Integers
              are treated as indices of rows that belong in the subset.
            - A dictionary where keys are the names of each feature subset and
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
    osum, odetected, osubset_proportion = lib.compute_rna_qc_metrics(ptr.ptr, subvals, num_threads)
    osubset_proportion = biocutils.NamedList(osubset_proportion, subkeys)
    return ComputeRnaQcMetricsResults(osum, odetected, osubset_proportion)


@dataclass
class SuggestRnaQcThresholdsResults:
    """Results of :py:func:`~suggest_rna_qc_metrics`."""

    sum: Union[biocutils.NamedList, float]
    """Threshold on the sum of counts in each cell. Cells with lower totals are
    considered to be of low quality. If a blocking factor is provided, a
    separate threshold is computed for each level."""

    detected: Union[biocutils.NamedList, float]
    """Threshold on the number of detected genes. Cells with lower numbers of
    detected genes are considered to be of low quality. If a blocking factor is
    provided, a separate threshold is computed for each level."""

    subset_proportion: biocutils.NamedList
    """Thresholds on the proportion of each gene subset. Each element of the
    list corresponds to a gene subset. Cells with higher totals than the
    threshold are considered to be of low quality. If a blocking factor is
    provided, a separate threshold is computed for each level."""

    block: Optional[list]
    """Levels of the blocking factor. ``None`` if no blocking was performed."""


def suggest_rna_qc_thresholds(
    metrics: ComputeRnaQcMetricsResults,
    block: Optional[Sequence] = None,
    num_mads: float = 3.0,
) -> SuggestRnaQcThresholdsResults:
    """Suggest filter thresholds for the RNA-derived QC metrics, typically
    generated from :py:func:`~compute_rna_qc_metrics`.

    Args:
        metrics:
            RNA-derived QC metrics from :py:func:`~compute_rna_qc_metrics`.

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

    sums, detected, subset_proportions = lib.suggest_rna_qc_thresholds(
        (metrics.sum, metrics.detected, metrics.subset_proportion.as_list()),
        blockind,
        num_mads
    )

    if not blockind is None:
        sums = biocutils.NamedList(sums, blocklev)
        detected = biocutils.NamedList(detected, blocklev)
        for i, s in enumerate(subset_proportions):
            subset_proportions[i] = biocutils.NamedList(s, blocklev)

    subset_proportions = biocutils.NamedList(subset_proportions, metrics.subset_proportion.get_names())
    return SuggestRnaQcThresholdsResults(sums, detected, subset_proportions, blocklev)


def filter_rna_qc_metrics(
    thresholds: SuggestRnaQcThresholdsResults,
    metrics: ComputeRnaQcMetricsResults,
    block: Optional[Sequence] = None
) -> numpy.ndarray:  
    """Filter for high-quality cells based on RNA-derived QC metrics.

    Args:
        thresholds:
            Filter thresholds on the QC metrics, typically computed with
            :py:func:`~suggest_rna_qc_thresholds`.

        metrics:
            RNA-derived QC metrics, typically computed with
            :py:func:`~compute_rna_qc_metrics`.

        block:
            Factor specifying the block of origin (e.g., batch, sample) for
            each cell in ``metrics``. This should be the same as that used
            in :py:func:`~suggest_rna_qc_thresholds`.

    Returns:
        A NumPy vector of length equal to the number of cells in ``metrics``,
        containing truthy values for putative high-quality cells.
    """
    if not thresholds.block is None:
        if block is None:
            raise ValueError("'block' must be supplied if it was used in 'suggest_rna_qc_thresholds'")
        blockind = biocutils.match(block, thresholds.block, dtype=numpy.uint32, fail_missing=True)
        if (blockind < 0).any():
            raise ValueError("values in 'block' are not present in 'thresholds.block'")
        sums = numpy.array(thresholds.sum.as_list(), dtype=numpy.float64)
        detected = numpy.array(thresholds.detected.as_list(), dtype=numpy.float64)
        subset_proportion = [numpy.array(s.as_list(), dtype=numpy.float64) for s in thresholds.subset_proportion.as_list()]
    else:
        if not block is None:
            raise ValueError("'block' cannot be supplied if it was not used in 'suggest_rna_qc_thresholds'")
        blockind = None
        sums = thresholds.sum
        detected = thresholds.detected
        subset_proportion = numpy.array(thresholds.subset_proportion.as_list(), dtype=numpy.float64)

    return lib.filter_rna_qc_metrics(
        (sums, detected, subset_proportion),
        (metrics.sum, metrics.detected, metrics.subset_proportion.as_list()),
        blockind
    )
