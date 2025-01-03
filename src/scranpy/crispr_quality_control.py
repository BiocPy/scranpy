from typing import Union, Sequence, Any, Optional
from dataclasses import dataclass

import numpy
import biocutils
import mattress

from . import lib_scranpy as lib


@dataclass
class ComputeCrisprQcMetricsResults:
    """Results of :py:func:`~compute_crispr_qc_metrics`."""

    sum: numpy.ndarray 
    """Floating-point array of length equal to the number of cells, containing the sum of counts across all guides for each cell."""

    detected: numpy.ndarray 
    """Integer array of length equal to the number of cells, containing the number of detected guides in each cell."""

    max_value: numpy.ndarray 
    """Floating-point array of length equal to the number of cells, containing the maximum count for each cell."""

    max_index: numpy.ndarray 
    """Integer array of length equal to the number of cells, containing the row index of the guide with the maximum count in each cell."""

    def to_biocframe(self):
        """Convert the results into a :py:class:`~biocframe.BiocFrame.BiocFrame`.

        Returns:
            A :py:class:`~biocframe.BiocFrame.BiocFrame` where each row corresponds to a cell and each column is one of the metrics.
        """
        colnames = ["sum", "detected", "max_value", "max_index"]
        contents = {}
        for n in colnames:
            contents[n] = getattr(self, n)
        import biocframe
        return biocframe.BiocFrame(contents, column_names=colnames)


def compute_crispr_qc_metrics(x: Any, num_threads: int = 1):
    """Compute quality control metrics from CRISPR count data.

    Args: 
        x:
            A matrix-like object containing CRISPR counts.

        num_threads:
            Number of threads to use.

    Returns:
        QC metrics computed from the count matrix for each cell.

    References:
        The ``compute_crispr_qc_metrics`` function in the `scran_qc <https://github.com/libscran/scran_qc>`_ C++ library, which describes the rationale behind these QC metrics.
    """
    ptr = mattress.initialize(x)
    osum, odetected, omaxval, omaxind = lib.compute_crispr_qc_metrics(ptr.ptr, num_threads)
    return ComputeCrisprQcMetricsResults(osum, odetected, omaxval, omaxind)


@dataclass
class SuggestCrisprQcThresholdsResults:
    """Results of :py:func:`~suggest_crispr_qc_thresholds`."""

    max_value: Union[biocutils.NamedList, float]
    """Threshold on the maximum count in each cell.
    Cells with lower maxima are considered to be of low quality.

    If ``block`` is provided in :py:func:`~suggest_crispr_qc_thresholds`, a list is returned containing a separate threshold for each level of the factor.
    Otherwise, a single float is returned containing the threshold for all cells."""

    block: Optional[list]
    """Levels of the blocking factor.
    Each entry corresponds to a element of :py:attr:`~max_value` if ``block`` was provided in :py:func:`~suggest_crispr_qc_thresholds`.
    This is set to ``None`` if no blocking was performed."""


def suggest_crispr_qc_thresholds(
    metrics: ComputeCrisprQcMetricsResults,
    block: Optional[Sequence] = None,
    num_mads: float = 3.0,
) -> SuggestCrisprQcThresholdsResults:
    """Suggest filter thresholds for the CRISPR-derived QC metrics, typically generated from :py:func:`~compute_crispr_qc_metrics`.

    Args:
        metrics:
            CRISPR-derived QC metrics from :py:func:`~compute_crispr_qc_metrics`.

        block:
            Blocking factor specifying the block of origin (e.g., batch, sample) for each cell in ``metrics``.
            If supplied, a separate threshold is computed from the cells in each block.
            Alternatively ``None``, if all cells are from the same block.

        num_mads:
            Number of MADs from the median to define the threshold for outliers in each QC metric.

    Returns:
        Suggested filters on the relevant QC metrics.

    References:
        The ``compute_crispr_qc_filters`` and ``compute_crispr_qc_filters_blocked`` functions in the `scran_qc <https://github.com/libscran/scran_qc>`_ C++ library, which describes the rationale behind the suggested filters.
    """
    if block is not None:
        blocklev, blockind = biocutils.factorize(block, sort_levels=True, dtype=numpy.uint32, fail_missing=True)
    else:
        blocklev = None
        blockind = None

    max_value, = lib.suggest_crispr_qc_thresholds(
        (metrics.sum, metrics.detected, metrics.max_value, metrics.max_index),
        blockind,
        num_mads
    )

    if blockind is not None:
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
            Filter thresholds on the QC metrics, typically computed with :py:func:`~suggest_crispr_qc_thresholds`.

        metrics:
            CRISPR-derived QC metrics, typically computed with :py:func:`~compute_crispr_qc_metrics`.

        block:
            Blocking factor specifying the block of origin (e.g., batch, sample) for each cell in ``metrics``.
            The levels should be a subset of those used in :py:func:`~suggest_crispr_qc_thresholds`.

    Returns:
        A NumPy vector of length equal to the number of cells in ``metrics``, containing truthy values for putative high-quality cells.
    """
    if thresholds.block is not None:
        if block is None:
            raise ValueError("'block' must be supplied if it was used in 'suggest_crispr_qc_thresholds'")
        blockind = biocutils.match(block, thresholds.block, dtype=numpy.uint32, fail_missing=True)
        max_value = numpy.array(thresholds.max_value.as_list(), dtype=numpy.float64)
    else:
        if block is not None:
            raise ValueError("'block' cannot be supplied if it was not used in 'suggest_crispr_qc_thresholds'")
        blockind = None
        max_value = thresholds.max_value

    return lib.filter_crispr_qc_metrics(
        (max_value,),
        (metrics.sum, metrics.detected, metrics.max_value, metrics.max_index),
        blockind
    )
