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

    sum: numpy.ndarray 
    """Floating-point array of length equal to the number of cells, containing the sum of counts across all genes for each cell."""

    detected: numpy.ndarray 
    """Integer array of length equal to the number of cells, containing the number of detected genes in each cell."""

    subset_proportion: biocutils.NamedList
    """Proportion of counts in each gene subset in each cell.
    Each list element corresponds to a gene subset and is a NumPy array of length equal to the number of cells.
    Each entry of the array contains the proportion of counts in that subset in each cell."""

    def to_biocframe(self, flatten: bool = True): 
        """Convert the results into a :py:class:`~biocframe.BiocFrame.BiocFrame`.

        Args:
            flatten:
                Whether to flatten the subset proportions into separate columns.
                If ``True``, each entry of :py:attr:`~subset_proportion` is represented by a ``subset_proportion_<NAME>`` column,
                where ``<NAME>`` is the the name of each entry (if available) or its index (otherwise).
                If ``False``, :py:attr:`~subset_proportion` is represented by a nested :py:class:`~biocframe.BiocFrame.BiocFrame`.

        Returns:
            A :py:class:`~biocframe.BiocFrame.BiocFrame` where each row corresponds to a cell and each column is one of the metrics.
        """
        colnames = ["sum", "detected"]
        contents = {}
        for n in colnames:
            contents[n] = getattr(self, n)

        subnames = self.subset_proportion.get_names()
        if subnames is not None:
            subnames = subnames.as_list()
        else:
            subnames = [str(i) for i in range(len(self.subset_proportion))]

        import biocframe
        if flatten:
            for i, n in enumerate(subnames):
                nn = "subset_proportion_" + n
                colnames.append(nn)
                contents[nn] = self.subset_proportion[i]
        else:
            subcontents = {}
            for i, n in enumerate(subnames):
                subcontents[n] = self.subset_proportion[i]
            colnames.append("subset_proportion")
            contents["subset_proportion"] = biocframe.BiocFrame(subcontents, column_names=subnames, number_of_rows=len(self.sum))

        return biocframe.BiocFrame(contents, column_names=colnames)



def compute_rna_qc_metrics(x: Any, subsets: Union[Mapping, Sequence], num_threads: int = 1):
    """Compute quality control metrics from RNA count data.

    Args: 
        x:
            A matrix-like object containing RNA counts.

        subsets:
            Subsets of genes corresponding to "control" features like mitochondrial genes.
            This may be either:

            - A list of arrays.
              Each array corresponds to an gene subset and can either contain boolean or integer values.
              For booleans, the array should be of length equal to the number of rows, and values should be truthy for rows that belong in the subset.
              For integers, each element of the array is treated the row index of an gene in the subset.
            - A dictionary where keys are the names of each gene subset and the values are arrays as described above.
            - A :py:class:`~biocutils.NamedList.NamedList` where each element is an array as described above, possibly with names.

        num_threads:
            Number of threads to use.

    Returns:
        QC metrics computed from the count matrix for each cell.

    References:
        The ``compute_rna_qc_metrics`` function in the `scran_qc <https://github.com/libscran/scran_qc>`_ C++ library, which describes the rationale behind these QC metrics.
    """
    ptr = mattress.initialize(x)
    subkeys, subvals = _sanitize_subsets(subsets, x.shape[0])
    osum, odetected, osubset_proportion = lib.compute_rna_qc_metrics(ptr.ptr, subvals, num_threads)
    osubset_proportion = biocutils.NamedList(osubset_proportion, subkeys)
    return ComputeRnaQcMetricsResults(osum, odetected, osubset_proportion)


@dataclass
class SuggestRnaQcThresholdsResults:
    """Results of :py:func:`~suggest_rna_qc_thresholds`."""

    sum: Union[biocutils.NamedList, float]
    """Threshold on the sum of counts in each cell.
    Cells with lower totals are considered to be of low quality. 

    If ``block`` is provided in :py:func:`~suggest_rna_qc_thresholds`, a list is returned containing a separate threshold for each level of the factor.
    Otherwise, a single float is returned containing the threshold for all cells."""

    detected: Union[biocutils.NamedList, float]
    """Threshold on the number of detected genes.
    Cells with lower numbers of detected genes are considered to be of low quality.

    If ``block`` is provided in :py:func:`~suggest_rna_qc_thresholds`, a list is returned containing a separate threshold for each level of the factor.
    Otherwise, a single float is returned containing the threshold for all cells."""

    subset_proportion: biocutils.NamedList
    """Thresholds on the sum of counts in each gene subset.
    Each element of the list corresponds to a gene subset. 
    Cells with higher sums than the threshold for any subset are considered to be of low quality. 

    If ``block`` is provided in :py:func:`~suggest_rna_qc_thresholds`, each entry of the returned list is another :py:class:`~biocutils.NamedList.NamedList`  containing a separate threshold for each level.
    Otherwise, each entry of the list is a single float containing the threshold for all cells."""

    block: Optional[list]
    """Levels of the blocking factor.
    Each entry corresponds to a element of :py:attr:`~sum`, :py:attr:`~detected`, etc., if ``block`` was provided in :py:func:`~suggest_rna_qc_thresholds`.
    This is set to ``None`` if no blocking was performed."""


def suggest_rna_qc_thresholds(
    metrics: ComputeRnaQcMetricsResults,
    block: Optional[Sequence] = None,
    num_mads: float = 3.0,
) -> SuggestRnaQcThresholdsResults:
    """Suggest filter thresholds for the RNA-derived QC metrics, typically generated from :py:func:`~compute_rna_qc_metrics`.

    Args:
        metrics:
            RNA-derived QC metrics from :py:func:`~compute_rna_qc_metrics`.

        block:
            Blocking factor specifying the block of origin (e.g., batch, sample) for each cell in ``metrics``.
            If supplied, a separate threshold is computed from the cells in each block.
            Alternatively ``None``, if all cells are from the same block.

        num_mads:
            Number of MADs from the median to define the threshold for outliers in each QC metric.

    Returns:
        Suggested filters on the relevant QC metrics.

    References:
        The ``compute_rna_qc_filters`` and ``compute_rna_qc_filters_blocked`` functions in the `scran_qc <https://github.com/libscran/scran_qc>`_ C++ library, which describes the rationale behind the suggested filters.
    """
    if block is not None:
        blocklev, blockind = biocutils.factorize(block, sort_levels=True, dtype=numpy.uint32, fail_missing=True)
    else:
        blocklev = None
        blockind = None

    sums, detected, subset_proportions = lib.suggest_rna_qc_thresholds(
        (metrics.sum, metrics.detected, metrics.subset_proportion.as_list()),
        blockind,
        num_mads
    )

    if blockind is not None:
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
            Filter thresholds on the QC metrics, typically computed with :py:func:`~suggest_rna_qc_thresholds`.

        metrics:
            RNA-derived QC metrics, typically computed with :py:func:`~compute_rna_qc_metrics`.

        block:
            Blocking factor specifying the block of origin (e.g., batch, sample) for each cell in ``metrics``.
            The levels should be a subset of those used in :py:func:`~suggest_rna_qc_thresholds`.

    Returns:
        A NumPy vector of length equal to the number of cells in ``metrics``, containing truthy values for putative high-quality cells.
    """
    if thresholds.block is not None:
        if block is None:
            raise ValueError("'block' must be supplied if it was used in 'suggest_rna_qc_thresholds'")
        blockind = biocutils.match(block, thresholds.block, dtype=numpy.uint32, fail_missing=True)
        sums = numpy.array(thresholds.sum.as_list(), dtype=numpy.float64)
        detected = numpy.array(thresholds.detected.as_list(), dtype=numpy.float64)
        subset_proportion = [numpy.array(s.as_list(), dtype=numpy.float64) for s in thresholds.subset_proportion.as_list()]
    else:
        if block is not None:
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
