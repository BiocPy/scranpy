from typing import Optional, Any, Sequence, Tuple, Union
from dataclasses import dataclass

import numpy

from . import lib_scranpy as lib


@dataclass
class GroupwiseSummarizedEffects:
    """Summarized effect sizes for a single group, typically created by :py:func:`~summarize_effects` or :py:func:`~scranpy.score_markers.score_markers`."""

    min: numpy.ndarray
    """Floating-point array of length equal to the number of genes.
    Each entry is the minimum effect size for that gene from all pairwise comparisons to other groups."""

    mean: numpy.ndarray
    """Floating-point array of length equal to the number of genes.
    Each entry is the mean effect size for that gene from all pairwise comparisons to other groups."""

    median: numpy.ndarray
    """Floating-point array of length equal to the number of genes.
    Each entry is the median effect size for that gene from all pairwise comparisons to other groups."""

    max: numpy.ndarray
    """Floating-point array of length equal to the number of genes.
    Each entry is the maximum effect size for that gene from all pairwise comparisons to other groups."""

    min_rank: numpy.ndarray
    """Floating-point array of length equal to the number of genes.
    Each entry is the minimum rank of the gene from all pairwise comparisons to other groups."""

    def to_biocframe(self):
        """Convert the results to a :py:class:`~biocframe.BiocFrame.BiocFrame`.

        Returns:
            A :py:class:`~biocframe.BiocFrame.BiocFrame` where each row is a gene and each column is a summary statistic.
        """
        cols = ["min", "mean", "median", "max", "min_rank"]
        contents = {}
        for n in cols:
            contents[n] = getattr(self, n)
        import biocframe
        return biocframe.BiocFrame(contents, column_names=cols)


def summarize_effects(effects: numpy.ndarray, num_threads: int = 1) -> list[GroupwiseSummarizedEffects]: 
    """For each group, summarize the effect sizes for all pairwise comparisons
    to other groups. This yields a set of summary statistics that can be used
    to rank marker genes for each group.

    Args:
        effects:
            A 3-dimensional numeric containing the effect sizes from each pairwise comparison between groups.
            The extents of the first two dimensions should be equal to the number of groups, while the extent of the final dimension is equal to the number of genes. 
            The entry ``[i, j, k]`` should represent the effect size from the comparison of group ``j`` against group ``i`` for gene ``k``.
            See also the output of :py:func:`~scranpy.score_markers.score_markers` with ``all_pairwise = True``.

        num_threads:
            Number of threads to use.

    Returns:
        List of length equal to the number of groups (i.e., the extents of the first two dimensions of ``effects``).
        Each entry contains the summary statistics of the effect sizes of the comparisons involving the corresponding group.
    """
    results = lib.summarize_effects(effects, num_threads)
    output = []
    for val in results:
        output.append(GroupwiseSummarizedEffects(*val))
    return output
