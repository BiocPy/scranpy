from typing import Optional, Any, Sequence, Tuple, Union, Literal
from dataclasses import dataclass

import numpy
import mattress
import biocutils

from . import lib_scranpy as lib
from .summarize_effects import GroupwiseSummarizedEffects


@dataclass
class ScoreMarkersResults:
    """Results of :py:func:`~score_markers`."""

    groups: list
    """Identities of the groups."""

    mean: numpy.ndarray
    """Floating-point matrix containing the mean expression for each gene in each group.
    Each row is a gene and each column is a group, ordered as in :py:attr:`~groups`."""

    detected: numpy.ndarray
    """Floating-point matrix containing the proportion of cells with detected expression for each gene in each group.
    Each row is a gene and each column is a group, ordered as in :py:attr:`~groups`."""

    cohens_d: Optional[Union[numpy.ndarray, biocutils.NamedList]]
    """If ``all_pairwise = False``, this is a named list of :py:class:`~scranpy.summarize_effects.GroupwiseSummarizedEffects` objects.
    Each object corresponds to a group in the same order as :py:attr:`~groups`, and contains a summary of Cohen's d from pairwise comparisons to all other groups.
    This includes the min, mean, median, max and min-rank.

    If ``all_pairwise = True``, this is a 3-dimensional numeric array containing the Cohen's d from each pairwise comparison between groups.
    The extents of the first two dimensions are equal to the number of groups, while the extent of the final dimension is equal to the number of genes.
    The entry ``[i, j, k]`` represents Cohen's d from the comparison of group ``j`` over group ``i`` for gene ``k``.

    If ``compute_cohens_d = False``, this is ``None``."""

    auc: Optional[Union[numpy.ndarray, biocutils.NamedList]]
    """Same as :py:attr:`~cohens_d` but for the AUCs.
    If ``compute_auc = False``, this is ``None``."""

    delta_mean: Optional[Union[numpy.ndarray, biocutils.NamedList]]
    """Same as :py:attr:`~cohens_d` but for the delta-mean.
    If ``compute_delta_mean = False``, this is ``None``."""

    delta_detected: Optional[Union[numpy.ndarray, biocutils.NamedList]]
    """Same as :py:attr:`~cohens_d` but for the delta-detected.
    If ``compute_delta_detected = False``, this is ``None``."""

    def to_biocframes(self,
        effect_sizes: Optional[list] = None,
        summaries: Optional[list] = None,
        include_mean: bool = True,
        include_detected: bool = True
    ) -> biocutils.NamedList:
        """Convert the effect size summaries into a :py:class:`~biocframe.BiocFrame.BiocFrame` for each group.
        This should only be used if ``all_pairwise = False`` in :py:func:`~score_markers`.

        Args:
            effect_sizes:
                List of effect sizes to include in each :py:class:`~biocframe.BiocFrame.BiocFrame`.
                This can contain any of ``cohens_d``, ``auc``, ``delta_mean``, and ``delta_detected``.
                If ``None``, all non-``None`` effect sizes are reported.

            summaries:
                List of summary statistics to include in each :py:class:`~biocframe.BiocFrame.BiocFrame`.
                This can contain any of ``min``, ``mean``, ``median``, ``max``, and ``min_rank``.
                If ``None``, all summary statistics are reported.

            include_mean:
                Whether to include the mean for each group.

            include_detected:
                Whether to include the detected proportion for each group.

        Returns:
            A list of length equal to :py:attr:`~groups`, containing a :py:class:`~biocframe.BiocFrame.BiocFrame` with the effect size summaries for each group.
            Each row of the :py:class:`~biocframe.BiocFrame.BiocFrame` corresponds toa  gene.
            Each effect size summary is represented by a column named ``<EFFECT>_<SUMMARY>``.
            If ``include_mean = True`` or ``include_detected = True``, additional columns will be present with the mean and detected proportion, respectively.

            The list itself is named according to :py:attr:`~groups` if the elements can be converted to strings, otherwise it is unnamed.
        """
        if effect_sizes is None:
            effect_sizes = ["cohens_d", "auc", "delta_mean", "delta_detected"]
        if summaries is None:
            summaries = ["min", "mean", "median", "max", "min_rank"]

        import biocframe
        collected = []
        for g in range(len(self.groups)):
            current = biocframe.BiocFrame({}, number_of_rows=self.mean.shape[0])
            if include_mean:
                current.set_column("mean", self.mean[:,g], in_place=True)
            if include_detected:
                current.set_column("detected", self.detected[:,g], in_place=True)
            for eff in effect_sizes:
                eff_all = getattr(self, eff)
                if eff_all is None:
                    continue
                effdf = eff_all[g].to_biocframe()
                for summ in summaries:
                    current.set_column(eff + "_" + summ, effdf.get_column(summ), in_place=True)
            collected.append(current)

        group_names = None
        try:
            group_names = biocutils.Names(self.groups)
        except:
            pass
        return biocutils.NamedList(collected, group_names)


def score_markers(
    x: Any, 
    groups: Sequence, 
    block: Optional[Sequence] = None, 
    block_weight_policy: Literal["variable", "equal", "none"] = "variable",
    variable_block_weight: Tuple = (0, 1000),
    compute_delta_mean: bool = True,
    compute_delta_detected: bool = True,
    compute_cohens_d: bool = True,
    compute_auc: bool = True,
    threshold: float = 0, 
    all_pairwise: bool = False, 
    num_threads: int = 1
) -> ScoreMarkersResults:
    """Score marker genes for each group using a variety of effect sizes from pairwise comparisons between groups.
    This includes Cohen's d, the area under the curve (AUC), the difference in the means (delta-mean) and the difference in the proportion of detected cells (delta-detected).

    Args:
        x:
            A matrix-like object where rows correspond to genes or genomic features and columns correspond to cells. 
            It is typically expected to contain log-expression values, e.g., from :py:func:`~scranpy.normalize_counts.normalize_counts`.

        groups: 
            Group assignment for each cell in ``x``.
            This should have length equal to the number of columns in ``x``.

        block:
            Array of length equal to the number of columns of ``x``, containing the block of origin (e.g., batch, sample) for each cell.
            Alternatively ``None``, if all cells are from the same block.

        block_weight_policy:
            Policy to use for weighting different blocks when computing the average for each statistic.
            Only used if ``block`` is provided.

        variable_block_weight:
            Parameters for variable block weighting.
            This should be a tuple of length 2 where the first and second values are used as the lower and upper bounds, respectively, for the variable weight calculation.
            Only used if ``block`` is provided and ``block_weight_policy = "variable"``.

        compute_delta_mean:
            Whether to compute the delta-means, i.e., the log-fold change when ``x`` contains log-expression values.

        compute_delta_detected:
            Whether to compute the delta-detected, i.e., differences in the proportion of cells with detected expression.

        cohens_d:
            Whether to compute Cohen's d.

        compute_auc:
            Whether to compute the AUC.
            Setting this to ``False`` can improve speed and memory efficiency.

        threshold:
            Non-negative value specifying the minimum threshold on the differences in means (i.e., the log-fold change, if ``x`` contains log-expression values).
            This is incorporated into the calculation for Cohen's d and the AUC.

        all_pairwise: 
            Whether to report the full effects for every pairwise comparison between groups.
            If ``False``, only summaries are reported.

        num_threads:
            Number of threads to use.

    Returns:
        Scores for ranking marker genes in each group, based on the effect sizes for pairwise comparisons between groups.

    References:
        The ``score_markers_summary`` and ``score_markers_pairwise`` functions in the `scran_markers <https://github.com/libscran/scran_markers>`_ C++ library,
        which describes the rationale behind the choice of effect sizes and summary statistics.
    """
    ptr = mattress.initialize(x)
    glev, gind = biocutils.factorize(groups, sort_levels=True, fail_missing=True, dtype=numpy.uint32)

    if block is not None:
        _, block = biocutils.factorize(block, fail_missing=True, dtype=numpy.uint32)

    args = [
        ptr.ptr,
        gind,
        len(glev),
        block,
        block_weight_policy,
        variable_block_weight,
        threshold,
        num_threads,
        compute_cohens_d,
        compute_auc,
        compute_delta_mean,
        compute_delta_detected
    ]

    if all_pairwise:
        means, detected, cohen, auc, delta_mean, delta_detected = lib.score_markers_pairwise(*args)
        def san(y):
            return y
    else:
        means, detected, cohen, auc, delta_mean, delta_detected = lib.score_markers_summary(*args)
        def san(y):
            out = []
            for i, vals in enumerate(y):
                out.append(GroupwiseSummarizedEffects(*vals))
            return biocutils.NamedList(out, glev)

    return ScoreMarkersResults( 
        glev,
        means,
        detected,
        san(cohen) if compute_cohens_d else None,
        san(auc) if compute_auc else None,
        san(cohen) if compute_delta_mean else None,
        san(cohen) if compute_delta_detected else None
    )
