from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

from numpy import float64, ndarray, zeros

from .. import _cpphelpers as lib
from .._utils import MatrixTypes, factorize, tatamize_input, to_logical


@dataclass
class ScoreFeatureSetOptions:
    """Options to pass to :py:meth:`~scranpy.feature_set_enrichment.score_feature_set.score_feature_set`.

    Attributes:
        block:
            Block assignment for each cell.
            Thresholds are computed within each block to avoid inflated variances from
            inter-block differences.

            If provided, this should have length equal to the number of cells, where
            cells have the same value if and only if they are in the same block.
            Defaults to None, indicating all cells are part of the same block.

        scale:
            Whether to scale the features to unit variance before
            computing the scores.

        assay_type:
            Assay to use from ``input`` if it is a
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.

        num_threads:
            Number of threads to use.
    """

    block: Optional[Sequence] = None
    scale: bool = False
    assay_type: Union[int, str] = "logcounts"
    num_threads: int = 1


def score_feature_set(
    input: MatrixTypes,
    subset: Sequence,
    options=ScoreFeatureSetOptions(),
) -> Tuple[ndarray, ndarray]:
    """Compute a score for the activity of a feature set in each cell. This is
    done using a slightly modified version of the
    `GSDecon <https://github.com/JasonHackney/GSDecon>`_ algorithm,
    where we perform a PCA to obtain the rank-1 reconstruction of the feature
    set's expression values across all cells; the mean of the reconstructed
    values serves as the score per cell, while the rotation vector is reported
    as the weights on the features involved.

    Args:
        input:
            Matrix-like object containing cells in columns and features in
            rows, typically with log-normalized expression data.  This should
            be a matrix class that can be converted into a
            :py:class:`~mattress.TatamiNumericPointer`.  Developers may also
            provide the :py:class:`~mattress.TatamiNumericPointer` itself.

        subset:
            Array of integer indices, specifying the rows of `input` belonging
            to the features subset. Alternatively, an array of length
            equal to the number of rows in ``input``, containing booleans
            specifying that the corresponding row belongs to the subset.

        options:
            Optional parameters.

    Returns:
        Tuple where the first array is of length equal to the
        number of columns of ``input`` and contains the feature set score for
        each cell. The second array is of length equal to the number of
        features in ``subset`` and contains the weight for each feature.
    """

    x = tatamize_input(input, options.assay_type)
    subset = to_logical(subset, x.nrow())
    NC = x.ncol()

    use_block = options.block is not None
    block_offset = 0

    if use_block:
        if len(options.block) != NC:
            raise ValueError(
                "number of columns in 'x' should equal the length of 'block'"
            )
        block_levels, block_indices = factorize(options.block)
        block_offset = block_indices.ctypes.data

    output_scores = zeros(NC, dtype=float64)
    nfeatures = subset.sum()
    output_weights = zeros(nfeatures, dtype=float64)

    lib.score_feature_set(
        x.ptr,
        subset,
        use_block,
        block_offset,
        output_scores,
        output_weights,
        options.scale,
        options.num_threads,
    )

    return output_scores, output_weights
