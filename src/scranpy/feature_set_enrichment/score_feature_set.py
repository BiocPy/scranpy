from dataclasses import dataclass
from typing import Sequence, Union
from numpy import array, ndarray, int32, float64

from .. import cpphelpers as lib
from ..types import MatrixTypes
from ..utils import to_logical, validate_and_tatamize_input


@dataclass
class ScoreFeatureSetOptions:
    """Options to pass to
    :py:meth:`~scranpy.feature_set_enrichment.score_feature_set.score_feature_set`.

    Attributes:
        block (Sequence, optional):
            Block assignment for each cell.
            Thresholds are computed within each block to avoid inflated variances from
            inter-block differences.

            If provided, this should have length equal to the number of cells, where
            cells have the same value if and only if they are in the same block.
            Defaults to None, indicating all cells are part of the same block.

        scale (bool): Whether to scale the features to unit variance before 
            computing the scores.

        num_threads (int): Number of threads to use.
    """
    block: Optional[Sequence] = None
    scale: bool = False
    num_threads: int = 1


def score_feature_set(
    input: MatrixTypes,
    subset: Sequence,
    options = ScoreFeatureSetOptions(),
) -> Tuple[ndarray, ndarray]:
    """
    Compute a score for the activity of a feature set in each cell.
    This is done using a slightly modified version of the GSDecon algorithm,
    where we perform a PCA to obtain the rank-1 reconstruction of the
    feature set's expression values across all cells; the mean of the 
    reconstructed values serves as the score per cell, while the rotation
    vector is reported as the weights on the features involved.

    Args:
        input: 
            Matrix-like object containing cells in columns and features in
            rows, typically with log-normalized expression data.  This should
            be a matrix class that can be converted into a
            :py:class:`~mattress.TatamiNumericPointer`.  Developers may also
            provide the :py:class:`~mattress.TatamiNumericPointer` itself.

        subset (Sequence):
            Array of integer indices, specifying the rows of `input` belonging
            to the features subset. Alternatively, an array of length
            equal to the number of rows in ``input``, containing booleans
            specifying that the corresponding row belongs to the subset.

        options (ScoreFeatureSetOptions):
            Further options.

    Returns:
        Tuple[ndarray, ndarray]: The first array is of length equal to the
        number of columns of ``input`` and contains the feature set score for
        each cell. The second array is of length equal to the number of
        features in ``subset`` and contains the weight for each feature.
    """

    x = validate_and_tatamize_input(input)
    subset = to_logical(subset, x.nrow())
    NC = x.ncol()

    use_block = options.block is not None
    block_info = None
    block_offset = 0

    if use_block:
        if len(options.block) != NC:
            raise ValueError(
                "number of columns in 'x' should equal the length of 'block'"
            )
        block_info = factorize(options.block)
        block_offset = block_info.indices.ctypes.data

    output_scores = ndarray(NC, dtype=float64)
    nfeatures = subset.sum()
    output_weights = ndarray(nfeatures, dtype=float64)

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
