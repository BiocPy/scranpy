from dataclasses import dataclass

from numpy import bool_, float64, ndarray, uint8, zeros

from .. import _cpphelpers as lib

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


@dataclass
class ChooseHvgsOptions:
    """Optional arguments for :py:meth:`~scranpy.feature_selection.choose_hvgs.choose_hvgs`.

    Attributes:
        number:
            Number of HVGs to retain.
            Larger values preserve more biological structure at the cost of increasing
            computational work and random noise from less-variable genes.

            Defaults to 2500.
    """

    number: int = 2500


def choose_hvgs(
    stat: ndarray, options: ChooseHvgsOptions = ChooseHvgsOptions()
) -> ndarray:
    """Choose highly variable genes for high-dimensional downstream steps such as
    :py:meth:`~scranpy.dimensionality_reduction.run_pca.run_pca`. This ensures that those steps focus on interesting
    biology, under the assumption that biological variation is larger than random noise.

    Args:
        stat:
            Array of variance modelling statistics,
            where larger values correspond to higher variability.
            This usually contains the residuals of the fitted
            mean-variance trend from
            :py:meth:`~scranpy.feature_selection.model_gene_variances.model_gene_variances`.

        options:
            Optional parameters.

    Return:
        Array of booleans of length equal to ``stat``, specifying whether a
        given gene is considered to be highly variable.
    """

    output = zeros(len(stat), dtype=uint8)
    stat_internal = stat.astype(float64, copy=False)

    lib.choose_hvgs(len(stat_internal), stat_internal, options.number, output)

    return output.astype(bool_, copy=False)
