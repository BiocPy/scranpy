from dataclasses import dataclass

import numpy as np

from .. import cpphelpers as lib

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


@dataclass
class ChooseHvgsOptions:
    """Optional arguments for
    :py:meth:`~scranpy.feature_selection.choose_hvgs.choose_hvgs`.

    Attributes:
        number (int): 
            Number of HVGs to retain.
            Larger values preserve more biological structure at the cost of increasing computational work and random noise from less-variable genes.
            Defaults to 2500.

        verbose (bool): display logs? Defaults to False.
    """

    number: int = 2500
    verbose: bool = False


def choose_hvgs(
    stat: np.ndarray, options: ChooseHvgsOptions = ChooseHvgsOptions()
) -> np.ndarray:
    """Choose highly variable genes for high-dimensional downstream steps
    such as :py:meth:`~scranpy.dimensionality_reduction.run_pca.run_pca`.
    This ensures that those steps focus on interesting biology,
    under the assumption that biological variation is larger than random noise.

    Args:
        stat (np.ndarray): Array of variance modelling statistics,
            where larger values correspond to higher variability.
            This usually contains the residuals of the fitted
            mean-variance trend from
            :py:meth:`~scranpy.feature_selection.model_gene_variances.model_gene_variances`.

        options (ChooseHvgsOptions): Optional parameters.

    Return:
        np.ndarray: Array of booleans of length equal to ``stat``,
        specifying whether a given gene is considered to be highly variable.
    """

    output = np.zeros(len(stat), dtype=np.uint8)
    stat_internal = stat.astype(np.float64, copy=False)

    lib.choose_hvgs(len(stat_internal), stat_internal, options.number, output)

    return output.astype(np.bool_, copy=False)
