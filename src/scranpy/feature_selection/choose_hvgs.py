from dataclasses import dataclass

import numpy as np

from .. import cpphelpers as lib

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


@dataclass
class ChooseHvgsOptions:
    """Arguments to choose highly variable genes -
    :py:meth:`~scranpy.feature_selection.choose_hvgs.choose_hvgs`.

    Attributes:
        number (int): Number of HVGs to pick. Defaults to 2500.
        verbose (bool): display logs? Defaults to False.
    """

    number: int = 4000
    verbose: bool = False


def choose_hvgs(
    stat: np.ndarray, options: ChooseHvgsOptions = ChooseHvgsOptions()
) -> np.ndarray:
    """Choose highly variable genes.

    Args:
        stat (np.ndarray): Array of variance modelling statistics,
            where larger values correspond to higher variability.
            This usually contains the residuals of the fitted
            mean-variance trend from
            (:py:meth:`~scranpy.feature_selection.model_gene_variances.model_gene_variances`).
        options (ChooseHvgsOptions): Optional parameters.

    Return:
        np.ndarray: Array of booleans of length equal to ``stat``,
        specifying whether a given gene is considered as highly variable.
    """

    output = np.zeros(len(stat), dtype=np.uint8)
    stat_internal = stat.astype(np.float64, copy=False)

    lib.choose_hvgs(len(stat_internal), stat_internal, options.number, output)

    return output.astype(np.bool_, copy=False)
