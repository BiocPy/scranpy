import numpy as np

from .. import cpphelpers as lib
from .argtypes import ChooseHvgArgs

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


def choose_hvgs(
    stat: np.ndarray, options: ChooseHvgArgs = ChooseHvgArgs()
) -> np.ndarray:
    """Choose highly variable genes.

    Args:
        stat (np.ndarray): Array of variance modelling statistics,
            where larger values correspond to higher variability.
            This usually contains the residuals of the fitted
            mean-variance trend.
        options (ChooseHvgArgs): additional arguments defined by `ChooseHvgArgs`.

    Return:
        np.ndarray: Array of booleans of length equal to `stat`,
        specifying whether a given gene is considered as highly variable.
    """

    output = np.zeros(len(stat), dtype=np.uint8)
    stat_internal = stat.astype(np.float64, copy=False)

    lib.choose_hvgs(len(stat_internal), stat_internal, options.number, output)

    return output.astype(np.bool_, copy=False)
