import numpy as np

from .. import cpphelpers as lib


def choose_hvgs(stat: np.ndarray, number: int = 2500) -> np.ndarray:
    """Choose highly variable genes.

    Args:
        stat (np.ndarray): Array of variance modelling statistics,
            where larger values correspond to higher variability.
            This usually contains the residuals of the fitted
            mean-variance trend.
        number (int): Number of HVGs to pick. Defaults to 2500.

    Return:
        np.ndarray: Array of booleans of length equal to `stat`,
        specifying whether a given gene is considered as highly variable.
    """

    output = np.zeros(len(stat), dtype=np.uint8)
    stat_internal = stat.astype(np.float64, copy=False)

    lib.choose_hvgs(
        len(stat_internal), stat_internal.ctypes.data, number, output.ctypes.data
    )

    return output.astype(np.bool_, copy=False)
