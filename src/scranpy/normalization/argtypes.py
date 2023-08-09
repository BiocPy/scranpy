from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..types import validate_object_type

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


@dataclass
class LogNormalizeCountsArgs:
    """Arguments to log-normalize counts.

    Attributes:
        block (Sequence, optional): Block assignment for each cell.
            This is used to segregate cells in order to perform comparisons within
            each block. Defaults to None, indicating all cells are part of the same
            block.
        size_factors (Optional[np.ndarray], optional): Size factors for each cell.
            Defaults to None.
        center (bool, optional): Center the size factors?. Defaults to True.
        allow_zeros (bool, optional): Allow zeros?. Defaults to False.
        allow_non_finite (bool, optional): Allow `nan` or `inifnite` numbers?.
            Defaults to False.
        num_threads (int, optional): Number of threads. Defaults to 1.
        verbose (bool, optional): Display logs?. Defaults to False.
    """

    block: Optional[Sequence] = None
    size_factors: Optional[np.ndarray] = None
    center: bool = True
    allow_zeros: bool = False
    allow_non_finite: bool = False
    num_threads: int = 1
    verbose: bool = False


@dataclass
class NormalizationStepArgs:
    """Arguments to run the normalization step.

    Attributes:
        log_norm (LogNormalizeCountsArgs): Arguments to score markers.
    """

    log_norm: LogNormalizeCountsArgs = LogNormalizeCountsArgs()

    def __post_init__(self):
        validate_object_type(self.log_norm, LogNormalizeCountsArgs)
