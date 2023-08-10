from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..types import validate_object_type
from .log_norm_counts import LogNormalizeCountsArgs

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


@dataclass
class NormalizationStepArgs:
    """Arguments to run the normalization step.

    Attributes:
        log_norm (LogNormalizeCountsArgs): Arguments to compute log-normalized matrix
            :py:meth:`~scranpy.normalization.log_norm_counts.log_norm_counts`.
    """

    log_norm: LogNormalizeCountsArgs = LogNormalizeCountsArgs()

    def __post_init__(self):
        validate_object_type(self.log_norm, LogNormalizeCountsArgs)
