from dataclasses import dataclass, field
from typing import Optional, Sequence

from mattress import TatamiNumericPointer

from .._abstract import AbstractStepOptions
from ..types import validate_object_type
from .log_norm_counts import LogNormalizeCountsOptions

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


@dataclass
class NormalizationStepOptions(AbstractStepOptions):
    """Arguments to run the normalization step.

    Attributes:
        log_norm (LogNormalizeCountsOptions): Arguments to compute log-normalized matrix
            :py:meth:`~scranpy.normalization.log_norm_counts.log_norm_counts`.
    """

    log_normalize_counts: LogNormalizeCountsOptions = field(
        default_factory=LogNormalizeCountsOptions
    )

    def __post_init__(self):
        validate_object_type(self.log_normalize_counts, LogNormalizeCountsOptions)

    def set_threads(self, num_threads: int = 1):
        """Set number of threads to use.

        Args:
            num_threads (int, optional): Number of threads. Defaults to 1.
        """
        self.log_normalize_counts.num_threads = num_threads

    def set_verbose(self, verbose: bool = False):
        """Set verbose to display logs.

        Args:
            verbose (bool, optional): Display logs? Defaults to False.
        """
        self.log_normalize_counts.verbose = verbose

    def set_block(self, block: Optional[Sequence] = None):
        """Set block.

        Args:
            block (Sequence, optional): Blocks assignments
                for each cell. Defaults to None.
        """
        self.log_normalize_counts.block = block


@dataclass
class NormalizationStepResults:
    """Results of the normalization step.

    Attributes:
        log_normalized_counts (TatamiNumericPointer, optional): Log-normalized matrix
            :py:meth:`~scranpy.normalization.log_norm_counts.log_norm_counts`.
    """

    log_normalized_counts: Optional[TatamiNumericPointer] = None
