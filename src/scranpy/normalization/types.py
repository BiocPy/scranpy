from dataclasses import dataclass, field
from typing import Optional, Sequence

from mattress import TatamiNumericPointer

from .._abstract import AbstractOptions
from ..types import validate_object_type
from .log_norm_counts import LogNormCountsOptions

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


@dataclass
class NormalizationOptions(AbstractOptions):
    """Optional arguments for normalization.

    Attributes:
        log_norm_counts (LogNormCountsOptions): Options to pass to
            :py:meth:`~scranpy.normalization.log_norm_counts.log_norm_counts`.
    """

    log_norm_counts: LogNormCountsOptions = field(
        default_factory=LogNormCountsOptions
    )

    def __post_init__(self):
        validate_object_type(self.log_norm_counts, LogNormCountsOptions)

    def set_threads(self, num_threads: int = 1):
        """Set number of threads to use.

        Args:
            num_threads (int, optional): Number of threads. Defaults to 1.
        """
        self.log_norm_counts.num_threads = num_threads

    def set_verbose(self, verbose: bool = False):
        """Set verbose to display logs.

        Args:
            verbose (bool, optional): Display logs? Defaults to False.
        """
        self.log_norm_counts.verbose = verbose

    def set_block(self, block: Optional[Sequence] = None):
        """Set block.

        Args:
            block (Sequence, optional): Blocks assignments
                for each cell. Defaults to None.
        """
        self.log_norm_counts.block = block


@dataclass
class NormalizationResults:
    """Results of the normalization step.

    Attributes:
        log_norm_counts (TatamiNumericPointer, optional): Output of
            :py:meth:`~scranpy.normalization.log_norm_counts.log_norm_counts`.
    """

    log_norm_counts: Optional[TatamiNumericPointer] = None
