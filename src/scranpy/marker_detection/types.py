from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

from ..types import validate_object_type
from .score_markers import ScoreMarkersOptions

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


@dataclass
class MarkerDetectionOptions:
    """Arguments to run the marker detection step.

    Attributes:
        score_markers (ScoreMarkersOptions): Arguments to score markers
            (:py:meth:`~scranpy.marker_detection.score_markers.score_markers`).
    """

    score_markers: ScoreMarkersOptions = field(default_factory=ScoreMarkersOptions)

    def __post_init__(self):
        validate_object_type(self.score_markers, ScoreMarkersOptions)

    def set_threads(self, num_threads: int = 1):
        """Set number of threads to use.

        Args:
            num_threads (int, optional): Number of threads. Defaults to 1.
        """
        self.score_markers.num_threads = num_threads

    def set_verbose(self, verbose: bool = False):
        """Set verbose to display logs.

        Args:
            verbose (bool, optional): Display logs? Defaults to False.
        """
        self.score_markers.verbose = verbose

    def set_block(self, block: Optional[Sequence] = None):
        """Set block.

        Args:
            block (Sequence, optional): Blocks assignments
                for each cell. Defaults to None.
        """
        self.score_markers.block = block


@dataclass
class MarkerDetectionResults:
    """Results of the marker detection step.

    Attributes:
        markers (Mapping, optional): Result of score markers
            (:py:meth:`~scranpy.marker_detection.score_markers.score_markers`).
    """

    markers: Optional[Mapping] = None
