from dataclasses import dataclass

from ..types import validate_object_type
from .score_markers import ScoreMarkersArgs

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


@dataclass
class MarkerDetectionStepArgs:
    """Arguments to run the marker detection step.

    Attributes:
        score_markers (ScoreMarkersArgs): Arguments to score markers
            (:py:meth:`~scranpy.marker_detection.score_markers.score_markers`).
    """

    score_markers: ScoreMarkersArgs = ScoreMarkersArgs()

    def __post_init__(self):
        validate_object_type(self.score_markers, ScoreMarkersArgs)


def run(input, options: MarkerDetectionStepArgs = MarkerDetectionStepArgs()):
    pass
