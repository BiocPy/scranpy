import warnings
from dataclasses import dataclass
from typing import Optional, Sequence

from ..types import validate_object_type

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


@dataclass
class ScoreMarkersArgs:
    """Arguments to score markers.

    Attributes:
        grouping (Sequence, optional): Group assignment for each cell. Defaults to
            None, indicating all cells are part of the same group.
        block (Sequence, optional): Block assignment for each cell.
            This is used to segregate cells in order to perform comparisons within
            each block. Defaults to None, indicating all cells are part of the same
            block.
        threshold (float, optional): Log-fold change threshold to use for computing
            Cohen's d and AUC. Large positive values favor markers with large
            log-fold changes over those with low variance. Defaults to 0.
        compute_auc (bool, optional): Whether to compute the AUCs as an effect size.
            This can be set to false for greater speed and memory efficiency.
            Defaults to True.
        num_threads (int, optional): Number of threads to use. Defaults to 1.
        verbose (bool, optional): Display logs?. Defaults to False.
    """

    grouping: Optional[Sequence] = None
    block: Optional[Sequence] = None
    threshold: float = 0
    compute_auc: bool = True
    num_threads: int = 1
    verbose: bool = False

    def __post_init__(self):
        if self.grouping is None:
            warnings.warn(
                "no cluster/group information is provided for each cell "
                "in this scenario, we consider all cells to be the same cluster"
            )


@dataclass
class MarkerDetectionStepArgs:
    """Arguments to run the marker detection step.

    Attributes:
        score_markers (ScoreMarkersArgs): Arguments to score markers.
    """

    score_markers: ScoreMarkersArgs = ScoreMarkersArgs()

    def __post_init__(self):
        validate_object_type(self.score_markers, ScoreMarkersArgs)
