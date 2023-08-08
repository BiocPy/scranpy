from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Sequence

import numpy as np
from singlecellexperiment import SingleCellExperiment

from .types import is_matrix_expected_type

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"

# Arguments for various steps


@dataclass(slots=True)
class BaseArgs:
    input: Any
    num_threads: int = 1

    def __post_init__(self):
        if (not is_matrix_expected_type(self.input)) and (
            not isinstance(input, SingleCellExperiment)
        ):
            raise TypeError(
                "input matrix must either be a dense, sparse or `SingleCellExperiment`"
                f"provided `{type(input)}`"
            )


@dataclass(slots=True)
class ClusterArgs(BaseArgs):
    num_neighbors: int = 10
    approximate: bool = True
    weight_scheme: Literal["ranked", "jaccard", "number"] = "ranked"

    def __post_init__(self):
        if self.weight_scheme not in ["ranked", "jaccard", "number"]:
            raise ValueError(
                '\'weight_scheme\' must be one of "ranked", "jaccard", "number"'
                f"provided {self.weight_scheme}"
            )


@dataclass(slots=True)
class PcaArgs(BaseArgs):
    rank: int = 50
    subset: Optional[Sequence] = None
    block: Optional[Sequence] = None
    scale: bool = False
    block_method: Literal["none", "project", "regress"] = "project"
    block_weights: bool = True

    def __post_init__(self):
        if self.block_method not in ["none", "project", "regress"]:
            raise ValueError(
                '\'block_method\' must be one of "none", "project", "regress"'
                f"provided {self.block_method}"
            )


@dataclass(slots=True)
class TsneArgs(BaseArgs):
    perplexity: int = 30
    seed: int = 42
    max_iterations: int = 500


@dataclass(slots=True)
class UmapArgs(BaseArgs):
    min_dist: float = 0.1
    num_neighbors: int = 15
    num_epochs: int = 500
    seed: int = 42


@dataclass(slots=True)
class FeatureSelectionArgs(BaseArgs):
    block: Optional[Sequence] = None
    span: float = 0.3
    number_of_hvgs: int = 2500


@dataclass(slots=True)
class MarkerDetectionArgs(BaseArgs):
    grouping: Sequence
    block: Optional[Sequence] = None
    threshold: float = 0
    compute_auc: bool = True


@dataclass(slots=True)
class NNArgs(BaseArgs):
    approximate: bool = True
    num_neighbors: int = 10


@dataclass(slots=True)
class NormalizationArgs(BaseArgs):
    block: Optional[Sequence] = None
    size_factors: Optional[np.ndarray] = None
    center: bool = True
    allow_zeros: bool = False
    allow_non_finite: bool = False


@dataclass(slots=True)
class RnaQcArgs(BaseArgs):
    subsets: Optional[Mapping] = None
    block: Optional[Sequence] = None
    num_mads: int = 3
    mito_prefix: str = "mt-"
