from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from biocframe import BiocFrame

from .._abstract import AbstractOptions
from ..types import validate_object_type
from .choose_hvgs import ChooseHvgsOptions
from .model_gene_variances import ModelGeneVariancesOptions

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


@dataclass
class FeatureSelectionOptions:
    """Optional arguments for feature selection.

    Attributes:
        choose_hvgs (ChooseHvgsOptions): 
            Optional arguments for  :py:meth:`~scranpy.feature_selection.choose_hvgs.choose_hvgs`.
        model_gene_variances (ModelGeneVariancesOptions): 
            Optional arguments for :py:meth:`~scranpy.feature_selection.model_gene_variances.model_gene_variances`.
    """

    choose_hvgs: ChooseHvgsOptions = field(default_factory=ChooseHvgsOptions)
    model_gene_variances: ModelGeneVariancesOptions = field(
        default_factory=ModelGeneVariancesOptions
    )

    def __post_init__(self):
        validate_object_type(self.choose_hvgs, ChooseHvgsOptions)
        validate_object_type(self.model_gene_variances, ModelGeneVariancesOptions)

    def set_threads(self, num_threads: int = 1):
        """Set number of threads to use in each step.

        Args:
            num_threads (int, optional): Number of threads. Defaults to 1.
        """
        self.model_gene_variances.num_threads = num_threads

    def set_verbose(self, verbose: bool = False):
        """Set verbose to display logs.

        Args:
            verbose (bool, optional): Display logs? Defaults to False.
        """
        self.choose_hvgs.verbose = verbose
        self.model_gene_variances.verbose = verbose

    def set_block(self, block: Optional[Sequence] = None):
        """Set the block.

        Args:
            block (Sequence, optional): Blocks assignments
                for each cell. Defaults to None.
        """
        self.model_gene_variances.block = block


@dataclass
class FeatureSelectionResults:
    """Results of feature selection.

    Attributes:
        choose_hvgs (np.ndarray, optional): Output of :py:meth:`~scranpy.feature_selection.choose_hvgs.choose_hvgs`.
        model_gene_variances (BiocFrame, optional): Output of :py:meth:`~scranpy.feature_selection.model_gene_variances.model_gene_variances`.
    """

    hvgs: Optional[np.ndarray] = None
    gene_variances: Optional[BiocFrame] = None
