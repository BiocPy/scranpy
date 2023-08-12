from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from .._abstract import AbstractStepOptions
from ..types import validate_object_type
from .choose_hvgs import ChooseHvgsOptions
from .model_gene_variances import ModelGeneVariancesOptions

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


@dataclass
class FeatureSelectionStepOptions(AbstractStepOptions):
    """Arguments to run the feature selection step.

    Attributes:
        choose_hvg (ChooseHvgsOptions): Arguments to choose highly variable genes
            (:py:meth:`~scranpy.feature_selection.choose_hvgs.choose_hvgs`).
        model_gene_variance (ModelGeneVariancesOptions): Arguments to model gene
            variances
            (:py:meth:`~scranpy.feature_selection.model_gene_variances.model_gene_variances`).
    """

    choose_hvgs: ChooseHvgsOptions = ChooseHvgsOptions()
    model_gene_variances: ModelGeneVariancesOptions = ModelGeneVariancesOptions()

    def __post_init__(self):
        validate_object_type(self.choose_hvgs, ChooseHvgsOptions)
        validate_object_type(self.model_gene_variances, ModelGeneVariancesOptions)

    def set_threads(self, num_threads: int = 1):
        """Set number of threads to use.

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
        """Set block.

        Args:
            block (Sequence, optional): Blocks assignments
                for each cell. Defaults to None.
        """
        self.model_gene_variances.block = block
