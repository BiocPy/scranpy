from dataclasses import dataclass

from ..types import validate_object_type
from .choose_hvgs import ChooseHvgArgs
from .model_gene_variances import ModelGeneVariancesArgs

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


@dataclass
class FeatureSelectionStepArgs:
    """Arguments to run the feature selection step.

    Attributes:
        choose_hvg (ChooseHvgArgs): Arguments to choose highly variable genes
            (:py:meth:`~scranpy.feature_selection.choose_hvgs.choose_hvgs`).
        model_gene_variance (ModelGeneVariancesArgs): Arguments to model gene variances
            (:py:meth:`~scranpy.feature_selection.model_gene_variances.model_gene_variances`).
    """

    choose_hvg: ChooseHvgArgs = ChooseHvgArgs()
    model_gene_variance: ModelGeneVariancesArgs = ModelGeneVariancesArgs()

    def __post_init__(self):
        validate_object_type(self.choose_hvg, ChooseHvgArgs)
        validate_object_type(self.model_gene_variance, ModelGeneVariancesArgs)


def run(input, options: FeatureSelectionStepArgs = FeatureSelectionStepArgs()):
    pass
