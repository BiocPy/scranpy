from scranpy.analyze import AnalyzeResults, analyze
from singlecellexperiment import SingleCellExperiment

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def test_analyze(mock_data):
    x = mock_data.x
    out = analyze(x, features=[f"{i}" for i in range(1000)])

    assert isinstance(out, AnalyzeResults)

    as_sce = out.to_sce(x)

    assert isinstance(as_sce, SingleCellExperiment)
    assert as_sce.shape[1] == len(out.clustering.clusters)