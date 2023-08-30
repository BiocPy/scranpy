from scranpy import AnalyzeResults, analyze, AnalyzeOptions, MiscellaneousOptions
from singlecellexperiment import SingleCellExperiment
import delayedarray as da

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def test_analyze(mock_data):
    x = mock_data.x
    out = analyze(x, features=[f"{i}" for i in range(1000)])

    assert isinstance(out, AnalyzeResults)

    as_sce = out.to_sce(x)

    assert isinstance(as_sce, SingleCellExperiment)
    assert as_sce.shape[1] == len(out.clusters)

    assert isinstance(as_sce.assay("counts"), da.DelayedArray)
    assert isinstance(as_sce.assay("logcounts"), da.DelayedArray)

    dry = analyze(None, None, dry_run=True)
    assert isinstance(dry, str)


def test_analyze_blocked(mock_data):
    x = mock_data.x
    out = analyze(
        x,
        features=[f"{i}" for i in range(1000)],
        options=AnalyzeOptions(
            miscellaneous_options=MiscellaneousOptions(block=mock_data.block)
        ),
    )

    assert isinstance(out, AnalyzeResults)
    assert out.gene_variances.has_column("per_block")
