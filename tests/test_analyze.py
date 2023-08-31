from scranpy import AnalyzeResults, analyze, AnalyzeOptions, MiscellaneousOptions
from scranpy.normalization import LogNormCountsOptions
from singlecellexperiment import SingleCellExperiment
import delayedarray as da
import numpy as np

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def test_analyze(mock_data):
    x = mock_data.x
    out = analyze(x, features=[f"{i}" for i in range(1000)])

    assert isinstance(out, AnalyzeResults)
    assert out.mnn is None

    as_sce = out.to_sce(x)

    assert isinstance(as_sce, SingleCellExperiment)
    assert as_sce.shape[1] == len(out.clusters)

    assert isinstance(as_sce.assay("counts"), da.DelayedArray)
    assert isinstance(as_sce.assay("logcounts"), da.DelayedArray)

    dry = analyze(None, None, dry_run=True)
    assert isinstance(dry, str)


def test_analyze_size_factors(mock_data):
    x = mock_data.x
    out = analyze(
        x,
        features=[f"{i}" for i in range(1000)],
        options=AnalyzeOptions(
            log_norm_counts_options=LogNormCountsOptions(
                size_factors=np.ones(x.shape[1])
            )
        ),
    )

    assert (out.size_factors == np.ones(x.shape[1])).all()


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
    assert out.mnn is not None

    as_sce = out.to_sce(x)
    assert "mnn" in as_sce.reducedDims
