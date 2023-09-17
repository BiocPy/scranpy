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
    out = analyze(x)

    assert isinstance(out, AnalyzeResults)
    assert out.mnn is None

    f = [f"{i}" for i in range(1000)]
    as_sce = out.to_sce(x, f)

    assert isinstance(as_sce, SingleCellExperiment)
    assert as_sce.shape[1] == len(out.clusters)

    assert isinstance(as_sce.assay("counts"), da.DelayedArray)
    assert isinstance(as_sce.assay("logcounts"), da.DelayedArray)

    dry = analyze(x, dry_run=True)
    assert isinstance(dry, str)


def test_analyze_size_factors(mock_data):
    x = mock_data.x
    out = analyze(
        x,
        options=AnalyzeOptions(
            rna_log_norm_counts_options=LogNormCountsOptions(
                size_factors=np.ones(x.shape[1])
            )
        ),
    )

    assert (out.rna_size_factors == np.ones(x.shape[1])).all()


def test_analyze_blocked(mock_data):
    x = mock_data.x
    out = analyze(
        x,
        options=AnalyzeOptions(
            miscellaneous_options=MiscellaneousOptions(block=mock_data.block)
        ),
    )

    assert isinstance(out, AnalyzeResults)
    assert out.gene_variances.has_column("per_block")
    assert out.mnn is not None

    f = [f"{i}" for i in range(1000)]
    as_sce = out.to_sce(x, f)
    assert "mnn" in as_sce.reduced_dims


def test_analyze_multimodal():
    rna = np.random.rand(1000, 200)
    adt = np.random.rand(20, 200)
    crispr = np.random.rand(100, 200)

    out = analyze(rna, adt, crispr)
    assert out.adt_pca is not None
    assert out.crispr_pca is not None
    assert out.adt_markers is not None
    assert out.crispr_markers is not None

    rna_feat = [f"gene{i}" for i in range(1000)]
    adt_feat = [f"tag{i}" for i in range(20)]
    crispr_feat = [f"guide{i}" for i in range(100)]
    as_sce = out.to_sce(rna, rna_feat, adt, adt_feat, crispr, crispr_feat)

    assert isinstance(as_sce, SingleCellExperiment)
    assert as_sce.shape[1] == len(out.clusters)
    assert as_sce.main_experiment_name == "rna"
    assert list(as_sce.alternative_experiments.keys()) == ["adt", "crispr"]

    dry = analyze(rna, adt, crispr, dry_run=True)
    assert isinstance(dry, str)
