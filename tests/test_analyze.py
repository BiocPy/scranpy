from scranpy import AnalyzeResults, analyze, analyze_se, analyze_sce, AnalyzeOptions, MiscellaneousOptions
from scranpy.normalization import LogNormCountsOptions
from singlecellexperiment import SingleCellExperiment
from summarizedexperiment import SummarizedExperiment
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


def test_analyze_multimodal_skip_qc():
    rna = np.random.rand(1000, 200)
    adt = np.random.rand(20, 200)
    crispr = np.random.rand(100, 200)

    # Works with only one.
    out = analyze(
        rna,
        adt,
        crispr,
        options=AnalyzeOptions(
            miscellaneous_options=MiscellaneousOptions(
                filter_on_rna_qc=False,
                filter_on_adt_qc=False,
                filter_on_crispr_qc=True,
            )
        ),
    )
    assert (out.crispr_quality_control_filter != out.quality_control_retained).all()

    # Works with none.
    out = analyze(
        rna,
        adt,
        crispr,
        options=AnalyzeOptions(
            miscellaneous_options=MiscellaneousOptions(
                filter_on_rna_qc=False,
                filter_on_adt_qc=False,
                filter_on_crispr_qc=False,
            )
        ),
    )
    assert len(out.rna_size_factors) == 200


def test_analyze_summarizedexperiment(mock_data):
    se = SummarizedExperiment({ "counts": mock_data.x })
    se.row_names = [f"gene{i}" for i in range(1000)]
    out = analyze_se(se, assay_type="counts")
    assert out.gene_variances.row_names == se.row_names

    sce = SingleCellExperiment({ "counts": mock_data.x })
    sce.row_names = [f"gene{i}" for i in range(1000)]
    adt_se = SummarizedExperiment({ "counts": np.random.rand(20, mock_data.x.shape[1]) })
    adt_se.row_names = [f"tag{i}" for i in range(20)]
    sce.alternative_experiments = { "adt": adt_se }
    out = analyze_sce(sce, adt_exp = "adt", assay_type="counts")
    assert out.gene_variances.row_names == se.row_names
    assert out.adt_markers["0"].row_names == adt_se.row_names
