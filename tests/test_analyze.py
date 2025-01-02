import scranpy
import scrnaseq
import numpy

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


zeisel_sce = None
def get_zeisel_data():
    global zeisel_sce
    if zeisel_sce is None:
        zeisel_sce = scrnaseq.fetch_dataset("zeisel-brain-2015", "2023-12-14", realize_assays=True)
        zeisel_sce = zeisel_sce[:,:500] # only using the first 500 cells for speed.
    return zeisel_sce


def test_analyze_simple():
    sce = get_zeisel_data()

    # Lowering the MADs to check that filtering has some effect.
    res = scranpy.analyze(sce.assay(0), suggest_rna_qc_thresholds_options={ "num_mads": 1 })

    assert (res.combined_qc_filter == res.rna_qc_filter).all()
    assert res.combined_qc_filter.sum() < sce.shape[1]
    assert res.rna_filtered.shape[0] == sce.shape[0] 
    assert res.rna_filtered.shape[1] < sce.shape[1] 

    assert numpy.allclose(res.rna_size_factors.mean(), 1)
    assert res.rna_filtered.shape == res.rna_normalized.shape

    assert res.rna_pca.components.shape[1] == res.rna_filtered.shape[1]
    assert res.combined_pca == "rna_pca"
    assert res.mnn_corrected is None

    assert len(res.graph_clusters.membership) == res.rna_normalized.shape[1] 
    assert res.kmeans_clusters is None
    nclusters = len(set(res.graph_clusters.membership))
    assert nclusters > 1
    assert len(res.rna_markers.cohens_d) == nclusters

    assert res.tsne.shape[0] == 2
    assert res.umap.shape[0] == 2

    assert res.adt_filtered is None
    assert res.adt_normalized is None
    assert res.adt_size_factors is None
    assert res.adt_pca is None

    assert res.crispr_filtered is None
    assert res.crispr_normalized is None
    assert res.crispr_size_factors is None
    assert res.crispr_pca is None


def test_analyze_adt():
    sce = get_zeisel_data()

    # We're going to pretend the spike-in data are ADTs, for testing purposes.
    # Again using a lower threshold to check that some filtering gets performed.
    res = scranpy.analyze(
        sce.assay(0), 
        adt_x=sce.alternative_experiment("ERCC").assay(0),
        suggest_rna_qc_thresholds_options={ "num_mads": 1 }
    )

    assert not (res.combined_qc_filter == res.rna_qc_filter).all()
    assert (res.combined_qc_filter == numpy.logical_and(res.rna_qc_filter, res.adt_qc_filter)).all()

    assert res.adt_filtered.shape[1] == res.rna_filtered.shape[1]
    assert res.adt_filtered.shape == res.adt_normalized.shape
    assert numpy.allclose(res.adt_size_factors.mean(), 1)

    assert res.adt_pca.components.shape[1] == res.rna_pca.components.shape[1]
    assert len(res.combined_pca.scaling) == 2
    assert res.combined_pca.combined.shape[1] == res.adt_pca.components.shape[1]
    assert res.combined_pca.combined.shape[0] == res.rna_pca.components.shape[0] + res.adt_pca.components.shape[0]

    assert res.crispr_filtered is None
    assert res.crispr_normalized is None
    assert res.crispr_size_factors is None
    assert res.crispr_pca is None


def test_analyze_crispr():
    sce = get_zeisel_data()

    # We're going to pretend the spike-in data are ADTs and CRISPR, for testing purposes.
    mock = sce.alternative_experiment("ERCC").assay(0)
    res = scranpy.analyze(rna_x=None, crispr_x=mock, adt_x=mock)
    assert res.adt_filtered.shape[1] == res.crispr_filtered.shape[1]
    assert res.crispr_filtered.shape == res.crispr_normalized.shape
    assert numpy.allclose(res.crispr_size_factors.mean(), 1)

    assert res.adt_pca.components.shape[1] == res.crispr_pca.components.shape[1]
    assert len(res.combined_pca.scaling) == 2
    assert res.combined_pca.combined.shape[1] == res.adt_pca.components.shape[1]
    assert res.combined_pca.combined.shape[0] == res.crispr_pca.components.shape[0] + res.adt_pca.components.shape[0]

    # Checking that this all works even if no RNA data is supplied.
    assert res.rna_filtered is None
    assert res.rna_normalized is None
    assert res.rna_size_factors is None
    assert res.rna_pca is None


def test_analyze_block():
    sce = get_zeisel_data()

    # Using the tissue as the blocking factor for testing purposes.
    block = sce.get_column_data().get_column("tissue")
    levels = sorted(list(set(block)))
    res = scranpy.analyze(rna_x=sce.assay(0), block=block)

    assert res.mnn_corrected is not None
    assert sorted(res.mnn_corrected.merge_order) == levels
    assert len(res.rna_qc_thresholds.sum) == len(levels)
    assert res.rna_pca.center.shape[0] == len(levels)
    assert len(res.rna_gene_variances.per_block) == len(levels)
    assert not numpy.allclose(res.rna_size_factors.mean(), 1) # as the size factors are only scaled to a mean of 1 in the lowest-coverage block.


def test_analyze_nofilter():
    sce = get_zeisel_data()

    # Lowering the MADs to check that filtering would some effect, if not for filter_cells=False.
    res = scranpy.analyze(
        sce.assay(0), 
        filter_cells=False,
        suggest_rna_qc_thresholds_options={ "num_mads": 1 }
    )

    assert res.combined_qc_filter.sum() < sce.shape[1]
    assert res.rna_filtered.shape == sce.shape


def test_analyze_kmeans():
    sce = get_zeisel_data()

    res = scranpy.analyze(
        sce.assay(0), 
        kmeans_clusters = 13,
    )
    assert len(set(res.kmeans_clusters.clusters)) == 13
    assert len(res.rna_markers.auc) == len(set(res.graph_clusters.membership))

    res = scranpy.analyze(
        sce.assay(0), 
        kmeans_clusters = 13,
        clusters_for_markers = ["kmeans"]
    )
    assert len(res.rna_markers.auc) == 13


def test_analyze_nocluster():
    sce = get_zeisel_data()

    res = scranpy.analyze(
        sce.assay(0), 
        cluster_graph_options=None
    )

    # No clusters, no markers.
    assert res.kmeans_clusters is None
    assert res.graph_clusters is None
    assert res.rna_markers is None
