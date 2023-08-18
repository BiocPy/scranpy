from typing import Sequence
from mattress import TatamiNumericPointer, tatamize
from copy import deepcopy

from .. import clustering as clust
from .. import dimensionality_reduction as dimred
from .. import feature_selection as feat
from .. import marker_detection as mark
from .. import normalization as norm
from .. import quality_control as qc

from .AnalyzeOptions import AnalyzeOptions
from .AnalyzeResults import AnalyzeResults 
from .run_neighbor_suite import run_neighbor_suite
from ..types import MatrixTypes, is_matrix_expected_type, validate_object_type

__author__ = "ltla, jkanche"
__copyright__ = "ltla"
__license__ = "MIT"

def live_analyze(
    matrix: MatrixTypes,
    features: Sequence[str],
    options: AnalyzeOptions = AnalyzeOptions(),
) -> AnalyzeResults:
    if not is_matrix_expected_type(matrix):
        raise TypeError("matrix is not an expected type.")

    ptr = tatamize(matrix)
    if len(features) != ptr.nrow():
        raise ValueError(
            "Length of `features` not same as number of `rows` in the matrix."
        )

    # Start of the capture.
    results = AnalyzeResults()

    # Don't be tempted to create a shorter variable name, 
    # otherwise the dry-run generator won't work as expected.
    subsets = {}
    if isinstance(options.miscellaneous_options.mito_prefix, str):
        subsets["mito"] = qc.guess_mito_from_symbols(features, options.miscellaneous_options.mito_prefix)
    results.rna_quality_control_subsets = subsets

    rna_options = deepcopy(options.per_cell_rna_qc_metrics_options)
    rna_options.subsets = subsets
    results.rna_quality_control_metrics = qc.per_cell_rna_qc_metrics(
        matrix,
        options = rna_options
    )
    results.rna_quality_control_thresholds = qc.suggest_rna_qc_filters(
        results.rna_quality_control_metrics,
        options=options.suggest_rna_qc_filters_options,
    )

    results.rna_quality_control_filter = qc.create_rna_qc_filter(
        results.rna_quality_control_metrics,
        results.rna_quality_control_thresholds,
        options=options.create_rna_qc_filter_options,
    )

    filtered = qc.filter_cells(
        ptr, 
        filter=results.rna_quality_control_filter
    )

    # Until a delayed array is supported, we can't expose these pointers to 
    # users, so we'll just hold onto them.
    normed = norm.log_norm_counts(
        filtered,
        options=options.log_norm_counts_options,
    )

    results.gene_variances = feat.model_gene_variances(
        normed,
        options=options.model_gene_variances_options,
    )

    results.hvgs = feat.choose_hvgs(
        results.gene_variances.column("residuals"),
        options=options.choose_hvgs_options,
    )

    pca_options = deepcopy(options.run_pca_options)
    pca_options.subset = results.hvgs
    results.pca = dimred.run_pca(
        normed,
        options=options.run_pca_options,
    )

    get_tsne, get_umap, graph, remaining_threads = run_neighbor_suite(
        results.pca.principal_components,
        build_neighbor_index_options=options.build_neighbor_index_options,
        find_nearest_neighbors_options=options.find_nearest_neighbors_options,
        run_umap_options=options.run_umap_options,
        run_tsne_options=options.run_tsne_options,
        build_snn_graph_options=options.build_snn_graph_options,
        num_threads=options.find_nearest_neighbors_options.num_threads # using this as the parallelization extent.
    )

    results.snn_graph = graph
    results.clusters = (
        results.snn_graph.community_multilevel(
            resolution=options.miscellaneous_options.snn_graph_multilevel_resolution
        ).membership
    )

    marker_options = deepcopy(options.score_markers_options)
    marker_options.num_threads = remaining_threads
    results.markers = mark.score_markers(
        normed,
        grouping=results.clusters,
        options=marker_options
    )

    results.tsne = get_tsne()
    results.umap = get_umap()
    return results
