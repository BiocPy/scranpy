from typing import Tuple, List, Mapping, Optional, Sequence, Union, Callable

from mattress import TatamiNumericPointer, tatamize
from copy import deepcopy

from .. import clustering as clust
from .. import dimensionality_reduction as dimred
from .. import feature_selection as feat
from .. import marker_detection as mark
from .. import nearest_neighbors as nn
from .. import normalization as norm
from .. import quality_control as qc
from .types import MatrixTypes, is_matrix_expected_type, validate_object_type

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

    NR = ptr.nrow()
    NC = ptr.ncol()

    if len(features) != NR:
        raise ValueError(
            "Length of `features` not same as number of `rows` in the matrix."
        )

    if options.block is not None:
        if len(options.block) != NC:
            raise ValueError(
                "Length of `block` not same as number of `columns` in the matrix."
            )

    # Start of the capture.
    results = AnalyzeResults()

    subsets = {}
    if options.quality_control.mito_subset is not None:
        if isinstance(options.quality_control.mito_subset, str):
            subsets["mito"] = qc.guess_mito_from_symbols(
                features, options.quality_control.mito_subset
            )
        elif isinstance(options.quality_control.mito_subset, bool):
            subsets["mito"] = qc.guess_mito_from_symbols(features)
        else:
            raise ValueError(
                "Unsupported value provided for `qc_mito_subset`:"
                f" {options.quality_control.mito_subset}"
            )

    results.quality_control.subsets = subsets

    rna_options = deepcopy(options.quality_control.per_cell_rna_qc_metrics)
    rna_options.subsets = subsets
    results.quality_control.qc_metrics = qc.per_cell_rna_qc_metrics(
        matrix,
        options = rna_options
    )
    results.quality_control.qc_thresholds = qc.suggest_rna_qc_filters(
        results.quality_control.qc_metrics,
        options=options.quality_control.suggest_rna_qc_filters,
    )

    results.quality_control.qc_filter = qc.create_rna_qc_filter(
        results.quality_control.qc_metrics,
        results.quality_control.qc_thresholds,
        options.quality_control.create_rna_qc_filter,
    )

    results.quality_control.filtered_cells = qc.filter_cells(
        ptr, filter=results.quality_control.qc_filter
    )

    # Until a delayed array is supported, we can't expose these pointers to 
    # users, so we'll just hold onto them.
    normed = norm.log_norm_counts(
        results.quality_control.filtered_cells,
        options=options.normalization.log_norm_counts,
    )

    results.feature_selection.gene_variances = feat.model_gene_variances(
        normed,
        options=options.feature_selection.model_gene_variances,
    )

    results.feature_selection.hvgs = feat.choose_hvgs(
        results.feature_selection.gene_variances.column("residuals"),
        options=options.feature_selection.choose_hvgs,
    )

    pca_options = deepcopy(options.dimensionality_reduction.run_pca)
    pca_options.subset = results.feature_selection.hvgs
    results.dimensionality_reduction.pca = dimred.run_pca(
        normed,
        options=options.dimensionality_reduction.run_pca,
    )

    get_tsne, get_umap, graph, remaining_threads = run_neighbor_suite(
        results.dimensionality_reduction.pca.principal_components,
        build_neighbor_index_options=options.nearest_neighbors.build_neighbor_index,
        find_nearest_neighbors_options=options.nearest_neighbors.find_nearest_neighbors,
        run_umap_options=options.dimensionality_reduction.run_umap,
        run_tsne_options=options.dimensionality_reduction.run_tsne,
        build_snn_graph_options=options.clustering.build_snn_graph,
        num_threads=options.nearest_neighbors.find_nearest_neighbors.num_threads # using this as the parallelization extent.
    )

    results.clustering.build_snn_graph = graph
    results.clustering.clusters = (
        results.clustering.build_snn_graph.community_multilevel(
            resolution=options.clustering.resolution
        ).membership
    )

    marker_options = deepcopy(options.marker_detection)
    marker_options.num_threads = remaining_threads
    results.marker_detection.markers = mark.score_markers(
        normed,
        grouping=results.clustering.clusters,
        options=options.marker_detection.score_markers,
    )

    results.dimensionality_reduction.tsne = get_tsne()
    results.dimensionality_reduction.umap = get_umap()
    return results
