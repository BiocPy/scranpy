from typing import Sequence
from mattress import tatamize
import numpy

from .. import dimensionality_reduction as dimred
from .. import feature_selection as feat
from .. import marker_detection as mark
from .. import normalization as norm
from .. import quality_control as qc
from .. import batch_correction as correct

from .AnalyzeOptions import AnalyzeOptions
from .AnalyzeResults import AnalyzeResults
from .run_neighbor_suite import run_neighbor_suite
from .update import update
from ..types import MatrixTypes, is_matrix_expected_type

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

    subsets = {}
    if isinstance(options.miscellaneous_options.mito_prefix, str):
        subsets["mito"] = qc.guess_mito_from_symbols(
            features, options.miscellaneous_options.mito_prefix
        )
    results.rna_quality_control_subsets = subsets

    results.rna_quality_control_metrics = qc.per_cell_rna_qc_metrics(
        matrix, options=update(options.per_cell_rna_qc_metrics_options, subsets=subsets)
    )

    results.rna_quality_control_thresholds = qc.suggest_rna_qc_filters(
        results.rna_quality_control_metrics,
        options=update(
            options.suggest_rna_qc_filters_options,
            block=options.miscellaneous_options.block,
        ),
    )

    results.rna_quality_control_filter = qc.create_rna_qc_filter(
        results.rna_quality_control_metrics,
        results.rna_quality_control_thresholds,
        options=update(
            options.create_rna_qc_filter_options,
            block=options.miscellaneous_options.block,
        ),
    )

    filtered = qc.filter_cells(ptr, filter=results.rna_quality_control_filter)

    keep = numpy.logical_not(results.rna_quality_control_filter)
    if options.miscellaneous_options.block is not None:
        if isinstance(options.miscellaneous_options.block, numpy.ndarray):
            filtered_block = options.miscellaneous_options.block[keep]
        else:
            filtered_block = numpy.array(options.miscellaneous_options.block)[keep]
    else:
        filtered_block = None

    if options.log_norm_counts_options.size_factors is None:
        results.size_factors = norm.center_size_factors(
            results.rna_quality_control_metrics.column("sums")[keep],
            options=update(options.center_size_factors_options, block=filtered_block),
        )
    else:
        results.size_factors = options.log_norm_counts_options.size_factors[keep]

    normed = norm.log_norm_counts(
        filtered,
        options=update(
            options.log_norm_counts_options, size_factors=results.size_factors
        ),
    )

    results.gene_variances = feat.model_gene_variances(
        normed,
        options=update(options.model_gene_variances_options, block=filtered_block),
    )

    results.hvgs = feat.choose_hvgs(
        results.gene_variances.column("residuals"),
        options=options.choose_hvgs_options,
    )

    results.pca = dimred.run_pca(
        normed,
        options=update(
            options.run_pca_options, subset=results.hvgs, block=filtered_block
        ),
    )

    if options.miscellaneous_options.block is not None:
        results.mnn = correct.mnn_correct(
            results.pca.principal_components,
            filtered_block,
            options = options.mnn_correct_options,
        )
        lowdim = results.mnn.corrected
    else:
        lowdim = results.pca.principal_components

    get_tsne, get_umap, graph, remaining_threads = run_neighbor_suite(
        lowdim,
        build_neighbor_index_options=options.build_neighbor_index_options,
        find_nearest_neighbors_options=options.find_nearest_neighbors_options,
        run_umap_options=options.run_umap_options,
        run_tsne_options=options.run_tsne_options,
        build_snn_graph_options=options.build_snn_graph_options,
        num_threads=options.find_nearest_neighbors_options.num_threads,  # using this as the parallelization extent.
    )

    results.snn_graph = graph
    results.clusters = results.snn_graph.community_multilevel(
        resolution=options.miscellaneous_options.snn_graph_multilevel_resolution
    ).membership

    results.markers = mark.score_markers(
        normed,
        grouping=results.clusters,
        options=update(
            options.score_markers_options,
            block=filtered_block,
            num_threads=remaining_threads,
        ),
    )

    results.tsne = get_tsne()
    results.umap = get_umap()
    return results
