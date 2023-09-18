from mattress import tatamize
import numpy

from .. import dimensionality_reduction as dimred
from .. import feature_selection as feat
from .. import marker_detection as mark
from .. import normalization as norm
from .. import quality_control as qc
from .. import batch_correction as correct

from .AnalyzeResults import AnalyzeResults
from .run_neighbor_suite import run_neighbor_suite
from .update import update

__author__ = "ltla, jkanche"
__copyright__ = "ltla"
__license__ = "MIT"


def live_analyze(rna_matrix, adt_matrix, crispr_matrix, options):
    _do_rna = rna_matrix is not None
    _do_adt = adt_matrix is not None
    _do_crispr = crispr_matrix is not None

    NC = None
    if _do_rna:
        rna_ptr = tatamize(rna_matrix)
        NC = rna_ptr.ncol()

    if _do_adt:
        adt_ptr = tatamize(adt_matrix)
        if NC is None and NC != adt_ptr.ncol():
            raise ValueError(
                "all '*_matrix' inputs should have the same number of columns"
            )

    if _do_crispr:
        crispr_ptr = tatamize(crispr_matrix)
        if NC is None and NC != crispr_ptr.ncol():
            raise ValueError(
                "all '*_matrix' inputs should have the same number of columns"
            )

    # Start of the capture.
    results = AnalyzeResults()

    if _do_rna:
        results.rna_quality_control_metrics = qc.per_cell_rna_qc_metrics(
            rna_ptr,
            options=update(
                options.per_cell_rna_qc_metrics_options,
                cell_names=options.miscellaneous_options.cell_names,
            ),
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

    if _do_adt:
        results.adt_quality_control_metrics = qc.per_cell_adt_qc_metrics(
            adt_ptr,
            options=update(
                options.per_cell_adt_qc_metrics_options,
                cell_names=options.miscellaneous_options.cell_names,
            ),
        )

        results.adt_quality_control_thresholds = qc.suggest_adt_qc_filters(
            results.adt_quality_control_metrics,
            options=update(
                options.suggest_adt_qc_filters_options,
                block=options.miscellaneous_options.block,
            ),
        )

        results.adt_quality_control_filter = qc.create_adt_qc_filter(
            results.adt_quality_control_metrics,
            results.adt_quality_control_thresholds,
            options=update(
                options.create_adt_qc_filter_options,
                block=options.miscellaneous_options.block,
            ),
        )

    if _do_crispr:
        results.crispr_quality_control_metrics = qc.per_cell_crispr_qc_metrics(
            crispr_ptr,
            options=update(
                options.per_cell_crispr_qc_metrics_options,
                cell_names=options.miscellaneous_options.cell_names,
            ),
        )

        results.crispr_quality_control_thresholds = qc.suggest_crispr_qc_filters(
            results.crispr_quality_control_metrics,
            options=update(
                options.suggest_crispr_qc_filters_options,
                block=options.miscellaneous_options.block,
            ),
        )

        results.crispr_quality_control_filter = qc.create_crispr_qc_filter(
            results.crispr_quality_control_metrics,
            results.crispr_quality_control_thresholds,
            options=update(
                options.create_crispr_qc_filter_options,
                block=options.miscellaneous_options.block,
            ),
        )

    if _do_rna:
        discard = numpy.zeros(rna_ptr.shape[1], dtype=bool)
    elif _do_adt:
        discard = numpy.zeros(adt_ptr.shape[1], dtype=bool)
    elif _do_crispr:
        discard = numpy.zeros(crispr_ptr.shape[1], dtype=bool)

    if _do_rna and options.miscellaneous_options.filter_on_rna_qc:
        discard = numpy.logical_or(discard, results.rna_quality_control_filter)
    if _do_adt and options.miscellaneous_options.filter_on_adt_qc:
        discard = numpy.logical_or(discard, results.adt_quality_control_filter)
    if _do_crispr and options.miscellaneous_options.filter_on_crispr_qc:
        discard = numpy.logical_or(discard, results.crispr_quality_control_filter)

    if _do_rna:
        rna_filtered = qc.filter_cells(rna_ptr, filter=discard)
    if _do_adt:
        adt_filtered = qc.filter_cells(adt_ptr, filter=discard)
    if _do_crispr:
        crispr_filtered = qc.filter_cells(crispr_ptr, filter=discard)

    results.quality_control_retained = numpy.logical_not(discard)
    if options.miscellaneous_options.block is not None:
        if isinstance(options.miscellaneous_options.block, numpy.ndarray):
            filtered_block = options.miscellaneous_options.block[
                results.quality_control_retained
            ]
        else:
            filtered_block = numpy.array(options.miscellaneous_options.block)[
                results.quality_control_retained
            ]
    else:
        filtered_block = None

    if _do_rna:
        if options.rna_log_norm_counts_options.size_factors is None:
            raw_size_factors = results.rna_quality_control_metrics.column("sums")[
                results.quality_control_retained
            ]
        else:
            raw_size_factors = options.rna_log_norm_counts_options.size_factors

        rna_normed, final_size_factors = norm.log_norm_counts(
            rna_filtered,
            options=update(
                options.rna_log_norm_counts_options,
                size_factors=raw_size_factors,
                center_size_factors_options=update(
                    options.rna_log_norm_counts_options.center_size_factors_options,
                    block=filtered_block,
                ),
                with_size_factors=True,
            ),
        )

        results.rna_size_factors = final_size_factors

        results.gene_variances = feat.model_gene_variances(
            rna_normed,
            options=update(
                options.model_gene_variances_options,
                block=filtered_block,
                feature_names=options.miscellaneous_options.rna_feature_names,
            ),
        )

        results.hvgs = feat.choose_hvgs(
            results.gene_variances.column("residuals"),
            options=options.choose_hvgs_options,
        )

        results.rna_pca = dimred.run_pca(
            rna_normed,
            options=update(
                options.rna_run_pca_options, subset=results.hvgs, block=filtered_block
            ),
        )

    if _do_adt:
        if options.adt_log_norm_counts_options.size_factors is None:
            raw_size_factors = results.adt_quality_control_metrics.column("sums")[
                results.quality_control_retained
            ]

            raw_size_factors = norm.grouped_size_factors(
                adt_filtered,
                options=update(
                    options.grouped_size_factors_options,
                    block=filtered_block,
                    initial_size_factors=raw_size_factors,
                ),
            )
        else:
            raw_size_factors = options.adt_log_norm_counts_options.size_factors

        adt_normed, final_size_factors = norm.log_norm_counts(
            adt_filtered,
            options=update(
                options.adt_log_norm_counts_options,
                size_factors=raw_size_factors,
                center_size_factors_options=update(
                    options.adt_log_norm_counts_options.center_size_factors_options,
                    block=filtered_block,
                ),
                with_size_factors=True,
            ),
        )

        results.adt_size_factors = final_size_factors

        results.adt_pca = dimred.run_pca(
            adt_normed,
            options=update(options.adt_run_pca_options, block=filtered_block),
        )

    if _do_crispr:
        if options.crispr_log_norm_counts_options.size_factors is None:
            raw_size_factors = results.crispr_quality_control_metrics.column("sums")[
                results.quality_control_retained
            ]
        else:
            raw_size_factors = options.crispr_log_norm_counts_options.size_factors

        crispr_normed, final_size_factors = norm.log_norm_counts(
            crispr_filtered,
            options=update(
                options.crispr_log_norm_counts_options,
                size_factors=raw_size_factors,
                center_size_factors_options=update(
                    options.crispr_log_norm_counts_options.center_size_factors_options,
                    block=filtered_block,
                ),
                with_size_factors=True,
            ),
        )

        results.crispr_size_factors = final_size_factors

        results.crispr_pca = dimred.run_pca(
            crispr_normed,
            options=update(options.crispr_run_pca_options, block=filtered_block),
        )

    if _do_rna + _do_adt + _do_crispr > 1:
        all_embeddings = []
        if _do_rna:
            all_embeddings.append(results.rna_pca.principal_components)
        if _do_adt:
            all_embeddings.append(results.adt_pca.principal_components)
        if _do_crispr:
            all_embeddings.append(results.crispr_pca.principal_components)

        results.combined_pcs = dimred.combine_embeddings(
            all_embeddings, options=options.combine_embeddings_options
        )
        lowdim = results.combined_pcs
    else:
        if _do_rna:
            lowdim = results.rna_pca.principal_components
        elif _do_adt:
            lowdim = results.adt_pca.principal_components
        elif _do_crispr:
            lowdim = results.crispr_pca.principal_components

    if options.miscellaneous_options.block is not None:
        results.mnn = correct.mnn_correct(
            lowdim,
            filtered_block,
            options=options.mnn_correct_options,
        )
        lowdim = results.mnn.corrected

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

    if _do_rna:
        results.rna_markers = mark.score_markers(
            rna_normed,
            grouping=results.clusters,
            options=update(
                options.rna_score_markers_options,
                block=filtered_block,
                feature_names=options.miscellaneous_options.rna_feature_names,
                num_threads=remaining_threads,
            ),
        )

    if _do_adt:
        results.adt_markers = mark.score_markers(
            adt_normed,
            grouping=results.clusters,
            options=update(
                options.adt_score_markers_options,
                block=filtered_block,
                feature_names=options.miscellaneous_options.adt_feature_names,
                num_threads=remaining_threads,
            ),
        )

    if _do_crispr:
        results.crispr_markers = mark.score_markers(
            crispr_normed,
            grouping=results.clusters,
            options=update(
                options.crispr_score_markers_options,
                block=filtered_block,
                feature_names=options.miscellaneous_options.crispr_feature_names,
                num_threads=remaining_threads,
            ),
        )

    results.tsne = get_tsne()
    results.umap = get_umap()
    return results
