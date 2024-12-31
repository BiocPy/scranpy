from .adt_quality_control import *
from .rna_quality_control import *
from .crispr_quality_control import *
from .normalize_counts import *
from .center_size_factors import *
from .compute_clrm1_factors import *
from .model_gene_variances import *
from .fit_variance_trend import *
from .choose_highly_variable_genes import *
from .run_pca import *
from .cluster_kmeans import *
from .run_all_neighbor_steps import *
from .score_markers import *
from .summarize_effects import *
from .correct_mnn import *
from .scale_by_neighbors import *

from typing import Any, Sequence, Optional, Union
from collections.abc import Mapping
from dataclasses import dataclass

import biocutils
import delayedarray
import numpy


@dataclass
class AnalyzeResults:
    """Results of :py:func:`~scranpy.analyze.analyse`."""

    rna_qc_metrics: Optional[ComputeRnaQcMetricsResults]
    """Results of :py:func:`~scranpy.rna_quality_control.compute_rna_qc_metrics`.
    If RNA data is not available, this is set to ``None`` instead."""

    rna_qc_thresholds: Optional[SuggestRnaQcThresholdsResults]
    """Results of :py:func:`~scranpy.rna_quality_control.suggest_rna_qc_thresholds`.
    If RNA data is not available, this is set to ``None`` instead."""

    rna_qc_filter: Optional[numpy.ndarray]
    """Results of :py:func:`~scranpy.rna_quality_control.filter_rna_qc_metrics`.
    If RNA data is not available, this is set to ``None`` instead."""

    adt_qc_metrics: Optional[ComputeAdtQcMetricsResults]
    """Results of :py:func:`~scranpy.adt_quality_control.compute_adt_qc_metrics`.
    If ADT data is not available, this is set to ``None`` instead."""

    adt_qc_thresholds: Optional[SuggestAdtQcThresholdsResults]
    """Results of :py:func:`~scranpy.adt_quality_control.suggest_adt_qc_thresholds`.
    If ADT data is not available, this is set to ``None`` instead."""

    adt_qc_filter: Optional[numpy.ndarray]
    """Results of :py:func:`~scranpy.adt_quality_control.filter_adt_qc_metrics`.
    If ADT data is not available, this is set to ``None`` instead."""

    crispr_qc_metrics: Optional[ComputeCrisprQcMetricsResults]
    """Results of :py:func:`~scranpy.crispr_quality_control.compute_crispr_qc_metrics`.
    If CRISPR data is not available, this is set to ``None`` instead."""

    crispr_qc_thresholds: Optional[SuggestCrisprQcThresholdsResults]
    """Results of :py:func:`~scranpy.crispr_quality_control.suggest_crispr_qc_thresholds`.
    If CRISPR data is not available, this is set to ``None`` instead."""

    crispr_qc_filter: Optional[numpy.ndarray]
    """Results of :py:func:`~scranpy.crispr_quality_control.filter_crispr_qc_metrics`.
    If CRISPR data is not available, this is set to ``None`` instead."""

    combined_filter: numpy.ndarray
    """Array of booleans indicating which cells are of high quality and should be retained for downstream analyses."""

    filtered_rna: Optional[delayedarray.DelayedArray]
    """Matrix of RNA counts that has been filtered to only contain the high-quality cells in :py:attr:`~combined_filter`.
    If RNA data is not available, this is set to ``None`` instead."""

    filtered_adt: Optional[delayedarray.DelayedArray]
    """Matrix of ADT counts that has been filtered to only contain the high-quality cells in :py:attr:`~combined_filter`.
    If ADT data is not available, this is set to ``None`` instead."""

    filtered_crispr: Optional[delayedarray.DelayedArray]
    """Matrix of CRISPR counts that has been filtered to only contain the high-quality cells in :py:attr:`~combined_filter`.
    If CRISPR data is not available, this is set to ``None`` instead."""

    rna_size_factors: Optional[numpy.ndarray]
    """Size factors for the RNA count matrix, derived from the sum of counts for each cell.
    Size factors are centered with :py:func:`~scranpy.center_size_factors.center_size_factors`.
    If RNA data is not available, this is set to ``None`` instead."""

    normalized_rna: Optional[delayedarray.DelayedArray]
    """Matrix of (log-)normalized expression values derived from RNA counts, as computed by :py:func:`~scranpy.normalize_counts.normalize_counts` using :py:attr:`~rna_size_factors`.
    If RNA data is not available, this is set to ``None`` instead."""

    adt_size_factors: Optional[numpy.ndarray]
    """Size factors for the ADT count matrix, computed using the CLRm1 method.
    Size factors are centered with :py:func:`~scranpy.center_size_factors.center_size_factors`.
    If ADT data is not available, this is set to ``None`` instead."""

    normalized_adt: Optional[delayedarray.DelayedArray]
    """Matrix of (log-)normalized expression values derived from ADT counts, as computed by :py:func:`~scranpy.normalize_counts.normalize_counts` using :py:attr:`~adt_size_factors`.
    If ADT data is not available, this is set to ``None`` instead."""

    crispr_size_factors: Optional[numpy.ndarray]
    """Size factors for the CRISPR count matrix, derived from the sum of counts for each cell.
    Size factors are centered with :py:func:`~scranpy.center_size_factors.center_size_factors`.
    If CRISPR data is not available, this is set to ``None`` instead."""

    normalized_crispr: Optional[delayedarray.DelayedArray]
    """Matrix of (log-)normalized expression values derived from CRISPR counts, as computed by :py:func:`~scranpy.normalize_counts.normalize_counts` using :py:attr:`~crispr_size_factors`.
    If CRISPR data is not available, this is set to ``None`` instead."""

    rna_gene_variances: Optional[ModelGeneVariancesResults]
    """Results of :py:func:`~scranpy.model_gene_variances.model_gene_variances`.
    If RNA data is not available, this is set to ``None`` instead."""

    rna_highly_variable_genes: Optional[numpy.ndarray]
    """Results of :py:func:`~scranpy.choose_highly_variable_genes.choose_highly_variable_genes`.
    If RNA data is not available, this is set to ``None`` instead."""

    rna_pca: Optional[RunPcaResults]
    """Results of calling :py:func:`~scranpy.run_pca.run_pca` on the RNA log-expression matrix.
    If RNA data is not available, this is set to ``None`` instead."""

    adt_pca: Optional[RunPcaResults]
    """Results of calling :py:func:`~scranpy.run_pca.run_pca` on the ADT log-expression matrix.
    If ADT data is not available, this is set to ``None`` instead."""

    crispr_pca: Optional[RunPcaResults]
    """Results of calling :py:func:`~scranpy.run_pca.run_pca` on the CRISPR log-expression matrix.
    If CRISPR data is not available, this is set to ``None`` instead."""

    combined_pca: Union[Literal["rna_pca", "adt_pca", "crispr_pca"], ScaleByNeighborsResults]
    """If only one modality is used for the downstream analysis, this is a string specifying the attribute containing the components to be used.
    If multiple modalities are to be combined for downstream analysis, this contains the results of :py:func:`~scranpy.scale_by_neighbors.scale_by_neighbors` on the PCs of those modalities."""

    batch_corrected: Optional[CorrectMnnResults]
    """Results of :py:func:`~scranpy.correct_mnn.correct_mnn`.
    If no blocking factor is supplied, this is set to ``None`` instead."""

    run_tsne: Optional[numpy.ndarray]
    """Results of :py:func:`~scranpy.run_tsne.run_tsne`.
    This is ``None`` if t-SNE was not performed."""

    run_umap: Optional[numpy.ndarray]
    """Results of :py:func:`~scranpy.run_umap.run_umap`. 
    This is ``None`` if UMAP was not performed."""

    snn_graph: Optional[GraphComponents]
    """Results of :py:func:`~scranpy.build_snn_graph.build_snn_graph`. 
    This is ``None`` if graph-based clustering was not performed."""

    graph_clusters: Optional[ClusterGraphResults]
    """Results of :py:func:`~scranpy.cluster_graph.cluster_graph`.
    This is ``None`` if graph-based clustering was not performed."""

    kmeans_clusters: Optional[ClusterGraphResults]
    """Results of :py:func:`~scranpy.cluster_kmeans.cluster_kmeans`.
    This is ``None`` if k-means clustering was not performed."""

    rna_score_markers: Optional[RunPcaResults]
    """Results of calling :py:func:`~scranpy.score_markers.score_markers` on the RNA log-expression matrix.
    If RNA data is not available, this is set to ``None`` instead.
    This will also be ``None`` if no suitable clusterings are available."""

    adt_score_markers: Optional[RunPcaResults]
    """Results of calling :py:func:`~scranpy.score_markers.score_markers` on the ADT log-expression matrix.
    If ADT data is not available, this is set to ``None`` instead.
    This will also be ``None`` if no suitable clusterings are available."""

    crispr_score_markers: Optional[RunPcaResults]
    """Results of calling :py:func:`~scranpy.score_markers.score_markers` on the CRISPR log-expression matrix.
    If CRISPR data is not available, this is set to ``None`` instead.
    This will also be ``None`` if no suitable clusterings are available."""


def analyze(
    x_rna: Optional[Any],
    x_adt: Optional[Any] = None,
    x_crispr: Optional[Any] = None,
    block: Optional[Sequence] = None,
    rna_subsets: Union[Mapping, Sequence] = [],
    adt_subsets: Union[Mapping, Sequence] = [],
    suggest_rna_qc_thresholds_options: dict = {},
    suggest_adt_qc_thresholds_options: dict = {},
    suggest_crispr_qc_thresholds_options: dict = {},
    filter_cells: bool = True,
    center_size_factors_options: dict = {},
    compute_clrm1_factors_options: dict = {},
    model_gene_variances_options: dict = {},
    choose_highly_variable_genes_options: dict = {},
    run_pca_options: dict = {},
    use_rna_pca: bool = True,
    use_adt_pca: bool = True,
    use_crispr_pca: bool = True,
    scale_by_neighbors_options: dict = {},
    correct_mnn_options: dict= {},
    run_umap_options: Optional[dict] = {},
    run_tsne_options: Optional[dict] = {},
    build_snn_graph_options: Optional[dict] = {},
    cluster_graph_options: dict = {},
    kmeans_clusters: Optional[int] = None,
    cluster_kmeans_options: dict = {},
    clusters_for_markers: list = ["graph", "kmeans"],
    score_markers_options: dict = {},
    nn_parameters: knncolle.Parameters = knncolle.AnnoyParameters(),
    num_threads: int = 3
) -> AnalyzeResults:
    """Run through a simple single-cell analysis pipeline, starting from a count matrix and ending with clusters, visualizations and markers.
    This also supports integration of multiple modalities and correction of batch effects.

    Args:
        x_rna:
            A matrix-like object containing RNA counts.
            This should have the same number of columns as the other ``x_*`` arguments.
            Alternatively ``None``, if no RNA counts are available.

        x_adt:
            A matrix-like object containing ADT counts.
            This should have the same number of columns as the other ``x_*`` arguments.
            Alternatively ``None``, if no ADT counts are available.

        x_crispr:
            A matrix-like object containing CRISPR counts.
            This should have the same number of columns as the other ``x_*`` arguments.
            Alternatively ``None``, if no CRISPR counts are available.

        block:
            Factor specifying the block of origin (e.g., batch, sample) for each cell in the ``x_*`` matrices.
            Alternatively ``None``, if all cells are from the same block.

        rna_subsets:
            Gene subsets for quality control, typically used for mitochondrial genes.
            Check out the ``subsets`` arguments in :py:func:`~scranpy.rna_quality_control.compute_rna_qc_metrics` for details.

        adt_subsets:
            ADT subsets for quality control, typically used for IgG controls.
            Check out the ``subsets`` arguments in :py:func:`~scranpy.adt_quality_control.compute_adt_qc_metrics` for details.

        suggest_rna_qc_thresholds_options:
            Arguments to pass to :py:func:`~scranpy.rna_quality_control.suggest_rna_qc_thresholds`.

        suggest_adt_qc_thresholds_options:
            Arguments to pass to :py:func:`~scranpy.adt_quality_control.suggest_adt_qc_thresholds`.

        suggest_crispr_qc_thresholds_options:
            Arguments to pass to :py:func:`~scranpy.crispr_quality_control.suggest_crispr_qc_thresholds`.

        filter_cells:
            Whether to filter the count matrices to only retain high-quality cells in all modalities.
            If ``False``, QC metrics and thresholds are still computed but are not used to filter the count matrices.

        center_size_factors_options:
            Arguments to pass to :py:func:`~scranpy.center_size_factors.center_size_factors`.

        compute_clrm1_factors_options:
            Arguments to pass to :py:func:`~scranpy.compute_clrm1_factors.compute_clrm1_factors`.

        model_gene_variances_options:
            Arguments to pass to :py:func:`~scranpy.model_gene_variances.model_gene_variances`.

        choose_highly_variable_genes_options:
            Arguments to pass to :py:func:`~scranpy.choose_highly_variable_genes.choose_highly_variable_genes`.

        run_pca_options:
            Arguments to pass to :py:func:`~scranpy.run_pca.run_pca`.

        use_rna_pca:
            Whether to use the RNA-derived PCs for downstream steps (i.e., clustering, visualization).

        use_adt_pca:
            Whether to use the ADT-derived PCs for downstream steps (i.e., clustering, visualization).

        use_crispr_pca:
            Whether to use the CRISPR-derived PCs for downstream steps (i.e., clustering, visualization).

        scale_by_neighbors_options:
            Arguments to pass to :py:func:`~scranpy.scale_by_neighbors.scale_by_neighbors`.
            Only used if multiple modalities are available and their corresponding ``use_*`` arguments are ``True``.

        correct_mnn_options:
            Arguments to pass to :py:func:`~scranpy.correct_mnn.correct_mnn`.
            Only used if ``block`` is supplied.

        run_umap_options:
            Arguments to pass to :py:meth:`~scranpy.run_umap.run_umap`.
            If ``None``, UMAP is not performed.

        run_tsne_options:
            Arguments to pass to :py:meth:`~scranpy.run_tsne.run_tsne`.
            If ``None``, t-SNE is not performed.

        build_snn_graph_options:
            Arguments to pass to :py:meth:`~scranpy.build_snn_graph.build_snn_graph`.
            Ignored if ``cluster_graph_options = None``.

        cluster_graph_options:
            Arguments to pass to :py:meth:`~scranpy.cluster_graph.cluster_graph`.
            If ``None``, graph-based clustering is not performed.

        kmeans_clusters:
            Number of clusters to use in k-means clustering.
            If ``None``, k-means clustering is not performed.

        cluster_kmeans_options:
            Arguments to pass to :py:meth:`~scranpy.cluster_kmeans.cluster_kmeans`.
            Ignored if ``kmeans_clusters = None``.

        clusters_for_markers:
            List of clustering algorithms (either ``graph`` or ``kmeans``), specifying the clustering to be used for marker detection.
            The first available clustering will be chosen.

        score_markers_options:
            Arguments to pass to :py:meth:`~scranpy.score_markers.score_markers`.
            Ignored if no suitable clusterings are available.

        nn_parameters:
            Algorithm to use for nearest-neighbor searches in the various steps.

        num_threads:
            Number of threads to use in each step.

    Returns:
        The results of the entire analysis, including the results from each step.
    """
    store = {}
    all_ncols = set() 
    if not x_rna is None:
        all_ncols.add(x_rna.shape[1])
    if not x_adt is None:
        all_ncols.add(x_adt.shape[1])
    if not x_crispr is None:
        all_ncols.add(x_crispr.shape[1])
    if len(all_ncols) > 1:
        raise ValueError("'x_*' have differing numbers of columns")
    if len(all_ncols) == 0:
        raise ValueError("at least one 'x_*' must be non-None")
    ncells = list(all_ncols)[0]

    ############ Quality control o(*°▽°*)o #############

    if not x_rna is None:
        store["rna_qc_metrics"] = compute_rna_qc_metrics(x_rna, subsets=rna_subsets, num_threads=num_threads)
        store["rna_qc_thresholds"] = suggest_rna_qc_thresholds(store["rna_qc_metrics"], block=block, **suggest_rna_qc_thresholds_options)
        store["rna_qc_filter"] = filter_rna_qc_metrics(store["rna_qc_threshold"], store["rna_qc_metrics"], block=block)
    else:
        store["rna_qc_metrics"] = None
        store["rna_qc_thresholds"] = None
        store["rna_qc_filter"] = None

    if not x_adt is None:
        store["adt_qc_metrics"] = compute_adt_qc_metrics(x_adt, subsets=adt_subsets, num_threads=num_threads)
        store["adt_qc_thresholds"] = suggest_adt_qc_thresholds(store["adt_qc_metrics"], block=block, **suggest_adt_qc_thresholds_options)
        store["adt_qc_filter"] = filter_adt_qc_metrics(store["adt_qc_thresholds"], store["adt_qc_metrics"], block=block)
    else:
        store["adt_qc_metrics"] = None
        store["adt_qc_thresholds"] = None
        store["adt_qc_filter"] = None

    if not x_crispr is None:
        store["crispr_qc_metrics"] = compute_crispr_qc_metrics(x_crispr, num_threads=num_threads)
        store["crispr_qc_thresholds"] = suggest_crispr_qc_thresholds(store["crispr_qc_metrics"], block=block, **suggest_crispr_qc_thresholds_options)
        store["crispr_qc_filter"] = filter_crispr_qc_metrics(store["crispr_qc_threshold"], store["crispr_qc_metrics"], block=block)
    else:
        store["crispr_qc_metrics"] = None
        store["crispr_qc_thresholds"] = None
        store["crispr_qc_filter"] = None

    # Combining all filters.
    combined_filter = numpy.full((ncells,), True)
    for mod in ["rna", "adt", "crispr"]:
        modality_filter = store[mod + "_qc_filter"]
        if not modality_filter is None:
            combined_filter = numpy.logical_and(combined_filter, modality_filter)
    store["combined_filter"] = combined_filter

    if filter_cells:
        if not x_rna is None:
            store["filtered_rna"] = delayedarray.DelayedArray(x_rna)[:,combined_filter]
        if not x_adt is None:
            store["filtered_adt"] = delayedarray.DelayedArray(x_adt)[:,combined_filter]
        if not x_crispr is None:
            store["filtered_crispr"] = delayedarray.DelayedArray(x_crispr)[:,combined_filter]
        if not block is None:
            block = biocutils.subset_sequence(block, numpy.where(combined_filter)[0])
    else:
        if not x_rna is None:
            store["filtered_rna"] = delayedarray.DelayedArray(x_rna)
        if not x_adt is None:
            store["filtered_adt"] = delayedarray.DelayedArray(x_adt)
        if not x_crispr is None:
            store["filtered_crispr"] = deleyedarray.DelayedArray(x_crispr)

    ############ Normalization ( ꈍᴗꈍ) #############

    if not x_rna is None:
        store["rna_size_factors"] = center_size_factors(store["rna_qc_metrics"].sum[combined_filter], block=block, **center_size_factors_options)
        store["normalized_rna"] = normalize_counts(store["filtered_rna"], store["rna_size_factors"], **normalize_counts_options)
    else:
        store["rna_size_factors"] = None
        store["normalized_rna"] = None

    if not x_adt is None:
        raw_adt_sf = compute_clrm1_factors(store["filtered_adt"], **compute_clrm1_factors, num_threads=num_threads)
        store["adt_size_factors"] = center_size_factors(raw_adt_sf, block=block, **center_size_factors_options)
        store["normalized_adt"] = normalize_counts(store["filtered_adt"], store["adt_size_factors"], **normalize_counts_options)
    else:
        store["adt_size_factors"] = None
        store["normalized_adt"] = None

    if not x_crispr is None:
        store["crispr_size_factors"] = center_size_factors(store["crispr_qc_metrics"].sum[combined_filter], block=block, **center_size_factors_options)
        store["normalized_crispr"] = normalize_counts(store["filtered_crispr"], store["crispr_size_factors"], **normalize_counts_options)
    else:
        store["crispr_size_factors"] = None
        store["normalized_crispr"] = None

    ############ Variance modelling (～￣▽￣)～ #############

    if not x_rna is None:
        store["rna_gene_variances"] = model_gene_variances(store["normalized_rna"], block=block, **model_gene_variances_options, num_threads=num_threads)
        store["rna_highly_variable_genes"] = choose_highly_variable_genes(store["rna_gene_variances"].residuals, **choose_highly_variable_genes_options)
    else:
        store["rna_gene_variances"] = None
        store["rna_highly_variable_genes"] = None

    ############ Principal components analysis \(>⩊<)/ #############

    if not x_rna is None:
        store["rna_pca"] = run_pca(store["normalized_rna"][store["rna_highly_variable_genes"],:], **run_pca_options, block=block, num_threads=num_threads)
    else:
        store["rna_pca"] = None

    if not x_adt is None:
        store["adt_pca"] = run_pca(store["normalized_adt"], **run_pca_options, block=block, num_threads=num_threads)
    else:
        store["adt_pca"] = None

    if not x_crispr is None:
        store["crispr_pca"] = run_pca(store["normalized_crispr"], **run_pca_options, block=block, num_threads=num_threads)
    else:
        store["crispr_pca"] = None

    ############ Combining modalities and batches („• ᴗ •„) #############

    embeddings = []
    if use_rna_pca and not x_rna is None:
        embeddings.push("rna_pca")
    if use_adt_pca and not x_adt is None:
        embeddings.push("adt_pca")
    if use_crispr_pca and not x_crispr is None:
        embeddings.push("crispr_pca")

    if len(embeddings) == 0:
        raise ValueError("at least one 'use_*' must be True")
    if len(embeddings) == 1:
        store["combined_pca"] = embeddings[0]
    else:
        embeddings = [store[e].components for e in embeddings]
        store["combined_pca"] = scale_by_neighbors(embeddings, **scale_by_neighbors_options, nn_parameters=nn_parameters, num_threads=num_threads)

    if isinstance(store["combined_pca"], str):
        chosen_pcs = store[store["combined_pca"]].components
    else:
        chosen_pcs = store["combined_pca"].combined

    if block is None:
        store["batch_corrected"] = None
    else:
        store["batch_corrected"] = correct_mnn(chosen_pcs, block=block, **correct_mnn_options, nn_parameters=nn_parameters, num_threads=num_threads)
        chosen_pcs = store["batch_corrected"].corrected

    ############ Assorted neighbor-related stuff ⸜(⸝⸝⸝´꒳`⸝⸝⸝)⸝ #############

    all_out = run_all_neighbor_steps(
        chosen_pcs,
        run_umap_options=run_umap_options, 
        run_tsne_options=run_tsne_options, 
        build_snn_graph_options=build_snn_graph_options,
        cluster_graph_options=cluster_graph_options,
        nn_parameters=nn_parameters,
        **run_all_neighbor_steps_options,
        num_threads=num_threads
    )

    store["tsne"] = all_out.run_tsne
    store["umap"] = all_out.run_umap
    store["snn_graph"] = all_out.build_snn_graph
    store["graph_clusters"] = all_out.cluster_graph

    ############ Finding markers ⸜(⸝⸝⸝´꒳`⸝⸝⸝)⸝ #############

    if not kmeans_clusters is None:
        store["kmeans_clusters"] = cluster_kmeans(chosen_pcs, k=kmeans_clusters, **cluster_kmeans_options, num_threads=num_threads)
    else:
        store["kmeans_clusters"] = None

    chosen_clusters = None
    for c in clusters_for_markers:
        if c == "graph":
            chosen_clusters = store["graph_clusters"].membership
            break
        elif c == "kmeans":
            chosen_clusters = store["kmeans_clusters"].clusters
            break

    store["rna_markers"] = None
    store["adt_markers"] = None
    store["crispr_markers"] = None

    if not chosen_clusters is None:
        if not x_rna is None:
            store["rna_markers"] = score_markers(x["normalized_rna"], groups=chosen_clusters, num_threads=num_threads, block=block, **score_markers_options)
        if not x_adt is None:
            store["adt_markers"] = score_markers(x["normalized_adt"], groups=chosen_clusters, num_threads=num_threads, block=block, **score_markers_options)
        if not x_crispr is None:
            store["crispr_markers"] = score_markers(x["normalized_crispr"], groups=chosen_clusters, num_threads=num_threads, block=block, **score_markers_options)

    return AnalyzeResults(**store)
