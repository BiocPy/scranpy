from dataclasses import dataclass, field
from typing import Optional, Sequence

from .. import batch_correction as correct
from .. import clustering as clust
from .. import dimensionality_reduction as dimred
from .. import feature_selection as feat
from .. import marker_detection as mark
from .. import nearest_neighbors as nn
from .. import normalization as norm
from .. import quality_control as qc


@dataclass
class MiscellaneousOptions:
    """Miscellaneous options for :py:meth:`~scranpy.analyze.analyze.analyze`.

    Attributes:
        cell_names:
            Names for all cells in the dataset, to be added to any per-cell data frames.
            This should have the same length as the number of columns in each data matrix.

        rna_feature_names:
            Names for all features in the RNA data.
            This should have the same length as the number of rows in the RNA count matrix.

        adt_feature_names:
            Names for all tags in the ADT data.
            This should have the same length as the number of rows in the ADT count matrix.

        crispr_feature_names:
            Names for all guides in the CRISPR data.
            This should have the same length as the number of rows in the CRISPR count matrix.

        filter_on_rna_qc:
            Whether to filter cells on the RNA-based quality control metrics,
            when RNA data is available.

        filter_on_adt_qc:
            Whether to filter cells on the ADT-based quality control metrics,
            when ADT data is available.

        filter_on_crispr_qc:
            Whether to filter cells on the CRISPR-based quality control metrics,
            when CRISPR data is available.

        snn_graph_multilevel_resolution:
            Resolution to use for multi-level clustering of the SNN graph.

        block:
            Block assignment for each cell.
            This should have length equal to the total number of cells in the dataset, before any quality control
            is applied.
    """

    cell_names: Optional[Sequence[str]] = None
    rna_feature_names: Optional[Sequence[str]] = None
    adt_feature_names: Optional[Sequence[str]] = None
    crispr_feature_names: Optional[Sequence[str]] = None
    filter_on_rna_qc: bool = True
    filter_on_adt_qc: bool = True
    filter_on_crispr_qc: bool = True
    snn_graph_multilevel_resolution: int = 1
    block: Optional[Sequence] = None


@dataclass
class AnalyzeOptions:
    """Optional parameters for all :py:meth:`~scranpy.analyze.analyze.analyze` steps.

    Optional parameters for each function are named after the function with the ``_options`` suffix.
    In most cases, these can be modified directly to refine the behavior of the
    :py:meth:`~scranpy.analyze.analyze.analyze` function. However, for a few options, it usually makes more sense
    to set them across multiple parameter objects simultaneously;
    check out the setter methods of this class for more details.

    Attributes:
        per_cell_rna_qc_metrics_options:
            Options to pass to :py:meth:`~scranpy.quality_control.per_cell_rna_qc_metrics.per_cell_rna_qc_metrics`.

        suggest_rna_qc_filters_options:
            Options to pass to :py:meth:`~scranpy.quality_control.suggest_rna_qc_filters.suggest_rna_qc_filters`.

        create_rna_qc_filter_options:
            Options to pass to :py:meth:`~scranpy.quality_control.create_rna_qc_filter.create_rna_qc_filter`.

        per_cell_adt_qc_metrics_options:
            Options to pass to :py:meth:`~scranpy.quality_control.per_cell_adt_metrics.per_cell_adt_qc_metrics`.

        suggest_adt_qc_filters_options:
            Options to pass to :py:meth:`~scranpy.quality_control.suggest_adt_qc_filters.suggest_adt_qc_filters`.

        create_adt_qc_filter_options:
            Options to pass to :py:meth:`~scranpy.quality_control.create_adt_qc_filter.create_adt_qc_filter`.

        per_cell_crispr_qc_metrics_options:
            Options to pass to
            :py:meth:`~scranpy.quality_control.per_cell_crispr_qc_metrics.per_cell_crispr_qc_metrics`.

        suggest_crispr_qc_filters_options:
            Options to pass to :py:meth:`~scranpy.quality_control.suggest_crispr_qc_filters.suggest_crispr_qc_filters`.

        create_crispr_qc_filter_options:
            Options to pass to :py:meth:`~scranpy.quality_control.create_crispr_qc_filter.create_crispr_qc_filter`.

        filter_cells_options:
            Options to pass to :py:meth:`~scranpy.quality_control.filter_cells.filter_cells`.

        rna_log_norm_counts_options:
            Options to pass to :py:meth:`~scranpy.normalization.log_norm_counts.log_norm_counts`
            for the RNA count matrix.

        grouped_size_factors_options:
            Options to pass to :py:meth:`~scranpy.normalization.grouped_size_factors.grouped_size_factors`
            to compute ADT size factors.

        adt_log_norm_counts_options:
            Options to pass to :py:meth:`~scranpy.normalization.log_norm_counts.log_norm_counts`
            for the ADT count matrix.

        crispr_log_norm_counts_options:
            Options to pass to :py:meth:`~scranpy.normalization.log_norm_counts.log_norm_counts`
            for the CRISPR count matrix.

        choose_hvgs_options:
            Options to pass to :py:meth:`~scranpy.feature_selection.choose_hvgs.choose_hvgs`
            to choose highly variable genes for the RNA data.

        model_gene_variances_options:
            Options to pass to :py:meth:`~scranpy.feature_selection.model_gene_variances.model_gene_variances`
            to model per-gene variances for the RNA data.

        rna_run_pca_options:
            Options to pass to :py:meth:`~scranpy.dimensionality_reduction.run_pca.run_pca`
            for the RNA log-expression matrix.

        adt_run_pca_options:
            Options to pass to :py:meth:`~scranpy.dimensionality_reduction.run_pca.run_pca`
            for the ADT log-expression matrix.

        crispr_run_pca_options:
            Options to pass to :py:meth:`~scranpy.dimensionality_reduction.run_pca.run_pca`
            for the CRISPR log-expression matrix.

        mnn_correct_options:
            Options to pass to :py:meth:`~scranpy.batch_correction.mnn_correct.mnn_correct`.

        build_neighbor_index_options:
            Options to pass to :py:meth:`~scranpy.nearest_neighbors.build_neighbor_index.build_neighbor_index`.

        find_nearest_neighbors_options:
            Options to pass to :py:meth:`~scranpy.nearest_neighbors.find_nearest_neighbors.find_nearest_neighbors`.

        run_tsne_options:
            Options to pass to :py:meth:`~scranpy.dimensionality_reduction.run_tsne.run_tsne`.

        run_umap_options:
            Options to pass to :py:meth:`~scranpy.dimensionality_reduction.run_umap.run_umap`.

        build_snn_graph_options:
            Options to pass to :py:meth:`~scranpy.clustering.build_snn_graph.build_snn_graph`.

        rna_score_markers_options:
            Options to pass to :py:meth:`~scranpy.marker_detection.score_markers.score_markers`
            for the RNA log-expression values.

        adt_score_markers_options:
            Options to pass to :py:meth:`~scranpy.marker_detection.score_markers.score_markers`
            for the ADT log-abundances.

        crispr_score_markers_options:
            Options to pass to :py:meth:`~scranpy.marker_detection.score_markers.score_markers`
            for the CRISPR log-abundances.

        miscellaneous_options:
            Further options that are not associated with any single function call.
    """

    per_cell_rna_qc_metrics_options: qc.PerCellRnaQcMetricsOptions = field(
        default_factory=qc.PerCellRnaQcMetricsOptions
    )

    suggest_rna_qc_filters_options: qc.SuggestRnaQcFiltersOptions = field(
        default_factory=qc.SuggestRnaQcFiltersOptions
    )

    create_rna_qc_filter_options: qc.CreateRnaQcFilterOptions = field(
        default_factory=qc.CreateRnaQcFilterOptions
    )

    per_cell_adt_qc_metrics_options: qc.PerCellAdtQcMetricsOptions = field(
        default_factory=qc.PerCellAdtQcMetricsOptions
    )

    suggest_adt_qc_filters_options: qc.SuggestAdtQcFiltersOptions = field(
        default_factory=qc.SuggestAdtQcFiltersOptions
    )

    create_adt_qc_filter_options: qc.CreateAdtQcFilterOptions = field(
        default_factory=qc.CreateAdtQcFilterOptions
    )

    per_cell_crispr_qc_metrics_options: qc.PerCellCrisprQcMetricsOptions = field(
        default_factory=qc.PerCellCrisprQcMetricsOptions
    )

    suggest_crispr_qc_filters_options: qc.SuggestCrisprQcFiltersOptions = field(
        default_factory=qc.SuggestCrisprQcFiltersOptions
    )

    create_crispr_qc_filter_options: qc.CreateCrisprQcFilterOptions = field(
        default_factory=qc.CreateCrisprQcFilterOptions
    )

    filter_cells_options: qc.FilterCellsOptions = field(
        default_factory=qc.FilterCellsOptions
    )

    rna_log_norm_counts_options: norm.LogNormCountsOptions = field(
        default_factory=norm.LogNormCountsOptions
    )

    grouped_size_factors_options: norm.GroupedSizeFactorsOptions = field(
        default_factory=norm.GroupedSizeFactorsOptions
    )

    adt_log_norm_counts_options: norm.LogNormCountsOptions = field(
        default_factory=norm.LogNormCountsOptions
    )

    crispr_log_norm_counts_options: norm.LogNormCountsOptions = field(
        default_factory=norm.LogNormCountsOptions
    )

    choose_hvgs_options: feat.ChooseHvgsOptions = field(
        default_factory=feat.ChooseHvgsOptions
    )

    model_gene_variances_options: feat.ModelGeneVariancesOptions = field(
        default_factory=feat.ModelGeneVariancesOptions
    )

    rna_run_pca_options: dimred.RunPcaOptions = field(
        default_factory=dimred.RunPcaOptions
    )

    adt_run_pca_options: dimred.RunPcaOptions = field(
        default_factory=dimred.RunPcaOptions
    )

    crispr_run_pca_options: dimred.RunPcaOptions = field(
        default_factory=dimred.RunPcaOptions
    )

    combine_embeddings_options: dimred.CombineEmbeddingsOptions = field(
        default_factory=dimred.CombineEmbeddingsOptions
    )

    mnn_correct_options: correct.MnnCorrectOptions = field(
        default_factory=correct.MnnCorrectOptions
    )

    build_neighbor_index_options: nn.BuildNeighborIndexOptions = field(
        default_factory=nn.BuildNeighborIndexOptions
    )

    run_tsne_options: dimred.RunTsneOptions = field(
        default_factory=dimred.RunTsneOptions
    )

    run_umap_options: dimred.RunUmapOptions = field(
        default_factory=dimred.RunUmapOptions
    )

    find_nearest_neighbors_options: nn.FindNearestNeighborsOptions = field(
        default_factory=nn.FindNearestNeighborsOptions
    )

    build_snn_graph_options: clust.BuildSnnGraphOptions = field(
        default_factory=clust.BuildSnnGraphOptions
    )

    rna_score_markers_options: mark.ScoreMarkersOptions = field(
        default_factory=mark.ScoreMarkersOptions
    )

    adt_score_markers_options: mark.ScoreMarkersOptions = field(
        default_factory=mark.ScoreMarkersOptions
    )

    crispr_score_markers_options: mark.ScoreMarkersOptions = field(
        default_factory=mark.ScoreMarkersOptions
    )

    miscellaneous_options: MiscellaneousOptions = field(
        default_factory=MiscellaneousOptions
    )

    # Multi-step setters.
    def set_seed(self, seed: int = 42):
        """Set seed for RNG. This calls the method of the same name for
        :py:meth:`~scranpy.dimensionality_reduction.run_pca.run_pca`,
        :py:meth:`~scranpy.dimensionality_reduction.run_tsne.run_tsne`,
        :py:meth:`~scranpy.dimensionality_reduction.run_umap.run_umap`.

        Args:
            seed:
                Seed for random number generation.
        """
        self.run_pca_options.set_seed(seed)
        self.run_tsne_options.set_seed(seed)
        self.run_umap_options.set_seed(seed)

    def set_threads(self, num_threads: int = 1):
        """Set number of threads to use. This calls the method of the same name for
        :py:meth:`~scranpy.quality_control.rna.per_cell_rna_qc_metrics`,
        :py:meth:`~scranpy.normalization.log_norm_counts.log_norm_counts.`,
        :py:meth:`~scranpy.feature_selection.model_gene_variances`,
        :py:meth:`~scranpy.dimensionality_reduction.run_pca.run_pca`,
        :py:meth:`~scranpy.nearest_neighbors.find_nearest_neighbors.find_nearest_neighbors`.

        The number of threads provided to
        :py:meth:`~scranpy.nearest_neighbors.find_nearest_neighbors.find_nearest_neighbors`
        is also used in :py:meth:`~scranpy.analyze.run_neighbor_suite`,
        which determines the thread allocation to subsequent steps like
        :py:meth:`~scranpy.dimensionality_reduction.run_tsne.run_tsne`
        and
        :py:meth:`~scranpy.marker_detection.score_markers.score_markers`.
        In all cases, thread utilization will not exceed the limit specified here in ``num_threads``.

        Args:
            num_threads:
                Number of threads. Defaults to 1.
        """
        self.per_cell_rna_qc_metrics_options.set_threads(num_threads)
        self.log_norm_counts_options.set_threads(num_threads)
        self.choose_hvgs_options.set_threads(num_threads)
        self.model_gene_variances_options.set_threads(num_threads)
        self.run_pca_options.set_threads(num_threads)
        self.find_nearest_neighbors_options.set_threads(num_threads)
        self.grouped_size_factors_options.set_threads(num_threads)
