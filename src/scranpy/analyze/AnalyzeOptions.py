from dataclasses import dataclass, field
from typing import Optional, Sequence

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
        snn_graph_multilevel_resolution (float): Resolution to use for multi-level
            clustering of the SNN graph.

        mito_prefix (str, Optional): Prefix for mitochondrial genes, under the assumption that the feature names
            are gene symbols. If None, no attempt is made to guess the identities of mitochondrial genes.

        block (Sequence, optional): Block assignment for each cell. This should have length
            equal to the total number of cells in the dataset, before any quality control is applied.
    """

    snn_graph_multilevel_resolution: int = 1
    mito_prefix: Optional[str] = "mt-"
    block: Optional[Sequence] = None


@dataclass
class AnalyzeOptions:
    """Optional parameters for all :py:meth:`~scranpy.analyze.analyze.analyze` steps.

    Optional parameters for each function are named after the function with the
    ``_options`` suffix. In most cases, these can be modified directly to refine the behavior of the
    :py:meth:`~scranpy.analyze.analyze.analyze` function. However, for a few options, it usually makes
    more sense to set them across multiple parameter objects simultaneously;
    check out the setter methods of this class for more details.

    Attributes:
        per_cell_rna_qc_metrics_options (PerCellRnaQcMetricsOptions):
            Options to pass to :py:meth:`~scranpy.quality_control.rna.per_cell_rna_qc_metrics`.

        suggest_rna_qc_filters_options (SuggestRnaQcFiltersOptions):
            Options to pass to :py:meth:`~scranpy.quality_control.rna.suggest_rna_qc_filters`.

        create_rna_qc_filter_options (CreateRnaQcFilterOptions):
            Options to pass to :py:meth:`~scranpy.quality_control.rna.create_rna_qc_filter`.

        filter_cells_options (FilterCellsOptions):
            Options to pass to :py:meth:`~scranpy.quality_control.filter_cells.filter_cells`.

        center_size_factors_options (CenterSizeFactorsOptions):
            Options to pass to :py:meth:`~scranpy.normalization.center_size_factors.center_size_factors`.
            Ignored if ``log_norm_counts_options.size_factors`` is set.

        log_norm_counts_options (LogNormCountsOptions):
            Options to pass to :py:meth:`~scranpy.normalization.log_norm_counts.log_norm_counts`.

        choose_hvgs_options (ChooseHvgsOptions):
            Options to pass to :py:meth:`~scranpy.feature_selection.choose_hvgs.choose_hvgs`.

        model_gene_variances_options (ModelGeneVariancesOptions):
            Options to pass to :py:meth:`~scranpy.feature_selection.model_gene_variances.model_gene_variances`.

        run_pca_options (RunPcaOptions):
            Options to pass to :py:meth:`~scranpy.dimensionality_reduction.run_pca.run_pca`.

        build_neighbor_index_options (BuildNeighborIndexOptions):
            Options to pass to :py:meth:`~scranpy.nearest_neighbors.build_neighbor_index.build_neighbor_index`.

        find_nearest_neighbors_options (FindNearestNeighborsOptions):
            Options to pass to :py:meth:`~scranpy.nearest_neighbors.find_nearest_neighbors.find_nearest_neighbors`.

        run_tsne_options (RunTsneOptions):
            Options to pass to :py:meth:`~scranpy.dimensionality_reduction.run_tsne.run_tsne`.

        run_umap_options (RunUmapOptions):
            Options to pass to :py:meth:`~scranpy.dimensionality_reduction.run_umap.run_umap`.

        build_snn_graph_options (BuildSnnGraphOptions):
            Options to pass to :py:meth:`~scranpy.clustering.build_snn_graph.build_snn_graph`.

        score_markers_options (ScoreMarkersOptions):
            Options to pass to :py:meth:`~scranpy.marker_detection.score_markers.score_markers`.

        miscellaneous_options (MiscellaneousOptions):
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

    filter_cells_options: qc.FilterCellsOptions = field(
        default_factory=qc.FilterCellsOptions
    )

    center_size_factors_options: norm.CenterSizeFactorsOptions = field(
        default_factory=norm.CenterSizeFactorsOptions
    )

    log_norm_counts_options: norm.LogNormCountsOptions = field(
        default_factory=norm.LogNormCountsOptions
    )

    choose_hvgs_options: feat.ChooseHvgsOptions = field(
        default_factory=feat.ChooseHvgsOptions
    )

    model_gene_variances_options: feat.ModelGeneVariancesOptions = field(
        default_factory=feat.ModelGeneVariancesOptions
    )

    run_pca_options: dimred.RunPcaOptions = field(default_factory=dimred.RunPcaOptions)

    run_tsne_options: dimred.RunTsneOptions = field(
        default_factory=dimred.RunTsneOptions
    )

    run_umap_options: dimred.RunUmapOptions = field(
        default_factory=dimred.RunUmapOptions
    )

    build_neighbor_index_options: nn.BuildNeighborIndexOptions = field(
        default_factory=nn.BuildNeighborIndexOptions
    )

    find_nearest_neighbors_options: nn.FindNearestNeighborsOptions = field(
        default_factory=nn.FindNearestNeighborsOptions
    )

    build_snn_graph_options: clust.BuildSnnGraphOptions = field(
        default_factory=clust.BuildSnnGraphOptions
    )

    score_markers_options: mark.ScoreMarkersOptions = field(
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
            seed (int, optional):
                Seed for random number generation.
        """
        self.run_pca_options.set_seed(seed)
        self.run_tsne_options.set_seed(seed)
        self.run_umap_options.set_seed(seed)

    def set_verbose(self, verbose: bool = False):
        """Set verbose to display logs. This calls the method of the same name for all ``*_options`` objects.

        Args:
            verbose (bool, optional): Whether to print logs.
                Defaults to False.
        """
        self.per_cell_rna_qc_metrics_options.set_verbose(verbose)
        self.suggest_rna_qc_filters_options.set_verbose(verbose)
        self.create_rna_qc_filter_options.set_verbose(verbose)
        self.filter_cells_options.set_verbose(verbose)
        self.center_size_factors_options.set_verbose(verbose)
        self.log_norm_counts_options.set_verbose(verbose)
        self.choose_hvgs_options.set_verbose(verbose)
        self.model_gene_variances_options.set_verbose(verbose)
        self.run_pca_options.set_verbose(verbose)
        self.run_tsne_options.set_verbose(verbose)
        self.run_umap_options.set_verbose(verbose)
        self.build_snn_graph_options.set_verbose(verbose)
        self.score_markers_options.set_verbose(verbose)

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
            num_threads (int, optional): Number of threads. Defaults to 1.
        """
        self.per_cell_rna_qc_metrics_options.set_threads(num_threads)
        self.log_norm_counts_options.set_threads(num_threads)
        self.choose_hvgs_options.set_threads(num_threads)
        self.model_gene_variances_options.set_threads(num_threads)
        self.run_pca_options.set_threads(num_threads)
        self.find_nearest_neighbors.set_threads(num_threads)
