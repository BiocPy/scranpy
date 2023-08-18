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
    """Miscellaneous options for :py:meth:`~scranpy._analyze.live_analyze.analyze`.

    Attributes:
        snn_graph_multilevel_resolution (float):
            Resolution to use for multi-level clustering of the SNN graph.

        mito_prefix (Union[bool, str]):
            Prefix for mitochondrial genes, under the assumption that the feature names are gene symbols.
            If True, the default prefix in :py:meth:`~scranpy.quality_control.rna.guess_mito_from_symbols` is used.
            If False, no attempt is made to guess the identities of mitochondrial genes.
    """

    snn_graph_multilevel_resolution: int = 1
    mito_prefix: Union[bool, str] = "mt-"

@dataclass
class AnalyzeOptions:
    """Optional parameters for all :py:meth:`~scranpy._analyze.live_analyze.analyze` steps.

    Optional parameters for each function are named after the function with the ``_options`` suffix.
    In most cases, these can be modified directly to refine the behavior of the
    :py:meth:`~scranpy._analyze.live.analyze` function.

    However, for a few options, it makes more sense to set them across multiple parameter objects simultaneously.
    We provide the following methods in such cases:
    
    - :py:meth:`~scranpy._analyze.AnalyzeOptions.set_seed`, to set the random seed. 
    - :py:meth:`~scranpy._analyze.AnalyzeOptions.set_block`, to set the block assignments.
    - :py:meth:`~scranpy._analyze.AnalyzeOptions.set_threads`, to set the number of threads.
    - :py:meth:`~scranpy._analyze.AnalyzeOptions.set_verbose`, to set the verbosity level.
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

    log_norm_counts_options: norm.LogNormCountsOptions = field(
        default_factory=qc.LogNormCountsOptions
    )

    choose_hvgs_options: feat.ChooseHvgsOptions = field(
        default_factory=feat.ChooseHvgsOptions
    )

    model_gene_variances_options: feat.ModelGeneVariancesOptions = field(
        default_factory=feat.ModelGeneVariancesOptions
    )

    run_pca_options: dimred.RunPcaOptions = field(
        default_factory=dimred.RunPcaOptions
    )

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
        default_factory=clust.build_snn_graph_options
    )

    score_markers_options: mark.ScoreMarkersOptions = field(
        default_factory=mark.ScoreMarkersOptions
    )

    miscellaneous_options: MiscellaneousOptions = field(
        default_factory=MiscellaneousOptions
    )

    # Multi-step setters.
    def set_seed(self, seed: int = 42):
        """Set seed for RNG.

        Args:
            seed (int, optional): Seed for random number generation.
                Defaults to 42.
        """
        self.run_pca_options.set_seed(seed)
        self.run_tsne_options.set_seed(seed)
        self.run_umap_options.set_seed(seed)

    def set_block(self, block: Optional[Sequence] = None):
        """Set block assignments for each cell.

        Args:
            block (Sequence, optional): Blocks assignments
                for each cell. Defaults to None.
        """
        self.suggest_rna_qc_filters_options.set_block(block)
        self.create_rna_qc_filter_options.set_block(block)
        self.log_norm_counts_options.set_block(block)
        self.model_gene_variances_options.set_block(block)
        self.run_pca_options.set_block(block)
        self.score_markers_options.set_block(block)

    def set_verbose(self, verbose: bool = False):
        """Set verbose to display logs.

        Args:
            verbose (bool, optional): Whether to print logs.
                Defaults to False.
        """
        self.per_cell_rna_qc_metrics_options.set_verbose(verbose)
        self.suggest_rna_qc_filters_options.set_verbose(verbose)
        self.create_rna_qc_filter_options.set_verbose(verbose)
        self.filter_cells_options.set_verbose(verbose)
        self.log_norm_counts_options.set_verbose(verbose)
        self.choose_hvgs_options.set_verbose(verbose)
        self.model_gene_variances_options.set_verbose(verbose)
        self.run_pca_options.set_verbose(verbose)
        self.run_tsne_options.set_verbose(verbose)
        self.run_umap_options.set_verbose(verbose)
        self.build_snn_graph_options.set_verbose(verbose)
        self.score_markers_options.set_verbose(verbose)

    def set_threads(self, num_threads: int = 1):
        """Set number of threads to use.

        Args:
            num_threads (int, optional): Number of threads. Defaults to 1.
        """
        self.per_cell_rna_qc_metrics_options.set_threads(num_threads)
        self.log_norm_counts_options.set_threads(num_threads)
        self.choose_hvgs_options.set_threads(num_threads)
        self.model_gene_variances_options.set_threads(num_threads)
        self.run_pca_options.set_threads(num_threads)
        self.run_tsne_options.set_threads(num_threads)
        self.run_umap_options.set_threads(num_threads)
        self.build_snn_graph_options.set_threads(num_threads)
        self.score_markers_options.set_threads(num_threads)
