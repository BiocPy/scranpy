from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import singledispatch
from typing import Any, Mapping, Optional, Sequence, Union

from biocframe import BiocFrame
from mattress import tatamize
from singlecellexperiment import SingleCellExperiment

from . import clustering as clust
from . import dimensionality_reduction as dimred
from . import feature_selection as feat
from . import marker_detection as mark
from . import nearest_neighbors as nn
from . import normalization as norm
from . import quality_control as qc
from .types import is_matrix_expected_type, validate_object_type

__author__ = "ltla, jkanche"
__copyright__ = "ltla"
__license__ = "MIT"


@dataclass
class AnalyzeOptions:
    quality_control: qc.RnaQualityControlOptions = qc.RnaQualityControlOptions()
    normalization: norm.NormalizationStepOptions = norm.NormalizationStepOptions()
    feature_selection: feat.FeatureSelectionStepOptions = (
        feat.FeatureSelectionStepOptions()
    )
    dimensionality_reduction: dimred.DimensionalityReductionStepOptions = (
        dimred.DimensionalityReductionStepOptions()
    )
    clustering: clust.ClusterStepOptions = clust.ClusterStepOptions()
    nearest_neighbors: nn.NearestNeighborStepOptions = nn.NearestNeighborStepOptions()
    marker_detection: mark.MarkerDetectionStepOptions = (
        mark.MarkerDetectionStepOptions()
    )
    block: Optional[Sequence] = None
    seed: int = 42
    num_threads: int = 1
    verbose: bool = False

    def __post_init__(self):
        validate_object_type(self.quality_control, qc.RnaQualityControlOptions)
        validate_object_type(self.normalization, norm.NormalizationStepOptions)
        validate_object_type(self.feature_selection, feat.FeatureSelectionStepOptions)
        validate_object_type(
            self.dimensionality_reduction, dimred.DimensionalityReductionStepOptions
        )
        validate_object_type(self.clustering, clust.ClusterStepOptions)
        validate_object_type(self.nearest_neighbors, nn.NearestNeighborStepOptions)
        validate_object_type(self.marker_detection, mark.MarkerDetectionStepOptions)

        if self.block is not None:
            self.set_block(block=self.block)

        if self.seed != 42:
            self.set_seed(seed=self.seed)

        if self.num_threads != 1:
            self.set_threads(num_threads=self.num_threads)

        self.set_verbose(verbose=self.set_verbose)

    def set_seed(self, seed: int = 42):
        """Set seed for RNG.

        Args:
            seed (int, optional): seed for RNG. Defaults to 42.
        """
        self.dimensionality_reduction.set_seed(seed)

    def set_block(self, block: Optional[Sequence] = None):
        """Set block assignments for each cell.

        Args:
            block (Sequence, optional): Blocks assignments
                for each cell. Defaults to None.
        """
        self.quality_control.set_block(block)
        self.normalization.set_block(block)
        self.feature_selection.set_block(block)
        self.dimensionality_reduction.set_block(block)
        self.clustering.set_block(block)
        self.nearest_neighbors.set_block(block)
        self.marker_detection.set_block(block)

    def set_subset(self, subset: Optional[Mapping] = None):
        """Set subsets.

        Args:
            subset (Mapping, optional): Set subsets. Defaults to None.
        """
        if subset is None:
            subset = {}

        self.quality_control.set_subset(subset)
        self.normalization.set_subset(subset)
        self.feature_selection.set_subset(subset)
        self.dimensionality_reduction.set_subset(subset)
        self.clustering.set_subset(subset)
        self.nearest_neighbors.set_subset(subset)
        self.marker_detection.set_subset(subset)

    def set_verbose(self, verbose: bool = False):
        """Set verbose to display logs.

        Args:
            verbose (bool, optional): Display logs? Defaults to False.
        """
        self.quality_control.set_verbose(verbose)
        self.normalization.set_verbose(verbose)
        self.feature_selection.set_verbose(verbose)
        self.dimensionality_reduction.set_verbose(verbose)
        self.clustering.set_verbose(verbose)
        self.nearest_neighbors.set_verbose(verbose)
        self.marker_detection.set_verbose(verbose)

    def set_threads(self, num_threads: int = 1):
        """Set number of threads to use.

        Args:
            num_threads (int, optional): Number of threads. Defaults to 1.
        """
        self.quality_control.set_threads(num_threads)
        self.normalization.set_threads(num_threads)
        self.feature_selection.set_threads(num_threads)
        self.dimensionality_reduction.set_threads(num_threads)
        self.clustering.set_threads(num_threads)
        self.nearest_neighbors.set_threads(num_threads)
        self.marker_detection.set_threads(num_threads)


def __analyze(
    matrix: Any, features: Sequence[str], options: AnalyzeOptions = AnalyzeOptions()
) -> Mapping:
    ptr = tatamize(matrix)

    NR = ptr.nrow()
    NC = ptr.ncol()

    if len(features) != NR:
        raise ValueError(
            "Length of `features` not same as number of rows in the matrix."
        )

    if options.block is not None:
        if len(options.block) != NC:
            raise ValueError(
                "Length of `block` not same as number of columns in the matrix."
            )

    # setting up QC metrics and filters.
    subsets = {}
    if options.quality_control.mito_subset is not None:
        if isinstance(options.quality_control.mito_subset, str):
            subsets["mito"] = qc.guess_mito_from_symbols(
                features, options.quality_control.options.quality_control.mito_subset
            )
        elif isinstance(options.quality_control.mito_subset, bool):
            subsets["mito"] = qc.guess_mito_from_symbols(features)
        else:
            raise ValueError(
                "Unsupported value provided for `qc_mito_subset`:"
                f" {options.quality_control.mito_subset}"
            )

    options.set_subset(subset=subsets)

    qc_metrics = qc.per_cell_rna_qc_metrics(
        matrix,
        options=options.quality_control.per_cell_rna_qc_metrics,
    )
    qc_thresholds = qc.suggest_rna_qc_filters(
        qc_metrics,
        options=options.quality_control.suggest_rna_qc_filters,
    )

    if options.quality_control.custom_thresholds is not None:
        if not isinstance(options.quality_control.custom_thresholds, BiocFrame):
            raise TypeError("'qc_custom_thresholds' is not a `BiocFrame` object.")

        for col in qc_thresholds.columnNames:
            if col in options.quality_control.custom_thresholds.columnNames:
                qc_thresholds.column(col).fill(
                    options.quality_control.custom_thresholds[col]
                )

    qc_filter = qc.create_rna_qc_filter(
        qc_metrics, qc_thresholds, options.quality_control.create_rna_qc_filters
    )

    # Finally QC cells
    qc_filtered = qc.filter_cells(ptr, filter=qc_filter)

    # Log-normalize counts
    normed = norm.log_norm_counts(
        qc_filtered, options=options.normalization.log_normalize_counts
    )

    #  Model gene variances
    var_stats = feat.model_gene_variances(
        normed, options=options.feature_selection.model_gene_variances
    )

    # Choose highly variable genes
    selected_feats = feat.choose_hvgs(
        var_stats.column("residuals"), options=options.feature_selection.choose_hvgs
    )

    # Compute PC's
    options.dimensionality_reduction.run_pca.subset = selected_feats
    pca = dimred.run_pca(
        normed,
        options=options.dimensionality_reduction.run_pca,
    )

    neighbor_idx = nn.build_neighbor_index(
        pca.principal_components, options=nn.BuildNeighborIndexOptions(approximate=True)
    )

    tsne_nn = dimred.tsne_perplexity_to_neighbors(
        options.dimensionality_reduction.run_tsne.initialize_tsne.perplexity
    )
    umap_nn = options.dimensionality_reduction.run_umap.initialize_umap.num_neighbors
    snn_nn = options.clustering.build_snn_graph.num_neighbors

    nn_dict = {}
    for k in set([umap_nn, tsne_nn, snn_nn]):
        nn_dict[k] = nn.find_nearest_neighbors(
            neighbor_idx,
            options=options.nearest_neighbors.find_nn,
        )

    executor = ProcessPoolExecutor(max_workers=2)
    _tasks = []

    _tasks.append(
        executor.submit(
            dimred.run_tsne,
            nn_dict[tsne_nn],
            options.dimensionality_reduction.run_tsne,
        )
    )

    _tasks.append(
        executor.submit(
            dimred.run_umap, nn_dict[umap_nn], options.dimensionality_reduction.run_umap
        )
    )

    remaining_threads = max(1, options.num_threads - 2)
    options.clustering.set_threads(remaining_threads)
    graph = clust.build_snn_graph(
        nn_dict[snn_nn], options=options.clustering.build_snn_graph
    )

    # clusters
    clusters = graph.community_multilevel(
        resolution=options.clustering.resolution
    ).membership

    # Score Markers for each cluster
    options.marker_detection.set_threads(remaining_threads)
    markers = mark.score_markers(
        normed,
        options=options.marker_detection.score_markers,
    )

    embeddings = []
    for task in _tasks:
        embeddings.append(task.result())

    executor.shutdown()

    return {
        "qc_metrics": qc_metrics,
        "qc_thresholds": qc_thresholds,
        "qc_filter": qc_filter,
        "variances": var_stats,
        "hvgs": selected_feats,
        "pca": pca,
        "tsne": embeddings[0],
        "umap": embeddings[1],
        "clustering": clusters,
        "marker_detection": markers,
    }


@singledispatch
def analyze(
    matrix: Any, features: Sequence[str], options: AnalyzeOptions = AnalyzeOptions()
) -> Mapping:
    """Run all steps of the scran workflow for single-cell RNA-seq datasets.

    - Remove low-quality cells
    - Normalization and log-transformation
    - Model mean-variance trend across genes
    - PCA on highly variable genes
    - graph-based clustering
    - dimensionality reductions, t-SNE & UMAP
    - Marker detection for each cluster


    Options(AbstractStepOptions)
        matrix (Any): "Count" matrix.
        features (Sequence[str]): Features information for the rows of the matrix.
        block (Sequence, optional): Block assignment for each cell.
            This is used to segregate cells in order to perform comparisons within
            each block. Defaults to None, indicating all cells are part of the same
            block.
        options (AnalyzeOptions): Optional analysis parameters.

    Raises:
        NotImplementedError: if ``matrix`` is not an expected type.

    Returns:
        Mapping: Results from various steps.
    """
    if is_matrix_expected_type(matrix):
        return __analyze(matrix, features=features, options=options)
    else:
        raise NotImplementedError(
            f"analyze is not supported for objects of class: {type(matrix)}"
        )


@analyze.register
def analyze_sce(
    matrix: SingleCellExperiment,
    features: Union[Sequence[str], str],
    assay: str = "counts",
    options: AnalyzeOptions = AnalyzeOptions(),
) -> Mapping:
    """Run all steps of the scran workflow for single-cell RNA-seq datasets.

    - Remove low-quality cells
    - Normalization and log-transformation
    - Model mean-variance trend across genes
    - PCA on highly variable genes
    - graph-based clustering
    - dimensionality reductions, t-SNE & UMAP
    - Marker detection for each cluster


    Options(AbstractStepOptions)
        matrix (Any): "Count" matrix.
        features (Union[Sequence[str], str]): Features information for the rows of
            the matrix.
        block (Union[Sequence, str], optional): Block assignment for each cell.
            This is used to segregate cells in order to perform comparisons within
            each block. Defaults to None, indicating all cells are part of the same
            block.
        assay (str): assay matrix to use for analysis. Defaults to "counts".
        options (AnalyzeOptions): Optional analysis parameters.

    Raises:
        ValueError: object does not contain a 'counts' matrix.

    Returns:
        Mapping: Results from various steps
    """
    if assay not in matrix.assayNames:
        raise ValueError(f"SCE does not contain a '{assay}' matrix.")

    if isinstance(features, str):
        if isinstance(matrix.rowData, BiocFrame):
            features = matrix.rowData.column(features)
        else:
            features = matrix.rowData[features]

    if isinstance(options.block, str):
        if isinstance(matrix.rowData, BiocFrame):
            block = matrix.colData.column(options.block)
        else:
            block = matrix.colData[block]

    return __analyze(matrix.assay("counts"), features=features, options=options)
