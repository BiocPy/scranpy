from concurrent.futures import ProcessPoolExecutor, wait
from functools import singledispatch, singledispatchmethod
from dataclasses import dataclass, field
from typing import Tuple, List, Mapping, Optional, Sequence, Union, Callable

from singlecellexperiment import SingleCellExperiment
from biocframe import BiocFrame
from mattress import TatamiNumericPointer, tatamize
from numpy import array
from copy import deepcopy
from igraph import Graph

from asyncio import Future

from . import clustering as clust
from . import dimensionality_reduction as dimred
from . import feature_selection as feat
from . import marker_detection as mark
from . import nearest_neighbors as nn
from . import normalization as norm
from . import quality_control as qc
from .types import MatrixTypes, is_matrix_expected_type, validate_object_type

__author__ = "ltla, jkanche"
__copyright__ = "ltla"
__license__ = "MIT"


@dataclass
class AnalyzeOptions:
    """Class to manage options across all steps."""

    quality_control: qc.RnaQualityControlOptions = field(
        default_factory=qc.RnaQualityControlOptions
    )
    normalization: norm.NormalizationOptions = field(
        default_factory=norm.NormalizationOptions
    )
    feature_selection: feat.FeatureSelectionOptions = field(
        default_factory=feat.FeatureSelectionOptions
    )
    dimensionality_reduction: dimred.DimensionalityReductionOptions = field(
        default_factory=dimred.DimensionalityReductionOptions
    )
    clustering: clust.ClusteringOptions = field(default_factory=clust.ClusteringOptions)
    nearest_neighbors: nn.NearestNeighborsOptions = field(
        default_factory=nn.NearestNeighborsOptions
    )
    marker_detection: mark.MarkerDetectionOptions = field(
        default_factory=mark.MarkerDetectionOptions
    )
    block: Optional[Sequence] = None
    seed: int = 42
    num_threads: int = 1
    verbose: bool = False

    def __post_init__(self):
        validate_object_type(self.quality_control, qc.RnaQualityControlOptions)
        validate_object_type(self.normalization, norm.NormalizationOptions)
        validate_object_type(self.feature_selection, feat.FeatureSelectionOptions)
        validate_object_type(
            self.dimensionality_reduction, dimred.DimensionalityReductionOptions
        )
        validate_object_type(self.clustering, clust.ClusteringOptions)
        validate_object_type(self.nearest_neighbors, nn.NearestNeighborsOptions)
        validate_object_type(self.marker_detection, mark.MarkerDetectionOptions)

        if self.block is not None:
            self.set_block(block=self.block)

        if self.seed != 42:
            self.set_seed(seed=self.seed)

        if self.num_threads != 1:
            self.set_threads(num_threads=self.num_threads)

        self.set_verbose(verbose=self.verbose)

    def set_seed(self, seed: int = 42):
        """Set seed for RNG.

        Args:
            seed (int, optional): Seed for random number generation.
                Defaults to 42.
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
        self.marker_detection.set_block(block)

    def set_subset(self, subset: Optional[Mapping] = None):
        """Set subsets.

        Args:
            subset (Mapping, optional): Set subsets.
                Defaults to None.
        """
        if subset is None:
            subset = {}

        self.quality_control.set_subset(subset)

    def set_verbose(self, verbose: bool = False):
        """Set verbose to display logs.

        Args:
            verbose (bool, optional): Whether to print logs.
                Defaults to False.
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


@dataclass
class AnalyzeResults:
    """Class to manage results across all analyis steps."""

    quality_control: qc.RnaQualityControlResults = field(
        default_factory=qc.RnaQualityControlResults
    )
    normalization: norm.NormalizationResults = field(
        default_factory=norm.NormalizationResults
    )
    feature_selection: feat.FeatureSelectionResults = field(
        default_factory=feat.FeatureSelectionResults
    )
    dimensionality_reduction: dimred.DimensionalityReductionResults = field(
        default_factory=dimred.DimensionalityReductionResults
    )
    clustering: clust.ClusteringResults = field(default_factory=clust.ClusteringResults)
    marker_detection: mark.MarkerDetectionResults = field(
        default_factory=mark.MarkerDetectionResults
    )
    nearest_neighbors: nn.NearestNeighborsResults = field(
        default_factory=nn.NearestNeighborsResults
    )

    def __to_sce(self, x: MatrixTypes, assay: str, include_gene_data: bool = False):
        if isinstance(x, TatamiNumericPointer):
            raise ValueError("`TatamiNumericPointer` is not yet supported (for 'x')")

        keep = [not y for y in self.quality_control.qc_filter.tolist()]

        # TODO: need to add logcounts
        sce = SingleCellExperiment(assays={assay: x[:, keep]})

        sce.colData = self.quality_control.qc_metrics[keep,:]
        sce.colData["clusters"] = self.clustering.clusters

        sce.reducedDims = {
            "pca": self.dimensionality_reduction.pca.principal_components,
            "tsne": array(
                [
                    self.dimensionality_reduction.tsne.x,
                    self.dimensionality_reduction.tsne.y,
                ]
            ).T,
            "umap": array(
                [
                    self.dimensionality_reduction.umap.x,
                    self.dimensionality_reduction.umap.y,
                ]
            ).T,
        }

        if include_gene_data is True:
            sce.rowData = self.feature_selection.gene_variances

        return sce

    @singledispatchmethod
    def to_sce(
        self, x, assay: str = "counts", include_gene_data: bool = False
    ) -> SingleCellExperiment:
        """Save results as a :py:class:`singlecellexperiment.SingleCellExperiment`.

        Args:
            x: Input object. usually a matrix of raw counts.
            assay (str, optional): assay name for the matrix.
                Defaults to "counts".
            include_gene_data (bool, optional): Whether to include gene variances.
                Defaults to False.

        Returns:
            SingleCellExperiment: An SCE with the results.
        """
        return self.__to_sce(x, assay, include_gene_data)

    @to_sce.register
    def _(
        self,
        x: SingleCellExperiment,
        assay: str = "counts",
        include_gene_data: bool = False,
    ) -> SingleCellExperiment:
        if assay not in x.assayNames:
            raise ValueError(f"SCE does not contain a '{assay}' matrix.")

        mat = x.assay(assay)
        return self.__to_sce(mat, assay, include_gene_data)


def _unserialize_neighbors_before_run(f, serialized, opt):
    nnres = nn.NeighborResults.unserialize(serialized)
    return f(nnres, opt)



def run_neighbor_suite(
    principal_components,
    build_neighbor_index_options: nn.BuildNeighborIndexOptions = nn.BuildNeighborIndexOptions(),
    find_nearest_neighbors_options: nn.FindNearestNeighborsOptions = nn.FindNearestNeighborsOptions(),
    run_umap_options: dimred.RunUmapOptions = dimred.RunUmapOptions(),
    run_tsne_options: dimred.RunTsneOptions = dimred.RunTsneOptions(),
    build_snn_graph_options: clust.BuildSnnGraphOptions = clust.BuildSnnGraphOptions(),
    num_threads: int = 1
) -> Tuple[Callable, Callable, Graph, int]:
    """Run the suite of nearest neighbor methods together.
    This builds the index once and re-uses it for all methods.
    Given enough threads, it also runs all post-neighbor-detection functions in parallel,
    as none of them depend on each other.

    Args:
        principal_components (ndarray):
            Matrix of principal components where rows are cells and columns are PCs.
            Thi is usually produced by :py:meth:`~scranpy.dimensionality_reduction.run_pca.run_pca`.

        build_neighbor_index_options (BuildNeighborIndexOptions, optional):
            Optional arguments to pass to :py:meth:`~scranpy.nearest_neighbors.build_neighbor_index.build_neighbor_index`.

        find_nearest_neighbors_options (FindNearestNeighborsOptions, optional):
            Optional arguments to pass to :py:meth:`~scranpy.nearest_neighbors.find_nearest_neighbors.find_nearest_neighbors`.

        run_umap_options (RunUmapOptions, optional):
            Optional arguments to pass to :py:meth:`~scranpy.dimensionality_reduction.run_umap.run_umap`.

        run_tsne_options (RunTsneOptions, optional):
            Optional arguments to pass to :py:meth:`~scranpy.dimensionality_reduction.run_tsne.run_tsne`.

        build_snn_graph_options (BuildSnnGraphOptions, optional):
            Optional arguments to pass to :py:meth:`~scranpy.clustering.build_snn_graph.build_snn_graph`.

        num_threads (int, optional):
            Number of threads to use for the parallel execution of UMAP, t-SNE and SNN graph construction.
            This overrides the specified number of threads in ``run_umap``, ``run_tsne`` and ``build_snn_graph``.

    Returns:
        A tuple containing, in order:
        - A function that takes no arguments and returns a tuple containing the t-SNE and UMAP coordinates.
        - The shared nearest neighbor graph from :py:meth:`~scranpy.clustering.build_snn_graph.build_snn_graph`.
        - The number of remaining threads.

        The idea is that the number of remaining threads can be used to perform tasks on the main thread (e.g., clustering, marker detection) while the t-SNE and UMAP are still being computed;
        once all tasks on the main thread have completed, the first function can be called to obtain the coordinates.
    """

    index = nn.build_neighbor_index(
        principal_components,
        options=build_neighbor_index_options,
    )

    tsne_nn = dimred.tsne_perplexity_to_neighbors(
        run_tsne_options.initialize_tsne.perplexity
    )
    umap_nn = run_umap_options.initialize_umap.num_neighbors
    snn_nn = build_snn_graph_options.num_neighbors

    nn_dict = {}
    for k in set([umap_nn, tsne_nn, snn_nn]):
        nn_dict[k] = nn.find_nearest_neighbors(
            index,
            k=k,
            options=find_nearest_neighbors_options,
        )

    serialized_dict = {}
    for k in set([umap_nn, tsne_nn]):
        serialized_dict[k] = nn_dict[k].serialize()

    # Attempting to evenly distribute threads across the tasks.  t-SNE and UMAP
    # are run on separate processes while the SNN graph construction is kept on
    # the main thread because we'll need the output for marker detection.
    threads_per_task = max(1, int(num_threads / 3))
    executor = ProcessPoolExecutor(max_workers=min(2, num_threads))
    _tasks = []

    run_tsne_copy = deepcopy(run_tsne_options)
    run_tsne_copy.set_threads(threads_per_task)
    _tasks.append(
        executor.submit(
            _unserialize_neighbors_before_run,
            dimred.run_tsne,
            serialized_dict[tsne_nn],
            run_tsne_copy,
        )
    )

    run_umap_copy = deepcopy(run_umap_options)
    run_umap_copy.set_threads(threads_per_task)
    _tasks.append(
        executor.submit(
            _unserialize_neighbors_before_run,
            dimred.run_umap, 
            serialized_dict[umap_nn], 
            run_umap_copy,
        )
    )

    def retrieve():
        wait(_tasks)
        executor.shutdown()

    def get_tsne():
        retrieve()
        return _tasks[0].result()
    
    def get_umap():
        retrieve()
        return _tasks[1].result()

    build_snn_graph_copy = deepcopy(build_snn_graph_options)
    remaining_threads = max(1, num_threads - threads_per_task * 2)
    build_snn_graph_copy.set_threads(remaining_threads)
    graph = clust.build_snn_graph(nn_dict[snn_nn], options=build_snn_graph_copy)

    return get_tsne, get_umap, graph, remaining_threads


def __analyze(
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
