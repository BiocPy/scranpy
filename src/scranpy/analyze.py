from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import singledispatch
from typing import Any, Literal, Mapping, Optional, Sequence, Union

import numpy as np
import summarizedexperiment as se
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
from .types import is_matrix_expected_type

__author__ = "ltla, jkanche"
__copyright__ = "ltla"
__license__ = "MIT"


@dataclass
class QualityControlOpts:
    """Options for Quality Control (RNA).

    Attributes:
        mito_subset (Unions[str, bool], optional): Prefix to filter
            mitochindrial genes. Can be a

            - ``True`` to use the default prefix "mt-".
            - a (string) custom prefix to identify mitochondrial genes.
            Defaults to None.
        num_mads (int, optional): Number of median absolute deviations to
            filter low-quality cells. Defaults to 3.
        custom_thresholds (BiocFrame, optional): Suggested (or modified) filters from
            :py:meth:`~scranpy.quality_control.rna.suggest_rna_qc_filters`
            function. Defaults to None.
    """

    mito_subset: Optional[Union[str, bool]] = None
    num_mads: int = qc.SuggestRnaQcFilters.num_mads
    custom_thresholds: Optional[BiocFrame] = None


@dataclass
class LogNormalizationOpts:
    """Log-normalize counts.

    Attributes:
        size_factors (np.ndarray, optional): Size factors for each cell.
            Defaults to None.
        center (bool, optional): Center the size factors?. Defaults to True.
        allow_zeros (bool, optional): Allow zeros?. Defaults to False.
        allow_non_finite (bool, optional): Allow `nan` or `inifnite` numbers?.
            Defaults to False.
    """

    size_factors: Optional[np.ndarray] = norm.LogNormalizeCountsArgs.size_factors
    center: bool = norm.LogNormalizeCountsArgs.center
    allow_zeros: bool = norm.LogNormalizeCountsArgs.allow_zeros
    allow_non_finite: bool = norm.LogNormalizeCountsArgs.allow_non_finite


@dataclass
class FeatureSelectionOpts:
    """Feature selection.

    Attributes:
        span (float, optional): Span to use for the LOWESS trend fitting.
            Defaults to 0.3.
        num_hvgs (int, optional): Number of HVGs to pick. Defaults to 4000.
    """

    span: float = feat.ModelGeneVariancesArgs.span
    num_hvgs: int = feat.ChooseHvgArgs.number


@dataclass
class PcaOpts:
    """Compute Principal Components.

    Attributes:
        rank (int): Number of top PC's to compute.
        scale (bool, optional): Whether to scale each feature to unit variance.
            Defaults to False.
        block_method (Literal["none", "project", "regress"], optional): How to adjust
            the PCA for the blocking factor.

            - ``"regress"`` will regress out the factor, effectively performing a PCA on
                the residuals. This only makes sense in limited cases, e.g., inter-block
                differences are linear and the composition of each block is the same.
            - ``"project"`` will compute the rotation vectors from the residuals but
                will project the cells onto the PC space. This focuses the PCA on
                within-block variance while avoiding any assumptions about the
                nature of the inter-block differences.
            - ``"none"`` will ignore any blocking factor, i.e., as if ``block = null``.
                Any inter-block differences will both contribute to the determination of
                the rotation vectors and also be preserved in the PC space.
                This option is only used if ``block`` is not `null`.
            Defaults to "project".
        block_weights (bool, optional): Whether to weight each block so that it
            contributes the same number of effective observations to the covariance
            matrix. Defaults to True.
    """

    rank: int = dimred.RunPcaArgs.rank
    scale: bool = dimred.RunPcaArgs.scale
    block_method: Literal["none", "project", "regress"] = dimred.RunPcaArgs.block_method
    block_weights: bool = dimred.RunPcaArgs.block_weights


@dataclass
class TsneOpts:
    """Compute t-SNE embeddings.

    Attributes:
        perplexity (int, optional): Perplexity to use when computing neighbor
            probabilities. Defaults to 30.
        max_iterations (int, optional): Maximum number of iterations. Defaults to 500.

    """

    perplexity: int = dimred.InitializeTsneArgs.perplexity
    max_iterations: int = dimred.RunTsneArgs.max_iterations


@dataclass
class UmapOpts:
    """Compute UMAP embeddings.

    Attributes:
        min_dist (float, optional): Minimum distance between points. Defaults to 0.1.
        num_neighbors (int, optional): Number of neighbors to use in the UMAP algorithm.
            Defaults to 15.
        num_epochs (int, optional): Number of epochs to run. Defaults to 500.
    """

    min_dist: float = dimred.InitializeUmapArgs.min_dist
    num_neighbors: int = dimred.InitializeUmapArgs.num_neighbors
    num_epochs: int = dimred.InitializeUmapArgs.num_epochs


@dataclass
class SharedNearestNeighborOpts:
    """Options for Clustering step.

    Attributes:
        num_neighbors (int, optional): Number of neighbors to use.
            Defaults to 15.
        approximate (bool, optional): Whether to build an index for an approximate
            neighbor search. Defaults to True.
        weight_scheme (Literal["ranked", "jaccard", "number"], optional): Weighting
            scheme for the edges between cells. This can be based on the top ranks
            of the shared neighbors ("rank"), the number of shared neighbors ("number")
            or the Jaccard index of the neighbor sets between cells ("jaccard").
            Defaults to "ranked".
        resolution (int): Resolution parameter to use in modularity to identify
            clusters. Defaults to 1.
    """

    num_neighbors: int = clust.BuildSnnGraphArgs.num_neighbors
    approximate: bool = True
    weight_scheme: Literal["ranked", "jaccard", "number"] = "ranked"
    resolution: int = 1


@dataclass
class ScoreMarkersOpts:
    """Options to rank markers for each cluster.

    Attributes:
        threshold (float, optional): Log-fold change threshold to use for computing
            `Cohen's d` and `AUC`. Large positive values favor markers with large
            log-fold changes over those with low variance. Defaults to 0.
        compute_auc (bool, optional): Whether to compute the AUCs as an effect size.
            This can be set to False for greater speed and memory efficiency.
            Defaults to True.
    """

    threshold: float = mark.ScoreMarkersArgs.threshold
    compute_auc: bool = mark.ScoreMarkersArgs.compute_auc


def __analyze(
    matrix: Any,
    features: Sequence[str],
    block: Optional[Sequence] = None,
    qc_options: QualityControlOpts = QualityControlOpts(),
    norm_options: LogNormalizationOpts = LogNormalizationOpts(),
    feature_selection_options: FeatureSelectionOpts = FeatureSelectionOpts(),
    pca_options: PcaOpts = PcaOpts(),
    tsne_options: TsneOpts = TsneOpts(),
    umap_options: UmapOpts = UmapOpts(),
    cluster_options: SharedNearestNeighborOpts = SharedNearestNeighborOpts(),
    marker_options: ScoreMarkersOpts = ScoreMarkersOpts(),
    seed: int = 42,
    num_threads: int = 1,
) -> Mapping:
    ptr = tatamize(matrix)

    NR = ptr.nrow()
    NC = ptr.ncol()

    if len(features) != NR:
        raise ValueError(
            "Length of `features` not same as number of rows in the matrix."
        )

    if block is not None:
        if len(block) != NC:
            raise ValueError(
                "Length of `block` not same as number of columns in the matrix."
            )

    # setting up QC metrics and filters.
    subsets = {}
    if qc_options.mito_subset is not None:
        if isinstance(qc_options.mito_subset, str):
            subsets["mito"] = qc.guess_mito_from_symbols(
                features, qc_options.mito_subset
            )
        elif isinstance(qc_options.mito_subset, bool):
            subsets["mito"] = qc.guess_mito_from_symbols(features)
        else:
            raise ValueError(
                f"Unsupported value provided for `qc_mito_subset`: {qc_options.mito_subset}"
            )

    qc_metrics = qc.per_cell_rna_qc_metrics(
        matrix,
        options=qc.PerCellRnaQcMetricsArgs(subsets=subsets, num_threads=num_threads),
    )
    qc_thresholds = qc.suggest_rna_qc_filters(
        qc_metrics,
        options=qc.SuggestRnaQcFilters(block=block, num_mads=qc_options.num_mads),
    )

    if qc_options.custom_thresholds is not None:
        if not isinstance(qc_options.custom_thresholds, BiocFrame):
            raise TypeError("'qc_custom_thresholds' is not a `BiocFrame` object.")

        for col in qc_thresholds.columnNames:
            if col in qc_options.custom_thresholds.columnNames:
                qc_thresholds.column(col).fill(qc_options.custom_thresholds[col])

    qc_filter = qc.create_rna_qc_filter(
        qc_metrics, qc_thresholds, options=qc.CreateRnaQcFilter(block=block)
    )

    # Finally QC cells
    qc_filtered = qc.filter_cells(ptr, filter=qc_filter)

    # Log-normalize counts
    __norm_opts = norm.LogNormalizeCountsArgs(
        block=block,
        size_factors=norm_options.size_factors,
        center=norm_options.center,
        allow_zeros=norm_options.allow_zeros,
        allow_non_finite=norm_options.allow_non_finite,
        num_threads=num_threads,
    )
    normed = norm.log_norm_counts(qc_filtered, options=__norm_opts)

    #  Model gene variances
    __mgv_opts = feat.ModelGeneVariancesArgs(
        block=block, span=feature_selection_options.span, num_threads=num_threads
    )
    var_stats = feat.model_gene_variances(normed, options=__mgv_opts)

    # Choose highly variable genes
    __chvh_opts = feat.ChooseHvgArgs(number=feature_selection_options.num_hvgs)
    selected_feats = feat.choose_hvgs(
        var_stats.column("residuals"), options=__chvh_opts
    )

    # Compute PC's
    pca = dimred.run_pca(
        normed,
        options=dimred.RunPcaArgs(
            rank=pca_options.rank,
            subset=selected_feats,
            scale=pca_options.scale,
            block_method=pca_options.block_method,
            block_weights=pca_options.block_weights,
            num_threads=num_threads,
        ),
    )

    neighbor_idx = nn.build_neighbor_index(
        pca.principal_components, options=nn.BuildNeighborIndexArgs(approximate=True)
    )

    tsne_nn = dimred.tsne_perplexity_to_neighbors(tsne_options.perplexity)
    umap_nn = umap_options.num_neighbors
    snn_nn = cluster_options.num_neighbors

    nn_dict = {}
    for k in set([umap_nn, tsne_nn, snn_nn]):
        nn_dict[k] = nn.find_nearest_neighbors(
            neighbor_idx,
            options=nn.FindNearestNeighborsArgs(k=k, num_threads=num_threads),
        )

    executor = ProcessPoolExecutor(max_workers=2)
    _tasks = []

    _tasks.append(
        executor.submit(
            dimred.run_tsne,
            nn_dict[tsne_nn],
            dimred.RunTsneArgs(
                max_iterations=tsne_options.max_iterations,
                initialize_tsne=dimred.InitializeTsneArgs(
                    perplexity=tsne_options.perplexity,
                    seed=seed,
                    num_threads=num_threads,
                ),
            ),
        )
    )

    _tasks.append(
        executor.submit(
            dimred.run_umap,
            nn_dict[umap_nn],
            dimred.RunUmapArgs(
                initialize_umap=dimred.InitializeUmapArgs(
                    min_dist=umap_options.min_dist,
                    num_neighbors=umap_options.num_neighbors,
                    num_epochs=umap_options.num_epochs,
                    seed=seed,
                    num_threads=num_threads,
                )
            ),
        )
    )

    remaining_threads = max(1, num_threads - 2)
    graph = clust.build_snn_graph(
        nn_dict[snn_nn],
        options=clust.BuildSnnGraphArgs(
            num_neighbors=cluster_options.num_neighbors,
            approximate=cluster_options.approximate,
            weight_scheme=cluster_options.weight_scheme,
            num_threads=remaining_threads,
        ),
    )

    # clusters
    clusters = graph.community_multilevel(
        resolution=cluster_options.resolution
    ).membership

    # Score Markers for each cluster
    markers = mark.score_markers(
        normed,
        options=mark.ScoreMarkersArgs(
            grouping=clusters,
            compute_auc=marker_options.compute_auc,
            threshold=marker_options.threshold,
            num_threads=remaining_threads,
        ),
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
    matrix: Any,
    features: Sequence[str],
    block: Optional[Sequence] = None,
    qc_options: QualityControlOpts = QualityControlOpts(),
    norm_options: LogNormalizationOpts = LogNormalizationOpts(),
    feature_selection_options: FeatureSelectionOpts = FeatureSelectionOpts(),
    pca_options: PcaOpts = PcaOpts(),
    tsne_options: TsneOpts = TsneOpts(),
    umap_options: UmapOpts = UmapOpts(),
    cluster_options: SharedNearestNeighborOpts = SharedNearestNeighborOpts(),
    marker_options: ScoreMarkersOpts = ScoreMarkersOpts(),
    seed: int = 42,
    num_threads: int = 1,
) -> Mapping:
    """Run all steps of the scran workflow for single-cell RNA-seq datasets.

    - Remove low-quality cells
    - Normalization and log-transformation
    - Model mean-variance trend across genes
    - PCA on highly variable genes
    - graph-based clustering
    - dimensionality reductions, t-SNE & UMAP
    - Marker detection for each cluster


    Args:
        matrix (Any): "Count" matrix.
        features (Sequence[str]): Features information for the rows of the matrix.
        block (Sequence, optional): Block assignment for each cell.
            This is used to segregate cells in order to perform comparisons within
            each block. Defaults to None, indicating all cells are part of the same
            block.
        qc_options (QualityControlOpts, optional): Additional parameters to the QC step.
            Defaults to :py:class:`~scranpy.analyze.QualityControlOpts`.
        norm_options (LogNormalizationOpts, optional): Additional params to compute
            log-normalization. Defaults to
            :py:class:`~scranpy.analyze.QualityControlOpts`.
        feature_selection_options (FeatureSelectionOpts, optional): Addition parameters
            for feature selection. Defaults to
            :py:class:`~scranpy.analyze.FeatureSelectionOpts`.
        pca_options (PcaOpts, optional): Additional params to compute PC's. Defaults to
            :py:class:`~scranpy.analyze.PcaOpts`.
        tsne_options (TsneOpts, optional): Additional parameters to compute t-SNE
            embedding. Defaults to :py:class:`~scranpy.analyze.TsneOpts`.
        umap_options (UmapOpts, optional): Additional parameters to compute UMAP
            embedding. Defaults to :py:class:`~scranpy.analyze.UmapOpts`.
        cluster_options (SharedNearestNeighborOpts, optional): Additional parameters to
            build the shared nearest neighbor index. Defaults to
            :py:class:`~scranpy.analyze.SharedNearestNeighborOpts`.
        marker_options (ScoreMarkersOpts, optional): Additional parameters to identify
            top ranked markers for each group. Defaults to
            :py:class:`~scranpy.analyze.ScoreMarkersOpts`.
        seed (int, optional): Seed to set for RNG. Defaults to 42.
        num_threads (int, optional): Number of threads to se. Defaults to 1.

    Raises:
        NotImplementedError: if ``matrix`` is not an expected type.

    Returns:
        Mapping: Results from various steps
    """
    if is_matrix_expected_type(matrix):
        return __analyze(
            matrix,
            features=features,
            block=block,
            qc_options=qc_options,
            norm_options=norm_options,
            feature_selection_options=feature_selection_options,
            pca_options=pca_options,
            tsne_options=tsne_options,
            umap_options=umap_options,
            cluster_options=cluster_options,
            marker_options=marker_options,
            seed=seed,
            num_threads=num_threads,
        )
    else:
        raise NotImplementedError(
            f"analyze is not supported for objects of class: {type(matrix)}"
        )


@analyze.register
def analyze_sce(
    matrix: SingleCellExperiment,
    features: Union[Sequence[str], str],
    block: Optional[Union[Sequence, str]] = None,
    assay: str = "counts",
    qc_options: QualityControlOpts = QualityControlOpts(),
    norm_options: LogNormalizationOpts = LogNormalizationOpts(),
    feature_selection_options: FeatureSelectionOpts = FeatureSelectionOpts(),
    pca_options: PcaOpts = PcaOpts(),
    tsne_options: TsneOpts = TsneOpts(),
    umap_options: UmapOpts = UmapOpts(),
    cluster_options: SharedNearestNeighborOpts = SharedNearestNeighborOpts(),
    marker_options: ScoreMarkersOpts = ScoreMarkersOpts(),
    seed: int = 42,
    num_threads: int = 1,
) -> Mapping:
    """Run all steps of the scran workflow for single-cell RNA-seq datasets.

    - Remove low-quality cells
    - Normalization and log-transformation
    - Model mean-variance trend across genes
    - PCA on highly variable genes
    - graph-based clustering
    - dimensionality reductions, t-SNE & UMAP
    - Marker detection for each cluster


    Args:
        matrix (Any): "Count" matrix.
        features (Union[Sequence[str], str]): Features information for the rows of
            the matrix.
        block (Union[Sequence, str], optional): Block assignment for each cell.
            This is used to segregate cells in order to perform comparisons within
            each block. Defaults to None, indicating all cells are part of the same
            block.
        assay (str): assay matrix to use for analysis. Defaults to "counts".
        qc_options (QualityControlOpts, optional): Additional parameters to the QC step.
            Defaults to :py:class:`~scranpy.analyze.QualityControlOpts`.
        norm_options (LogNormalizationOpts, optional): Additional params to compute
            log-normalization. Defaults to
            :py:class:`~scranpy.analyze.QualityControlOpts`.
        feature_selection_options (FeatureSelectionOpts, optional): Addition parameters
            for feature selection. Defaults to
            :py:class:`~scranpy.analyze.FeatureSelectionOpts`.
        pca_options (PcaOpts, optional): Additional params to compute PC's. Defaults to
            :py:class:`~scranpy.analyze.PcaOpts`.
        tsne_options (TsneOpts, optional): Additional parameters to compute t-SNE
            embedding. Defaults to :py:class:`~scranpy.analyze.TsneOpts`.
        umap_options (UmapOpts, optional): Additional parameters to compute UMAP
            embedding. Defaults to :py:class:`~scranpy.analyze.UmapOpts`.
        cluster_options (SharedNearestNeighborOpts, optional): Additional parameters to
            build the shared nearest neighbor index. Defaults to
            :py:class:`~scranpy.analyze.SharedNearestNeighborOpts`.
        marker_options (ScoreMarkersOpts, optional): Additional parameters to identify
            top ranked markers for each group. Defaults to
            :py:class:`~scranpy.analyze.ScoreMarkersOpts`.
        seed (int, optional): Seed to set for RNG. Defaults to 42.
        num_threads (int, optional): Number of threads to se. Defaults to 1.

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

    if isinstance(block, str):
        if isinstance(matrix.rowData, BiocFrame):
            block = block.colData.column(block)
        else:
            block = block.colData1[block]

    return __analyze(
        matrix.assay("counts"),
        features=features,
        block=block,
        qc_options=qc_options,
        norm_options=norm_options,
        feature_selection_options=feature_selection_options,
        pca_options=pca_options,
        tsne_options=tsne_options,
        umap_options=umap_options,
        cluster_options=cluster_options,
        marker_options=marker_options,
        seed=seed,
        num_threads=num_threads,
    )
