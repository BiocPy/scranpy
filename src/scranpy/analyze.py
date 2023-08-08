from functools import singledispatch
from multiprocessing import Process, Queue
from typing import Any, Mapping, Optional, Sequence

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
import inspect

__author__ = "ltla, jkanche"
__copyright__ = "ltla"
__license__ = "MIT"


def tsne_mp(q: Queue, *args, **kwargs):
    """Run the t-SNe in parallel along with other steps.

    Args:
        q (Queue): task queue.
        *args, *kwargs: arguments to pass on to `run_tsne` function.
    """
    res = dimred.run_tsne(*args, **kwargs)
    q.put({"x": res.x, "y": res.y, "type": "tsne"})


def umap_mp(q, *args, **kwargs):
    """Run the UMAP in parallel along with other steps.

    Args:
        q (Queue): task queue.
        *args, *kwargs: arguments to pass on to `run_umap` function.
    """
    res = dimred.run_umap(*args, **kwargs)
    q.put({"x": res.x, "y": res.y, "type": "umap"})


def __analyze(
    matrix: Any,
    features: Sequence[str],
    qc_mito_subset: Optional[str] = None,
    qc_num_mads: int = 3,
    qc_custom_thresholds: BiocFrame = None,
    feat_span: float = 0.3,
    feat_num_hvgs: int = 4000,
    pca_rank: int = 20,
    pca_args: Mapping = {},
    tsne_args: Mapping = {},
    umap_args: Mapping = {},
    snn_build_args: Mapping = {},
    snn_resolution: float = 1,
    num_threads: int = 1,
) -> Mapping:
    subsets = {}
    if qc_mito_subset is not None:
        if isinstance(qc_mito_subset, str):
            subsets["mito"] = qc.guess_mito_from_symbols(features, qc_mito_subset)
        else:
            raise ValueError("not supported yet")

    metrics = qc.per_cell_rna_qc_metrics(matrix, subsets, num_threads=num_threads)
    thresholds = qc.suggest_rna_qc_filters(metrics, num_mads=qc_num_mads)

    if qc_custom_thresholds is not None:
        for col in thresholds.columnNames:
            if col in qc_custom_thresholds:
                thresholds.column(col).fill(qc_custom_thresholds[col])

    filter = qc.create_rna_qc_filter(metrics, thresholds)

    ptr = tatamize(matrix)
    filtered = qc.filter_cells(ptr, filter)
    normed = norm.log_norm_counts(filtered, num_threads=num_threads)

    varstats = feat.model_gene_variances(
        normed, span=feat_span, num_threads=num_threads
    )
    selected = feat.choose_hvgs(varstats.column("residuals"), number=feat_num_hvgs)
    pca = dimred.run_pca(
        normed, rank=pca_rank, subset=selected, **pca_args, num_threads=num_threads
    )

    idx = nn.build_neighbor_index(pca.principal_components, approximate=True)

    umap_nn = umap_args.num_neighbors if "num_neighbors" in umap_args else inspect.signature(dimred.initialize_umap).parameters["num_neighbors"].default
    tsne_perplexity = tsne_args.perplexity if "perplexity" in tsne_args else inspect.signature(dimred.initialize_tsne).parameters["perplexity"].default
    snn_nn = snn_build_args.num_neighbors if "num_neighbors" in snn_build_args else inspect.signature(clust.build_snn_graph).parameters["num_neighbors"].default
    tsne_nn = dimred.tsne_perplexity_to_neighbors(tsne_perplexity)
    nn_dict = {}
    for k in set([umap_nn, tsne_nn, snn_nn]):
        nn_dict[k] = nn.find_nearest_neighbors(idx, k=k, num_threads=num_threads)

    Q = Queue()
    tsne_p = Process(
        target=tsne_mp,
        args=(
            Q,
            nn_dict[tsne_nn] 
        ),
        kwargs=tsne_args,
    )
    umap_p = Process(
        target=umap_mp,
        args=(
            Q,
            nn_dict[umap_nn]
        ),
        kwargs=umap_args,
    )

    tsne_p.start()
    umap_p.start()

    remaining_threads = max(1, num_threads - 2)
    graph = clust.build_snn_graph(nn_dict[snn_nn], **snn_build_args, num_threads=remaining_threads)
    clusters = graph.community_multilevel(resolution=snn_resolution).membership
    markers = mark.score_markers(normed, clusters, num_threads=remaining_threads)

    res1 = Q.get()
    res2 = Q.get()
    if res1["type"] == "tsne":
        tsne = res1
        umap = res2
    else:
        tsne = res2
        umap = res1

    del tsne["type"]
    del umap["type"]

    tsne_p.join()
    umap_p.join()

    return {
        "qc_metrics": metrics,
        "qc_thresholds": thresholds,
        "qc_filter": filter,
        "variances": varstats,
        "hvgs": selected,
        "pca": pca,
        "tsne": tsne,
        "umap": umap,
        "clustering": clusters,
        "marker_detection": markers,
    }


@singledispatch
def analyze(
    matrix: Any,
    features: Sequence[str],
    qc_mito_subset: Optional[str] = None,
    qc_num_mads: int = 3,
    qc_custom_thresholds: BiocFrame = None,
    feat_span: float = 0.3,
    feat_num_hvgs: int = 4000,
    pca_rank: int = 20,
    pca_args: Mapping = {},
    tsne_args: Mapping = {},
    umap_args: Mapping = {},
    snn_build_args: Mapping = {},
    snn_resolution: float = 1,
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
        matrix (MatrixTypes): Count matrix.
        features (Sequence[str]): Features information for the rows of the matrix.
        qc_mito_subset (str, optional): Prefix to filter
            mitochindrial genes. Defaults to None.
        qc_num_mads (int, optional): Number of median absolute deviations to
            filter low-quality cells. Defaults to 3.
        qc_custom_thresholds (BiocFrame, optional): Suggested (or modified) filters from
            `suggest_rna_qc_filters` function. Defaults to None.
        feat_span (float, optional): Span to use for the LOWESS trend fitting.
            Defaults to 0.3.
        feat_num_hvgs (int, optional): Number of HVGs to pick. Defaults to 4000.
        pca_rank (int, optional): Number of pc to compute. Defaults to 20.
        pca_args (Mapping, optional): Arguments to `run_pca` function. Defaults to {}.
        tsne_args (Mapping, optional): Arguments to `run_tsne` function. Defaults to {}.
        umap_args (Mapping, optional): Arguments to `run_umap` function. Defaults to {}.
        snn_build_args (Mapping, optional): Arguments to `build_snn_graph` function.
            Defaults to {}.
        snn_resolution (float, optional): Resolution to use for identifying cluters
            This arguments is forwarded to igraph's `community_multilevel` function.
            Defaults to 1.
        num_threads (int, optional): Number of threads to use. Defaults to 1.

    Raises:
        ValueError: when arguments don't meet expectations.

    Returns:
        Mapping: Results from various steps.
    """
    if is_matrix_expected_type(matrix):
        return __analyze(
            matrix,
            features=features,
            qc_mito_subset=qc_mito_subset,
            qc_num_mads=qc_num_mads,
            qc_custom_thresholds=qc_custom_thresholds,
            feat_span=feat_span,
            feat_num_hvgs=feat_num_hvgs,
            pca_rank=pca_rank,
            pca_args=pca_args,
            tsne_args=tsne_args,
            umap_args=umap_args,
            snn_build_args=snn_build_args,
            snn_resolution=snn_resolution,
            num_threads=num_threads,
        )
    else:
        raise NotImplementedError(
            f"analyze is not supported for objects of class: {type(matrix)}"
        )


@analyze.register
def analyze_sce(
    matrix: SingleCellExperiment,
    features: Sequence[str],
    qc_mito_subset: Optional[str] = None,
    qc_num_mads: int = 3,
    qc_custom_thresholds: BiocFrame = None,
    feat_span: float = 0.3,
    feat_num_hvgs: int = 4000,
    pca_rank: int = 20,
    pca_args: Mapping = {},
    tsne_args: Mapping = {},
    umap_args: Mapping = {},
    snn_build_args: Mapping = {},
    snn_resolution: float = 1,
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
        matrix (SingleCellExperiment): A `SingleCellExperiment` object.
        features (Sequence[str]): Features information for the rows of the matrix.
        qc_mito_subset (str, optional): Prefix to filter
            mitochindrial genes. Defaults to None.
        qc_num_mads (int, optional): Number of median absolute deviations to
            filter low-quality cells. Defaults to 3.
        qc_custom_thresholds (BiocFrame, optional): Suggested (or modified) filters from
            `suggest_rna_qc_filters` function. Defaults to None.
        feat_span (float, optional): Span to use for the LOWESS trend fitting.
            Defaults to 0.3.
        feat_num_hvgs (int, optional): Number of HVGs to pick. Defaults to 4000.
        pca_rank (int, optional): Number of pc to compute. Defaults to 20.
        pca_args (Mapping, optional): Arguments to `run_pca` function. Defaults to {}.
        tsne_args (Mapping, optional): Arguments to `run_tsne` function. Defaults to {}.
        umap_args (Mapping, optional): Arguments to `run_umap` function. Defaults to {}.
        snn_build_args (Mapping, optional): Arguments to `build_snn_graph` function.
            Defaults to {}.
        snn_resolution (float, optional): Resolution to use for identifying cluters
            This arguments is forwarded to igraph's `community_multilevel` function.
            Defaults to 1.
        num_threads (int, optional): Number of threads to use. Defaults to 1.

    Raises:
        ValueError: when arguments don't meet expectations.

    Returns:
        Mapping: Tesults from various steps.
    """
    if "counts" not in matrix.assayNames:
        raise ValueError("SCE does not contain a 'count' matrix.")

    return __analyze(
        matrix.assay("counts"),
        features=features,
        qc_mito_subset=qc_mito_subset,
        qc_num_mads=qc_num_mads,
        qc_custom_thresholds=qc_custom_thresholds,
        feat_span=feat_span,
        feat_num_hvgs=feat_num_hvgs,
        pca_rank=pca_rank,
        pca_args=pca_args,
        tsne_args=tsne_args,
        umap_args=umap_args,
        snn_build_args=snn_build_args,
        snn_resolution=snn_resolution,
        num_threads=num_threads,
    )
