from mattress import tatamize
from . import quality_control as qc
from . import normalization as norm
from . import feature_selection as feat
from . import dimensionality_reduction as dimred
from . import clustering as clust
from . import marker_detection as mark

from multiprocessing import Queue, Process

def tsne_mp(q, *args, **kwargs):
    res = dimred.run_tsne(*args, **kwargs)
    q.put({ "x": res.x, "y": res.y, "type": "tsne" })

def umap_mp(q, *args, **kwargs):
    res = dimred.run_umap(*args, **kwargs)
    q.put({ "x": res.x, "y": res.y, "type": "umap" })

def analyze(
    mat, 
    features, 

    qc_mito_subset = None,
    qc_num_mads = 3,
    qc_custom_thresholds = None,

    feat_span = 0.3,
    feat_num_hvgs = 4000,

    pca_rank = 20,
    pca_args = {},

    tsne_args = {},
    umap_args = {},

    snn_build_args = {},
    snn_resolution = 1,

    num_threads = 1
):
    subsets = {}
    if qc_mito_subset:
        if isinstance(qc_mito_subset, str):
            subsets["mito"] = qc.guess_mito_from_symbols(features, prefix)
        else:
            raise ValueError("not supported yet")

    metrics = qc.per_cell_rna_qc_metrics(mat, subsets, num_threads=num_threads)
    thresholds = qc.suggest_rna_qc_filters(metrics, num_mads = qc_num_mads)

    if qc_custom_thresholds:
        for col in thresholds.columnNames:
            if col in qc_custom_thresholds:
                thresholds.column(col).fill(qc_custom_thresholds[col])

    filter = qc.create_rna_qc_filter(metrics, thresholds)

    ptr = tatamize(mat)
    filtered = qc.filter_cells(ptr, filter)
    normed = norm.log_norm_counts(filtered, num_threads=num_threads)

    varstats = feat.model_gene_variances(normed, span=feat_span, num_threads=num_threads)
    selected = feat.choose_hvgs(varstats.column("residuals"), number=feat_num_hvgs) 
    pca = dimred.run_pca(normed, rank=pca_rank, subset=selected, **pca_args, num_threads=num_threads)

    Q = Queue()
    tsne_p = Process(target=tsne_mp, args=(Q, pca.principal_components,), kwargs=tsne_args)
    umap_p = Process(target=umap_mp, args=(Q, pca.principal_components,), kwargs=umap_args)

    tsne_p.start()
    umap_p.start()

    graph = clust.build_snn_graph(pca.principal_components, **snn_build_args)
    clusters = graph.community_multilevel(resolution=snn_resolution).membership
    markers = mark.score_markers(normed, clusters, num_threads=max(1, num_threads - 2))

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
        "marker_detection": markers
    }
