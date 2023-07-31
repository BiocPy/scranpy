import ctypes as ct
import os

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


def load_dll() -> ct.CDLL:
    """load the shared library.

    usually starts with core.<platform>.<so or dll>.

    Returns:
        ct.CDLL: shared object.
    """
    dirname = os.path.dirname(os.path.abspath(__file__))
    contents = os.listdir(dirname)
    for x in contents:
        if x.startswith("core") and not x.endswith("py"):
            return ct.CDLL(os.path.join(dirname, x))

    raise Exception("Cannot find the shared object file! Report this issue on github.")


lib = load_dll()
lib.per_cell_rna_qc_metrics.argtypes = [
    ct.c_void_p,
    ct.c_int,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int,
]

lib.log_norm_counts.restype = ct.c_void_p
lib.log_norm_counts.argtypes = [
    ct.c_void_p,
    ct.c_uint8,
    ct.c_void_p,
    ct.c_uint8,
    ct.c_void_p,
    ct.c_uint8,
    ct.c_uint8,
    ct.c_uint8,
    ct.c_int,
]

lib.suggest_rna_qc_filters.argtypes = [
    ct.c_int,
    ct.c_int,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_double,
]

lib.model_gene_variances.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_double,
    ct.c_int,
]

lib.model_gene_variances_blocked.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_double,
    ct.c_int,
]

lib.build_neighbor_index.restype = ct.c_void_p
lib.build_neighbor_index.argtypes = [ct.c_int, ct.c_int, ct.c_void_p, ct.c_uint8]
lib.fetch_neighbor_index_ndim.restype = ct.c_int
lib.fetch_neighbor_index_ndim.argtypes = [ct.c_void_p]
lib.fetch_neighbor_index_nobs.restype = ct.c_int
lib.fetch_neighbor_index_nobs.argtypes = [ct.c_void_p]
lib.free_neighbor_index.argtypes = [ct.c_void_p]

lib.find_nearest_neighbors.restype = ct.c_void_p
lib.find_nearest_neighbors.argtypes = [ct.c_void_p, ct.c_int, ct.c_int]
lib.fetch_neighbor_results_k.restype = ct.c_int
lib.fetch_neighbor_results_k.argtypes = [ct.c_void_p]
lib.fetch_neighbor_results_nobs.restype = ct.c_int
lib.fetch_neighbor_results_nobs.argtypes = [ct.c_void_p]
lib.fetch_neighbor_results_single.argtypes = [
    ct.c_void_p,
    ct.c_int,
    ct.c_void_p,
    ct.c_void_p,
]
lib.free_neighbor_results.argtypes = [ct.c_void_p]
lib.serialize_neighbor_results.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p]
lib.unserialize_neighbor_results.restype = ct.c_void_p
lib.unserialize_neighbor_results.argtypes = [
    ct.c_int,
    ct.c_int,
    ct.c_void_p,
    ct.c_void_p,
]
lib.fetch_simple_pca_coordinates.argtypes = [ct.c_void_p]
lib.fetch_simple_pca_coordinates.restype = ct.c_void_p
lib.fetch_simple_pca_variance_explained.argtypes = [ct.c_void_p]
lib.fetch_simple_pca_variance_explained.restype = ct.c_void_p
lib.fetch_simple_pca_total_variance.argtypes = [ct.c_void_p]
lib.fetch_simple_pca_total_variance.restype = ct.c_double
lib.fetch_simple_pca_num_dims.argtypes = [ct.c_void_p]
lib.fetch_simple_pca_num_dims.restype = ct.c_int
lib.free_simple_pca.argtypes = [ct.c_void_p]
lib.run_simple_pca.argtypes = [
    ct.c_void_p,
    ct.c_int,
    ct.c_uint8,
    ct.c_void_p,
    ct.c_uint8,
    ct.c_int,
]
lib.run_simple_pca.restype = ct.c_void_p

lib.fetch_residual_pca_coordinates.argtypes = [ct.c_void_p]
lib.fetch_residual_pca_coordinates.restype = ct.c_void_p
lib.fetch_residual_pca_variance_explained.argtypes = [ct.c_void_p]
lib.fetch_residual_pca_variance_explained.restype = ct.c_void_p
lib.fetch_residual_pca_total_variance.argtypes = [ct.c_void_p]
lib.fetch_residual_pca_total_variance.restype = ct.c_double
lib.fetch_residual_pca_num_dims.argtypes = [ct.c_void_p]
lib.fetch_residual_pca_num_dims.restype = ct.c_int
lib.free_residual_pca.argtypes = [ct.c_void_p]
lib.run_residual_pca.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_uint8,
    ct.c_int,
    ct.c_uint8,
    ct.c_void_p,
    ct.c_uint8,
    ct.c_int,
]
lib.run_residual_pca.restype = ct.c_void_p

lib.fetch_multibatch_pca_coordinates.argtypes = [ct.c_void_p]
lib.fetch_multibatch_pca_coordinates.restype = ct.c_void_p
lib.fetch_multibatch_pca_variance_explained.argtypes = [ct.c_void_p]
lib.fetch_multibatch_pca_variance_explained.restype = ct.c_void_p
lib.fetch_multibatch_pca_total_variance.argtypes = [ct.c_void_p]
lib.fetch_multibatch_pca_total_variance.restype = ct.c_double
lib.fetch_multibatch_pca_num_dims.argtypes = [ct.c_void_p]
lib.fetch_multibatch_pca_num_dims.restype = ct.c_int
lib.free_multibatch_pca.argtypes = [ct.c_void_p]
lib.run_multibatch_pca.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_uint8,
    ct.c_uint8,
    ct.c_int,
    ct.c_uint8,
    ct.c_void_p,
    ct.c_uint8,
    ct.c_int,
]
lib.run_multibatch_pca.restype = ct.c_void_p

lib.build_snn_graph_from_nn_results.restype = ct.c_void_p
lib.build_snn_graph_from_nn_results.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_int]
lib.build_snn_graph_from_nn_index.restype = ct.c_void_p
lib.build_snn_graph_from_nn_index.argtypes = [
    ct.c_void_p,
    ct.c_int,
    ct.c_char_p,
    ct.c_int,
]
lib.fetch_snn_graph_edges.restype = ct.c_int
lib.fetch_snn_graph_edges.argtypes = [ct.c_void_p]
lib.fetch_snn_graph_indices.restype = ct.c_void_p
lib.fetch_snn_graph_indices.argtypes = [ct.c_void_p]
lib.fetch_snn_graph_weights.restype = ct.c_void_p
lib.fetch_snn_graph_weights.argtypes = [ct.c_void_p]
lib.free_snn_graph.argtypes = [ct.c_void_p]

lib.initialize_tsne.restype = ct.c_void_p
lib.initialize_tsne.argtypes = [ct.c_void_p, ct.c_double, ct.c_int]
lib.randomize_tsne_start.argtypes = [ct.c_int, ct.c_void_p, ct.c_int]
lib.fetch_tsne_status_iteration.restype = ct.c_int
lib.fetch_tsne_status_iteration.argtypes = [ct.c_void_p]
lib.fetch_tsne_status_nobs.restype = ct.c_int
lib.fetch_tsne_status_nobs.argtypes = [ct.c_void_p]
lib.free_tsne_status.argtypes = [ct.c_void_p]
lib.clone_tsne_status.argtypes = [ct.c_void_p]
lib.clone_tsne_status.restype = ct.c_void_p
lib.perplexity_to_k.restype = ct.c_int
lib.perplexity_to_k.argtypes = [ct.c_double]
lib.run_tsne.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p]

lib.initialize_umap.restype = ct.c_void_p
lib.initialize_umap.argtypes = [
    ct.c_void_p,
    ct.c_int,
    ct.c_double,
    ct.c_void_p,
    ct.c_int,
]
lib.fetch_umap_status_nobs.restype = ct.c_int
lib.fetch_umap_status_nobs.argtypes = [ct.c_void_p]
lib.fetch_umap_status_epoch.restype = ct.c_int
lib.fetch_umap_status_epoch.argtypes = [ct.c_void_p]
lib.fetch_umap_status_num_epochs.restype = ct.c_int
lib.fetch_umap_status_num_epochs.argtypes = [ct.c_void_p]
lib.free_umap_status.argtypes = [ct.c_void_p]
lib.clone_umap_status.argtypes = [ct.c_void_p, ct.c_void_p]
lib.clone_umap_status.restype = ct.c_void_p
lib.run_umap.argtypes = [ct.c_void_p, ct.c_int]

lib.score_markers.argtypes = [
    ct.c_void_p,
    ct.c_int,
    ct.c_void_p,
    ct.c_int,
    ct.c_void_p,
    ct.c_uint8,
    ct.c_double,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int,
]
