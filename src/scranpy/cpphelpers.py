# DO NOT MODIFY: this is automatically generated by the ctypes-wrapper

import os
import ctypes as ct

def catch_errors(f):
    def wrapper(*args):
        errcode = ct.c_int32(0)
        errmsg = ct.c_char_p(0)
        output = f(*args, ct.byref(errcode), ct.byref(errmsg))
        if errcode.value != 0:
            msg = errmsg.value.decode('ascii')
            lib.free_error_message(errmsg)
            raise RuntimeError(msg)
        return output
    return wrapper

# TODO: surely there's a better way than whatever this is.
dirname = os.path.dirname(os.path.abspath(__file__))
contents = os.listdir(dirname)
lib = None
for x in contents:
    if x.startswith('core') and not x.endswith("py"):
        lib = ct.CDLL(os.path.join(dirname, x))
        break

if lib is None:
    raise ImportError("failed to find the core.* module")

lib.free_error_message.argtypes = [ ct.POINTER(ct.c_char_p) ]

lib.py_build_neighbor_index.restype = ct.c_void_p
lib.py_build_neighbor_index.argtypes = [
    ct.c_int32,
    ct.c_int32,
    ct.c_void_p,
    ct.c_uint8,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_build_snn_graph_from_nn_index.restype = ct.c_void_p
lib.py_build_snn_graph_from_nn_index.argtypes = [
    ct.c_void_p,
    ct.c_int32,
    ct.c_char_p,
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_build_snn_graph_from_nn_results.restype = ct.c_void_p
lib.py_build_snn_graph_from_nn_results.argtypes = [
    ct.c_void_p,
    ct.c_char_p,
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_clone_tsne_status.restype = ct.c_void_p
lib.py_clone_tsne_status.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_clone_umap_status.restype = ct.c_void_p
lib.py_clone_umap_status.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_multibatch_pca_coordinates.restype = ct.c_void_p
lib.py_fetch_multibatch_pca_coordinates.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_multibatch_pca_num_dims.restype = ct.c_int32
lib.py_fetch_multibatch_pca_num_dims.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_multibatch_pca_total_variance.restype = ct.c_double
lib.py_fetch_multibatch_pca_total_variance.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_multibatch_pca_variance_explained.restype = ct.c_void_p
lib.py_fetch_multibatch_pca_variance_explained.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_neighbor_index_ndim.restype = ct.c_int32
lib.py_fetch_neighbor_index_ndim.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_neighbor_index_nobs.restype = ct.c_int32
lib.py_fetch_neighbor_index_nobs.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_neighbor_results_k.restype = ct.c_int32
lib.py_fetch_neighbor_results_k.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_neighbor_results_nobs.restype = ct.c_int32
lib.py_fetch_neighbor_results_nobs.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_neighbor_results_single.argtypes = [
    ct.c_void_p,
    ct.c_int32,
    ct.c_void_p,
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_residual_pca_coordinates.restype = ct.c_void_p
lib.py_fetch_residual_pca_coordinates.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_residual_pca_num_dims.restype = ct.c_int32
lib.py_fetch_residual_pca_num_dims.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_residual_pca_total_variance.restype = ct.c_double
lib.py_fetch_residual_pca_total_variance.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_residual_pca_variance_explained.restype = ct.c_void_p
lib.py_fetch_residual_pca_variance_explained.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_simple_pca_coordinates.restype = ct.c_void_p
lib.py_fetch_simple_pca_coordinates.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_simple_pca_num_dims.restype = ct.c_int32
lib.py_fetch_simple_pca_num_dims.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_simple_pca_total_variance.restype = ct.c_double
lib.py_fetch_simple_pca_total_variance.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_simple_pca_variance_explained.restype = ct.c_void_p
lib.py_fetch_simple_pca_variance_explained.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_snn_graph_edges.restype = ct.c_int32
lib.py_fetch_snn_graph_edges.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_snn_graph_indices.restype = ct.c_void_p
lib.py_fetch_snn_graph_indices.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_snn_graph_weights.restype = ct.c_void_p
lib.py_fetch_snn_graph_weights.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_tsne_status_iteration.restype = ct.c_int32
lib.py_fetch_tsne_status_iteration.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_tsne_status_nobs.restype = ct.c_int32
lib.py_fetch_tsne_status_nobs.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_umap_status_epoch.restype = ct.c_int32
lib.py_fetch_umap_status_epoch.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_umap_status_nobs.restype = ct.c_int32
lib.py_fetch_umap_status_nobs.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_fetch_umap_status_num_epochs.restype = ct.c_int32
lib.py_fetch_umap_status_num_epochs.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_find_nearest_neighbors.restype = ct.c_void_p
lib.py_find_nearest_neighbors.argtypes = [
    ct.c_void_p,
    ct.c_int32,
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_free_multibatch_pca.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_free_neighbor_index.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_free_neighbor_results.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_free_residual_pca.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_free_simple_pca.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_free_snn_graph.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_free_tsne_status.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_free_umap_status.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_initialize_tsne.restype = ct.c_void_p
lib.py_initialize_tsne.argtypes = [
    ct.c_void_p,
    ct.c_double,
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_initialize_umap.restype = ct.c_void_p
lib.py_initialize_umap.argtypes = [
    ct.c_void_p,
    ct.c_int32,
    ct.c_double,
    ct.c_void_p,
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_log_norm_counts.restype = ct.c_void_p
lib.py_log_norm_counts.argtypes = [
    ct.c_void_p,
    ct.c_uint8,
    ct.c_void_p,
    ct.c_uint8,
    ct.c_void_p,
    ct.c_uint8,
    ct.c_uint8,
    ct.c_uint8,
    ct.c_int,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_model_gene_variances.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_double,
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_model_gene_variances_blocked.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int32,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_double,
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_per_cell_rna_qc_metrics.argtypes = [
    ct.c_void_p,
    ct.c_int32,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_perplexity_to_k.restype = ct.c_int32
lib.py_perplexity_to_k.argtypes = [
    ct.c_double,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_randomize_tsne_start.argtypes = [
    ct.c_size_t,
    ct.c_void_p,
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_run_multibatch_pca.restype = ct.c_void_p
lib.py_run_multibatch_pca.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_uint8,
    ct.c_uint8,
    ct.c_int32,
    ct.c_uint8,
    ct.c_void_p,
    ct.c_uint8,
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_run_residual_pca.restype = ct.c_void_p
lib.py_run_residual_pca.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_uint8,
    ct.c_int32,
    ct.c_uint8,
    ct.c_void_p,
    ct.c_uint8,
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_run_simple_pca.restype = ct.c_void_p
lib.py_run_simple_pca.argtypes = [
    ct.c_void_p,
    ct.c_int32,
    ct.c_uint8,
    ct.c_void_p,
    ct.c_uint8,
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_run_tsne.argtypes = [
    ct.c_void_p,
    ct.c_int32,
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_run_umap.argtypes = [
    ct.c_void_p,
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_score_markers.argtypes = [
    ct.c_void_p,
    ct.c_int32,
    ct.c_void_p,
    ct.c_int32,
    ct.c_void_p,
    ct.c_uint8,
    ct.c_double,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_serialize_neighbor_results.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_suggest_rna_qc_filters.argtypes = [
    ct.c_int32,
    ct.c_int32,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int32,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_double,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

lib.py_unserialize_neighbor_results.restype = ct.c_void_p
lib.py_unserialize_neighbor_results.argtypes = [
    ct.c_int32,
    ct.c_int32,
    ct.c_void_p,
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p)
]

def build_neighbor_index(ndim, nobs, ptr, approximate):
    return catch_errors(lib.py_build_neighbor_index)(ndim, nobs, ptr, approximate)

def build_snn_graph_from_nn_index(x, num_neighbors, weight_scheme, num_threads):
    return catch_errors(lib.py_build_snn_graph_from_nn_index)(x, num_neighbors, weight_scheme, num_threads)

def build_snn_graph_from_nn_results(x, weight_scheme, num_threads):
    return catch_errors(lib.py_build_snn_graph_from_nn_results)(x, weight_scheme, num_threads)

def clone_tsne_status(ptr):
    return catch_errors(lib.py_clone_tsne_status)(ptr)

def clone_umap_status(ptr, cloned):
    return catch_errors(lib.py_clone_umap_status)(ptr, cloned)

def fetch_multibatch_pca_coordinates(x):
    return catch_errors(lib.py_fetch_multibatch_pca_coordinates)(x)

def fetch_multibatch_pca_num_dims(x):
    return catch_errors(lib.py_fetch_multibatch_pca_num_dims)(x)

def fetch_multibatch_pca_total_variance(x):
    return catch_errors(lib.py_fetch_multibatch_pca_total_variance)(x)

def fetch_multibatch_pca_variance_explained(x):
    return catch_errors(lib.py_fetch_multibatch_pca_variance_explained)(x)

def fetch_neighbor_index_ndim(ptr):
    return catch_errors(lib.py_fetch_neighbor_index_ndim)(ptr)

def fetch_neighbor_index_nobs(ptr):
    return catch_errors(lib.py_fetch_neighbor_index_nobs)(ptr)

def fetch_neighbor_results_k(ptr0):
    return catch_errors(lib.py_fetch_neighbor_results_k)(ptr0)

def fetch_neighbor_results_nobs(ptr):
    return catch_errors(lib.py_fetch_neighbor_results_nobs)(ptr)

def fetch_neighbor_results_single(ptr0, i, outdex, outdist):
    return catch_errors(lib.py_fetch_neighbor_results_single)(ptr0, i, outdex, outdist)

def fetch_residual_pca_coordinates(x):
    return catch_errors(lib.py_fetch_residual_pca_coordinates)(x)

def fetch_residual_pca_num_dims(x):
    return catch_errors(lib.py_fetch_residual_pca_num_dims)(x)

def fetch_residual_pca_total_variance(x):
    return catch_errors(lib.py_fetch_residual_pca_total_variance)(x)

def fetch_residual_pca_variance_explained(x):
    return catch_errors(lib.py_fetch_residual_pca_variance_explained)(x)

def fetch_simple_pca_coordinates(x):
    return catch_errors(lib.py_fetch_simple_pca_coordinates)(x)

def fetch_simple_pca_num_dims(x):
    return catch_errors(lib.py_fetch_simple_pca_num_dims)(x)

def fetch_simple_pca_total_variance(x):
    return catch_errors(lib.py_fetch_simple_pca_total_variance)(x)

def fetch_simple_pca_variance_explained(x):
    return catch_errors(lib.py_fetch_simple_pca_variance_explained)(x)

def fetch_snn_graph_edges(ptr):
    return catch_errors(lib.py_fetch_snn_graph_edges)(ptr)

def fetch_snn_graph_indices(ptr):
    return catch_errors(lib.py_fetch_snn_graph_indices)(ptr)

def fetch_snn_graph_weights(ptr):
    return catch_errors(lib.py_fetch_snn_graph_weights)(ptr)

def fetch_tsne_status_iteration(ptr):
    return catch_errors(lib.py_fetch_tsne_status_iteration)(ptr)

def fetch_tsne_status_nobs(ptr):
    return catch_errors(lib.py_fetch_tsne_status_nobs)(ptr)

def fetch_umap_status_epoch(ptr):
    return catch_errors(lib.py_fetch_umap_status_epoch)(ptr)

def fetch_umap_status_nobs(ptr):
    return catch_errors(lib.py_fetch_umap_status_nobs)(ptr)

def fetch_umap_status_num_epochs(ptr):
    return catch_errors(lib.py_fetch_umap_status_num_epochs)(ptr)

def find_nearest_neighbors(index, k, nthreads):
    return catch_errors(lib.py_find_nearest_neighbors)(index, k, nthreads)

def free_multibatch_pca(x):
    return catch_errors(lib.py_free_multibatch_pca)(x)

def free_neighbor_index(ptr):
    return catch_errors(lib.py_free_neighbor_index)(ptr)

def free_neighbor_results(ptr):
    return catch_errors(lib.py_free_neighbor_results)(ptr)

def free_residual_pca(x):
    return catch_errors(lib.py_free_residual_pca)(x)

def free_simple_pca(x):
    return catch_errors(lib.py_free_simple_pca)(x)

def free_snn_graph(ptr):
    return catch_errors(lib.py_free_snn_graph)(ptr)

def free_tsne_status(ptr):
    return catch_errors(lib.py_free_tsne_status)(ptr)

def free_umap_status(ptr):
    return catch_errors(lib.py_free_umap_status)(ptr)

def initialize_tsne(neighbors, perplexity, nthreads):
    return catch_errors(lib.py_initialize_tsne)(neighbors, perplexity, nthreads)

def initialize_umap(neighbors, num_epochs, min_dist, Y, nthreads):
    return catch_errors(lib.py_initialize_umap)(neighbors, num_epochs, min_dist, Y, nthreads)

def log_norm_counts(mat0, use_block, block, use_size_factors, size_factors, center, allow_zeros, allow_non_finite, num_threads):
    return catch_errors(lib.py_log_norm_counts)(mat0, use_block, block, use_size_factors, size_factors, center, allow_zeros, allow_non_finite, num_threads)

def model_gene_variances(mat, means, variances, fitted, residuals, span, num_threads):
    return catch_errors(lib.py_model_gene_variances)(mat, means, variances, fitted, residuals, span, num_threads)

def model_gene_variances_blocked(mat, ave_means, ave_detected, ave_fitted, ave_residuals, num_blocks, block, block_means, block_variances, block_fitted, block_residuals, span, num_threads):
    return catch_errors(lib.py_model_gene_variances_blocked)(mat, ave_means, ave_detected, ave_fitted, ave_residuals, num_blocks, block, block_means, block_variances, block_fitted, block_residuals, span, num_threads)

def per_cell_rna_qc_metrics(mat, num_subsets, subset_ptrs, sum_output, detected_output, subset_output, num_threads):
    return catch_errors(lib.py_per_cell_rna_qc_metrics)(mat, num_subsets, subset_ptrs, sum_output, detected_output, subset_output, num_threads)

def perplexity_to_k(perplexity):
    return catch_errors(lib.py_perplexity_to_k)(perplexity)

def randomize_tsne_start(n, Y, seed):
    return catch_errors(lib.py_randomize_tsne_start)(n, Y, seed)

def run_multibatch_pca(mat, block, use_residuals, equal_weights, number, use_subset, subset, scale, num_threads):
    return catch_errors(lib.py_run_multibatch_pca)(mat, block, use_residuals, equal_weights, number, use_subset, subset, scale, num_threads)

def run_residual_pca(mat, block, equal_weights, number, use_subset, subset, scale, num_threads):
    return catch_errors(lib.py_run_residual_pca)(mat, block, equal_weights, number, use_subset, subset, scale, num_threads)

def run_simple_pca(mat, number, use_subset, subset, scale, num_threads):
    return catch_errors(lib.py_run_simple_pca)(mat, number, use_subset, subset, scale, num_threads)

def run_tsne(status, maxiter, Y):
    return catch_errors(lib.py_run_tsne)(status, maxiter, Y)

def run_umap(status, max_epoch):
    return catch_errors(lib.py_run_umap)(status, max_epoch)

def score_markers(mat, num_clusters, clusters, num_blocks, block, do_auc, threshold, raw_means, raw_detected, raw_cohen, raw_auc, raw_lfc, raw_delta_detected, num_threads):
    return catch_errors(lib.py_score_markers)(mat, num_clusters, clusters, num_blocks, block, do_auc, threshold, raw_means, raw_detected, raw_cohen, raw_auc, raw_lfc, raw_delta_detected, num_threads)

def serialize_neighbor_results(ptr0, outdex, outdist):
    return catch_errors(lib.py_serialize_neighbor_results)(ptr0, outdex, outdist)

def suggest_rna_qc_filters(num_cells, num_subsets, sums, detected, subset_proportions, num_blocks, block, sums_out, detected_out, subset_proportions_out, nmads):
    return catch_errors(lib.py_suggest_rna_qc_filters)(num_cells, num_subsets, sums, detected, subset_proportions, num_blocks, block, sums_out, detected_out, subset_proportions_out, nmads)

def unserialize_neighbor_results(nobs, k, indices, distances):
    return catch_errors(lib.py_unserialize_neighbor_results)(nobs, k, indices, distances)