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
lib.build_neighbor_index.argtypes = [
    ct.c_int,
    ct.c_int,
    ct.c_void_p,
    ct.c_uint8
]
lib.fetch_neighbor_index_ndim.restype = ct.c_int
lib.fetch_neighbor_index_ndim.argtypes = [ ct.c_void_p ]
lib.fetch_neighbor_index_nobs.restype = ct.c_int
lib.fetch_neighbor_index_nobs.argtypes = [ ct.c_void_p ]
lib.free_neighbor_index.argtypes = [ ct.c_void_p ]

lib.find_nearest_neighbors.restype = ct.c_void_p
lib.find_nearest_neighbors.argtypes = [
    ct.c_void_p,
    ct.c_int,
    ct.c_int
]
lib.fetch_neighbor_results_k.restype = ct.c_int
lib.fetch_neighbor_results_k.argtypes = [ ct.c_void_p ]
lib.fetch_neighbor_results_nobs.restype = ct.c_int
lib.fetch_neighbor_results_nobs.argtypes = [ ct.c_void_p ]
lib.fetch_neighbor_results_indices.argtypes = [ 
    ct.c_void_p, 
    ct.c_int, 
    ct.c_void_p, 
    ct.c_void_p 
]
lib.free_neighbor_results.argtypes = [ ct.c_void_p ]
