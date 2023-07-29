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

lib.fetch_simple_pca_coordinates.argtypes = [ ct.c_void_p ]
lib.fetch_simple_pca_coordinates.restype = ct.c_void_p
lib.fetch_simple_pca_variance_explained.argtypes = [ ct.c_void_p ]
lib.fetch_simple_pca_variance_explained.restype = ct.c_void_p
lib.fetch_simple_pca_total_variance.argtypes = [ ct.c_void_p ]
lib.fetch_simple_pca_total_variance.restype = ct.c_double
lib.fetch_simple_pca_num_dims.argtypes = [ ct.c_void_p ]
lib.fetch_simple_pca_num_dims.restype = ct.c_int
lib.free_simple_pca.argtypes = [ ct.c_void_p ]
lib.run_simple_pca.argtypes = [ 
    ct.c_void_p,
    ct.c_int,
    ct.c_uint8,
    ct.c_void_p,
    ct.c_uint8,
    ct.c_int
]
lib.run_simple_pca.restype = ct.c_void_p

lib.fetch_residual_pca_coordinates.argtypes = [ ct.c_void_p ]
lib.fetch_residual_pca_coordinates.restype = ct.c_void_p
lib.fetch_residual_pca_variance_explained.argtypes = [ ct.c_void_p ]
lib.fetch_residual_pca_variance_explained.restype = ct.c_void_p
lib.fetch_residual_pca_total_variance.argtypes = [ ct.c_void_p ]
lib.fetch_residual_pca_total_variance.restype = ct.c_double
lib.fetch_residual_pca_num_dims.argtypes = [ ct.c_void_p ]
lib.fetch_residual_pca_num_dims.restype = ct.c_int
lib.free_residual_pca.argtypes = [ ct.c_void_p ]
lib.run_residual_pca.argtypes = [ 
    ct.c_void_p,
    ct.c_void_p,
    ct.c_uint8,
    ct.c_int,
    ct.c_uint8,
    ct.c_void_p,
    ct.c_uint8,
    ct.c_int
]
lib.run_residual_pca.restype = ct.c_void_p

lib.fetch_multibatch_pca_coordinates.argtypes = [ ct.c_void_p ]
lib.fetch_multibatch_pca_coordinates.restype = ct.c_void_p
lib.fetch_multibatch_pca_variance_explained.argtypes = [ ct.c_void_p ]
lib.fetch_multibatch_pca_variance_explained.restype = ct.c_void_p
lib.fetch_multibatch_pca_total_variance.argtypes = [ ct.c_void_p ]
lib.fetch_multibatch_pca_total_variance.restype = ct.c_double
lib.fetch_multibatch_pca_num_dims.argtypes = [ ct.c_void_p ]
lib.fetch_multibatch_pca_num_dims.restype = ct.c_int
lib.free_multibatch_pca.argtypes = [ ct.c_void_p ]
lib.run_multibatch_pca.argtypes = [ 
    ct.c_void_p,
    ct.c_void_p,
    ct.c_uint8,
    ct.c_uint8,
    ct.c_int,
    ct.c_uint8,
    ct.c_void_p,
    ct.c_uint8,
    ct.c_int
]
lib.run_multibatch_pca.restype = ct.c_void_p
