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

lib.fetch_simple_pcs.argtypes = [ ct.c_void_p ]
lib.fetch_simple_pcs.restypes = ct.c_void_p
lib.fetch_simple_variance_explained.argtypes = [ ct.c_void_p ]
lib.fetch_simple_variance_explained.restypes = ct.c_void_p
lib.fetch_simple_total_variance.argtypes = [ ct.c_void_p ]
lib.fetch_simple_total_variance.restypes = ct.c_double
lib.fetch_simple_num_dims.argtypes = [ ct.c_void_p ]
lib.fetch_simple_num_dims.restypes = ct.c_int
lib.free_simple_pcs.argtypes = [ ct.c_void_p ]
lib.simple_pca.argtypes = [ 
    ct.c_void_p,
    ct.c_int,
    ct.c_uint8,
    ct.c_void_p,
    ct.c_uint8,
    ct.c_int
]
lib.fetch_simple_num_dims.restypes = ct.c_void_p

