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
