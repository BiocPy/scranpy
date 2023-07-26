import sys

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

import os
import ctypes as ct
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
    ct.c_int
]

import mattress as mt
import numpy as np
def per_cell_rna_qc_metrics(x, subsets = None, num_threads = 1):
    if not isinstance(x, mt.TatamiNumericPointer.TatamiNumericPointer): # ??? why twice?
        x = mt.tatamize(x)

    nc = x.ncol()
    sums = np.ndarray((nc,), dtype = np.float64)
    detected = np.ndarray((nc,), dtype = np.int32)

    num_subsets = 0

    lib.per_cell_rna_qc_metrics(x.ptr, num_subsets, 0, sums.ctypes.data, detected.ctypes.data, 0, num_threads)

    return {
        "sums": sums,
        "detected": detected,
        "subset_proportions": []
    }
