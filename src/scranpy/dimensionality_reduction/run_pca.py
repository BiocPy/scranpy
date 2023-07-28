import numpy as np
from mattress import TatamiNumericPointer, tatamize
import ctypes as ct

from ..cpphelpers import lib
from ..types import MatrixTypes

def run_pca(x: MatrixTypes, rank, subset = None, scale = False, num_threads = 1):
    if not isinstance(x, TatamiNumericPointer):
        x = tatamize(x)

    nr = x.nrow()
    nc = x.ncol()

    use_subset = subset != None
    temp_subset = None
    subset_offset = 0
    if use_subset:
        temp_subset = np.zeros((nr,), dtype = np.uint8)
        temp_subset[subset] = 1
        subset_offset = temp_subset.ctypes.data

    result = {}
    pptr = lib.run_simple_pca(x.ptr, rank, use_subset, subset_offset, scale, num_threads)
    try:
        actual_rank = lib.fetch_simple_pca_num_dims(pptr)

        pc_pointer = ct.cast(lib.fetch_simple_pca_coordinates(pptr), ct.POINTER(ct.c_double))
        pc_array = np.ctypeslib.as_array(pc_pointer, shape=(actual_rank, nc))
        result["principal_components"] = np.copy(pc_array)

        var_pointer = ct.cast(lib.fetch_simple_pca_variance_explained(pptr), ct.POINTER(ct.c_double))
        var_array = np.ctypeslib.as_array(var_pointer, shape=(actual_rank,))
        total = lib.fetch_simple_pca_total_variance(pptr)
        result["variance_explained"] = np.copy(var_array) / total

    finally:
        lib.free_simple_pca(pptr)

    return result 
