import numpy as np
from mattress import TatamiNumericPointer, tatamize
import ctypes as ct

from ..cpphelpers import lib
from ..types import MatrixTypes
from ..utils import factorize, to_logical

def run_pca(x: MatrixTypes, rank, subset = None, block = None, scale = False, block_method = "project", num_threads = 1):
    if not isinstance(x, TatamiNumericPointer):
        x = tatamize(x)

    nr = x.nrow()
    nc = x.ncol()

    use_subset = subset is not None
    temp_subset = None
    subset_offset = 0
    if use_subset:
        temp_subset = to_logical(subset, nr)
        subset_offset = temp_subset.ctypes.data

    result = {}
    if block is None or block_method == "none":
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

    else:
        block_info = factorize(block)
        block_offset = block_info.indices.ctypes.data

        if block_method == "regress":
            pptr = lib.run_residual_pca(x.ptr, block_offset, rank, use_subset, subset_offset, scale, num_threads)
            try:
                actual_rank = lib.fetch_residual_pca_num_dims(pptr)

                pc_pointer = ct.cast(lib.fetch_residual_pca_coordinates(pptr), ct.POINTER(ct.c_double))
                pc_array = np.ctypeslib.as_array(pc_pointer, shape=(actual_rank, nc))
                result["principal_components"] = np.copy(pc_array)

                var_pointer = ct.cast(lib.fetch_residual_pca_variance_explained(pptr), ct.POINTER(ct.c_double))
                var_array = np.ctypeslib.as_array(var_pointer, shape=(actual_rank,))
                total = lib.fetch_residual_pca_total_variance(pptr)
                result["variance_explained"] = np.copy(var_array) / total

            finally:
                lib.free_residual_pca(pptr)

        elif block_method == "multibatch":
            pptr = lib.run_multibatch_pca(x.ptr, block_offset, rank, use_subset, subset_offset, scale, num_threads)
            try:
                actual_rank = lib.fetch_multibatch_pca_num_dims(pptr)

                pc_pointer = ct.cast(lib.fetch_multibatch_pca_coordinates(pptr), ct.POINTER(ct.c_double))
                pc_array = np.ctypeslib.as_array(pc_pointer, shape=(actual_rank, nc))
                result["principal_components"] = np.copy(pc_array)

                var_pointer = ct.cast(lib.fetch_multibatch_pca_variance_explained(pptr), ct.POINTER(ct.c_double))
                var_array = np.ctypeslib.as_array(var_pointer, shape=(actual_rank,))
                total = lib.fetch_multibatch_pca_total_variance(pptr)
                result["variance_explained"] = np.copy(var_array) / total

            finally:
                lib.free_multibatch_pca(pptr)

        else:
            raise ValueError("'block_method' must be one of \"none\", \"residual\" or \"project\"")

    return result 
