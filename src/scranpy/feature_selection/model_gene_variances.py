import numpy as np
from mattress import TatamiNumericPointer, tatamize
from ..cpphelpers import lib

def model_gene_variances(x, block = None, span = 0.3, num_threads = 1):
    if not isinstance(x, TatamiNumericPointer):
        x = tatamize(x)

    NR = x.nrow()
    means = np.ndarray((NR,), dtype=np.float64)
    variances = np.ndarray((NR,), dtype=np.float64)
    fitted = np.ndarray((NR,), dtype=np.float64)
    residuals = np.ndarray((NR,), dtype=np.float64)
    extra = None

    if block == None:
        lib.model_gene_variances(
            x.ptr, 
            means.ctypes.data, 
            variances.ctypes.data, 
            fitted.ctypes.data, 
            residuals.ctypes.data, 
            span, 
            num_threads)

    else:
        # TODO: need some way of factorizing a block to a numpy Int32 array.
        # factorize(block)
        block32 = block.astype(np.int32)
        nlevels = block32.max() + 1

        all_means = []
        all_variances = []
        all_fitted = []
        all_residuals = []
        all_means_ptr = np.ndarray((nlevels,), dtype=np.uintp)
        all_variances_ptr = np.ndarray((nlevels,), dtype=np.uintp)
        all_fitted_ptr = np.ndarray((nlevels,), dtype=np.uintp)
        all_residuals_ptr = np.ndarray((nlevels,), dtype=np.uintp)

        for l in range(nlevels):
            cur_means = np.ndarray((nlevels,), dtype=np.uintp)
            cur_variances = np.ndarray((nlevels,), dtype=np.uintp)
            cur_fitted = np.ndarray((nlevels,), dtype=np.uintp)
            cur_residuals = np.ndarray((nlevels,), dtype=np.uintp)

            all_means_ptr[l] = cur_means.ctypes.data
            all_variances_ptr[l] = cur_variances.ctypes.data
            all_fitted_ptr[l] = cur_fitted.ctypes.data
            all_residuals_ptr[l] = cur_residuals.ctypes.data

            all_means.append(cur_means)
            all_variance.append(cur_variance)
            all_fitted.append(cur_fitted)
            all_residuals.append(cur_residuals)

        lib.model_gene_variances_blocked(
            x.ptr, 
            means.ctypes.data, 
            variances.ctypes.data, 
            fitted.ctypes.data, 
            residuals.ctypes.data, 
            nlevels,                                            
            block32.ctypes.data, 
            all_means_ptr.ctypes.data,
            all_variances_ptr.ctypes.data,
            all_fitted_ptr.ctypes.data,
            all_residuals_ptr.ctypes.data,
            span, 
            num_threads)

        extra = {
            "means": all_means,
            "variances": all_variances,
            "fitted": all_fitted,
            "residuals": all_residuals
        }

    return {
        "means": means,
        "variances": variances,
        "fitted": fitted,
        "residuals": residuals,
        "extras": extras
    }
