import numpy as np
from mattress import TatamiNumericPointer, tatamize
from ..cpphelpers import lib

# TODO: move out for more general use.
def factorize(x):
    levels = []
    mapping = {}
    output = np.ndarray((len(x),), dtype=np.int32)

    for i in range(len(x)):
        lev = x[i]
        if not lev in mapping:
            mapping[lev] = len(levels)
            levels.append(len(levels))
        output[i] = mapping[lev]

    return { "levels": levels, "indices": output }

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
        fac = factorize(block)
        nlevels = len(fac["levels"])

        all_means = []
        all_variances = []
        all_fitted = []
        all_residuals = []
        all_means_ptr = np.ndarray((nlevels,), dtype=np.uintp)
        all_variances_ptr = np.ndarray((nlevels,), dtype=np.uintp)
        all_fitted_ptr = np.ndarray((nlevels,), dtype=np.uintp)
        all_residuals_ptr = np.ndarray((nlevels,), dtype=np.uintp)

        for l in range(nlevels):
            cur_means = np.ndarray((NR,), dtype=np.float64)
            cur_variances = np.ndarray((NR,), dtype=np.float64)
            cur_fitted = np.ndarray((NR,), dtype=np.float64)
            cur_residuals = np.ndarray((NR,), dtype=np.float64)

            all_means_ptr[l] = cur_means.ctypes.data
            all_variances_ptr[l] = cur_variances.ctypes.data
            all_fitted_ptr[l] = cur_fitted.ctypes.data
            all_residuals_ptr[l] = cur_residuals.ctypes.data

            all_means.append(cur_means)
            all_variances.append(cur_variances)
            all_fitted.append(cur_fitted)
            all_residuals.append(cur_residuals)

        lib.model_gene_variances_blocked(
            x.ptr, 
            means.ctypes.data, 
            variances.ctypes.data, 
            fitted.ctypes.data, 
            residuals.ctypes.data, 
            nlevels,
            fac["indices"].ctypes.data, 
            all_means_ptr.ctypes.data,
            all_variances_ptr.ctypes.data,
            all_fitted_ptr.ctypes.data,
            all_residuals_ptr.ctypes.data,
            span, 
            num_threads)

        extra = {}
        for i in range(nlevels):
            extra[fac["levels"][i]] = {
                "means": all_means[i],
                "variances": all_variances[i],
                "fitted": all_fitted[i],
                "residuals": all_residuals[i]
            }

    return {
        "means": means,
        "variances": variances,
        "fitted": fitted,
        "residuals": residuals,
        "per_block": extra
    }
