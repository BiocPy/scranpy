import numpy as np
from mattress import tatamize, TatamiNumericPointer
from ..cpphelpers import lib
from ..utils import factorize
from biocframe import BiocFrame

def create_output_arrays(nr, ngroups):
    outptrs = np.ndarray((ngroups,), dtype=np.uintp)
    outarrs = []
    for g in range(ngroups):
        curarr = np.ndarray((nr,), dtype=np.float64)
        outptrs[g] = curarr.ctypes.data
        outarrs.append(curarr)
    return outarrs, outptrs

def create_output_summary_arrays(nr, ngroups):
    output = {
        "min": create_output_arrays(nr, ngroups),
        "mean": create_output_arrays(nr, ngroups),
        "min_rank": create_output_arrays(nr, ngroups) 
    }

    outptrs = np.ndarray((3,), dtype=np.uintp)
    outptrs[0] = output["min"][1].ctypes.data
    outptrs[1] = output["mean"][1].ctypes.data
    outptrs[2] = output["min_rank"][1].ctypes.data
    return output, outptrs

def create_summary_biocframe(summary, group):
    return BiocFrame({ 
        "min": summary[0]["min"][0][group],
        "mean": summary[0]["mean"][0][group],
        "min_rank": summary[0]["min_rank"][0][group],
    })

def score_markers(x, grouping, block = None, threshold = 0, compute_auc = True, num_threads = 1):
    if not isinstance(x, TatamiNumericPointer):
        x = tatamize(x)
    nr = x.nrow()
    nc = x.ncol()

    if len(grouping) != nc:
        raise ValueError("length of 'grouping' should be equal to the number of columns in 'x'")
    grouping = factorize(grouping)
    num_groups = len(grouping.levels)

    block_offset = 0
    num_blocks = 1
    if block is not None:
        if len(block) != nc:
            raise ValueError("length of 'block' should be equal to the number of columns in 'x'")
        block = factorize(block)
        num_blocks = len(block.levels)
        block_offset = block.indices.ctypes.data 

    means = create_output_arrays(nr, num_groups)
    detected = create_output_arrays(nr, num_groups)
    cohen = create_output_summary_arrays(nr, num_groups)
    lfc = create_output_summary_arrays(nr, num_groups)
    delta_detected = create_output_summary_arrays(nr, num_groups)

    auc = None
    auc_offset = 0
    if compute_auc:
        auc = create_output_summary_arrays(nr, num_groups)
        auc_offset = auc[1].ctypes.data

    lib.score_markers(
        x.ptr,
        num_groups,
        grouping.indices.ctypes.data,
        num_blocks,
        block_offset,
        compute_auc,
        threshold,
        means[1].ctypes.data,
        detected[1].ctypes.data,
        cohen[1].ctypes.data,
        auc_offset,
        lfc[1].ctypes.data,
        delta_detected[1].ctypes.data,
        num_threads
    )

    output = {}
    for g in range(num_groups):
        current = {
            "means": means[0][g],
            "detected": detected[0][g],
            "cohen": create_summary_biocframe(cohen, g),
            "lfc": create_summary_biocframe(lfc, g),
            "delta_detected": create_summary_biocframe(delta_detected, g)
        }
        if compute_auc:
            current["auc"] = create_summary_biocframe(auc, g)
        output[grouping.levels[g]] = BiocFrame(current)

    return output
