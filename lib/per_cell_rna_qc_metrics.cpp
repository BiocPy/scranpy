#include "parallel.h" // must be first, to set all macros.

#include "Mattress.h"
#include "scran/quality_control/PerCellRnaQcMetrics.hpp"
#include <cstdint>

//[[export]]
void per_cell_rna_qc_metrics(
    const void* mat,
    int32_t num_subsets,
    const uintptr_t* /** void_p */ subset_ptrs,
    double* /** numpy */ sum_output,
    int32_t* /** numpy */ detected_output,
    uintptr_t* /** void_p */ subset_output,
    int32_t num_threads)
{
    scran::PerCellRnaQcMetrics runner;
    runner.set_num_threads(num_threads);

    std::vector<const uint8_t*> subsets(num_subsets);
    for (int32_t i = 0; i < num_subsets; ++i) {
        subsets[i] = reinterpret_cast<const uint8_t*>(subset_ptrs[i]);
    }

    scran::PerCellRnaQcMetrics::Buffers<double, int32_t> buffer;
    buffer.sums = sum_output;
    buffer.detected = detected_output;
    buffer.subset_proportions.resize(num_subsets);
    for (int32_t i = 0; i < num_subsets; ++i) {
        buffer.subset_proportions[i] = reinterpret_cast<double*>(subset_output[i]);
    }

    runner.run(reinterpret_cast<const Mattress*>(mat)->ptr.get(), subsets, buffer);
    return;
}
