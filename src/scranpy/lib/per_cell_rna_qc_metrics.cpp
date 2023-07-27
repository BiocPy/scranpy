#include "Mattress.h"
#include "scran/quality_control/PerCellRnaQcMetrics.hpp"
#include <cstdint>

extern "C" {

void per_cell_rna_qc_metrics(const Mattress* mat, int num_subsets, const uintptr_t* subset_ptrs, double* sum_output, int32_t* detected_output, uintptr_t* subset_output, int num_threads) {
    scran::PerCellRnaQcMetrics runner;
    runner.set_num_threads(num_threads);

    std::vector<const uint8_t*> subsets(num_subsets);
    for (int i = 0; i < num_subsets; ++i) {
        subsets[i] = reinterpret_cast<const uint8_t*>(subset_ptrs[i]);
    }

    scran::PerCellRnaQcMetrics::Buffers<double, int32_t> buffer;
    buffer.sums = sum_output;
    buffer.detected = detected_output;
    buffer.subset_proportions.resize(num_subsets);
    for (int i = 0; i < num_subsets; ++i) {
        buffer.subset_proportions[i] = reinterpret_cast<double*>(subset_output[i]);
    }

    runner.run(mat->ptr.get(), subsets, buffer);
    return;
}

}
