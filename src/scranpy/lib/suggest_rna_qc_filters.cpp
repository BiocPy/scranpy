#include "scran/quality_control/SuggestRnaQcFilters.hpp"
#include <cstdint>

extern "C" {

void suggest_rna_qc_filters(
    int num_cells,
    int num_subsets, 
    double* sums, 
    int32_t* detected, 
    uintptr_t* subset_proportions, 
    int num_blocks,
    const int32_t* block,
    double* sums_out,
    double* detected_out,
    uintptr_t* subset_proportions_out,
    double nmads)
{
    scran::SuggestRnaQcFilters runner;
    runner.set_num_mads(nmads);

    scran::PerCellRnaQcMetrics::Buffers<double, int32_t> buffer;
    buffer.sums = sums;
    buffer.detected = detected;
    buffer.subset_proportions.resize(num_subsets);
    for (int i = 0; i < num_subsets; ++i) {
        buffer.subset_proportions[i] = reinterpret_cast<double*>(subset_proportions[i]);
    }

    scran::SuggestRnaQcFilters::Thresholds res;
    if (num_blocks == 1) {
        res = runner.run(num_cells, buffer);
    } else {
        res = runner.run_blocked(num_cells, block, buffer);
    }

    std::copy(res.sums.begin(), res.sums.end(), sums_out);
    std::copy(res.detected.begin(), res.detected.end(), detected_out);
    for (int i = 0; i < num_subsets; ++i) {
        std::copy(res.subset_proportions[i].begin(), res.subset_proportions[i].end(), reinterpret_cast<double*>(subset_proportions_out[i]));
    }

    return;
}

}
