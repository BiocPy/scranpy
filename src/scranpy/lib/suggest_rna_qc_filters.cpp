#include "parallel.h" // must be first, to set all macros.

#include "scran/quality_control/SuggestRnaQcFilters.hpp"
#include <cstdint>

static auto create_rna_buffers(
    int num_cells,
    int num_subsets,
    const double* /** numpy */ sums,
    const int32_t* /** numpy */ detected,
    const uintptr_t* /** void_p */ subset_proportions)
{
    scran::PerCellRnaQcMetrics::Buffers<double, int32_t> buffer;
    buffer.sums = const_cast<double*>(sums);
    buffer.detected = const_cast<int32_t*>(detected);
    buffer.subset_proportions.resize(num_subsets);
    for (int i = 0; i < num_subsets; ++i) {
        buffer.subset_proportions[i] = reinterpret_cast<double*>(subset_proportions[i]);
    }
    return buffer;
}

//[[export]]
void suggest_rna_qc_filters(
    int32_t num_cells,
    int32_t num_subsets,
    double* /** numpy */ sums,
    int32_t* /** numpy */ detected,
    uintptr_t* /** void_p */ subset_proportions,
    int32_t num_blocks,
    const int32_t* /** void_p */ block,
    double* /** numpy */ sums_out,
    double* /** numpy */ detected_out,
    uintptr_t* /** void_p */ subset_proportions_out,
    double nmads)
{
    scran::SuggestRnaQcFilters runner;
    runner.set_num_mads(nmads);

    auto buffer = create_rna_buffers(num_cells, num_subsets, sums, detected, subset_proportions);

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

//[[export]]
void create_rna_qc_filter(
    int num_cells,
    int num_subsets,
    const double* /** numpy */ sums,
    const int32_t* /** numpy */ detected,
    const uintptr_t* /** void_p */ subset_proportions,
    int num_blocks,
    const int32_t* /** void_p */ block,
    const double* /** numpy */ sums_thresholds,
    const double* /** numpy */ detected_thresholds,
    const uintptr_t* /** void_p */ subset_proportions_thresholds,
    uint8_t* output /** numpy */)
{
    scran::SuggestRnaQcFilters::Thresholds res;
    res.sums.insert(res.sums.end(), sums_thresholds, sums_thresholds + num_blocks);
    res.detected.insert(res.detected.end(), detected_thresholds, detected_thresholds + num_blocks);
    for (int i = 0; i < num_subsets; ++i) {
        auto spptr = reinterpret_cast<const double*>(subset_proportions_thresholds[i]);
        res.subset_proportions.emplace_back(spptr, spptr + num_blocks);
    }

    auto buffer = create_rna_buffers(num_cells, num_subsets, sums, detected, subset_proportions);
    res.filter_blocked(num_cells, (num_blocks > 1 ? block : NULL), buffer, output);
    return;
}
