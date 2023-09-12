#include "parallel.h" // must be first, to set all macros.

#include "scran/quality_control/SuggestAdtQcFilters.hpp"
#include <cstdint>

static auto create_adt_buffers(
    int num_cells,
    int num_subsets,
    const int32_t* /** numpy */ detected,
    const uintptr_t* /** void_p */ subset_totals)
{
    scran::PerCellAdtQcMetrics::Buffers<double, int32_t> buffer;
    buffer.detected = const_cast<int32_t*>(detected);
    buffer.subset_totals.resize(num_subsets);
    for (int i = 0; i < num_subsets; ++i) {
        buffer.subset_totals[i] = reinterpret_cast<double*>(subset_totals[i]);
    }
    return buffer;
}

//[[export]]
void suggest_adt_qc_filters(
    int32_t num_cells,
    int32_t num_subsets,
    int32_t* /** numpy */ detected,
    uintptr_t* /** void_p */ subset_totals,
    int32_t num_blocks,
    const int32_t* /** void_p */ block,
    double* /** numpy */ detected_out,
    uintptr_t* /** void_p */ subset_totals_out,
    double nmads)
{
    scran::SuggestAdtQcFilters runner;
    runner.set_num_mads(nmads);

    auto buffer = create_adt_buffers(num_cells, num_subsets, detected, subset_totals);

    scran::SuggestAdtQcFilters::Thresholds res;
    if (num_blocks == 1) {
        res = runner.run(num_cells, buffer);
    } else {
        res = runner.run_blocked(num_cells, block, buffer);
    }

    std::copy(res.detected.begin(), res.detected.end(), detected_out);
    for (int i = 0; i < num_subsets; ++i) {
        std::copy(res.subset_totals[i].begin(), res.subset_totals[i].end(), reinterpret_cast<double*>(subset_totals_out[i]));
    }

    return;
}

//[[export]]
void create_adt_qc_filter(
    int num_cells,
    int num_subsets,
    const int32_t* /** numpy */ detected,
    const uintptr_t* /** void_p */ subset_totals,
    int num_blocks,
    const int32_t* /** void_p */ block,
    const double* /** numpy */ detected_thresholds,
    const uintptr_t* /** void_p */ subset_totals_thresholds,
    uint8_t* output /** numpy */)
{
    scran::SuggestAdtQcFilters::Thresholds res;
    res.detected.insert(res.detected.end(), detected_thresholds, detected_thresholds + num_blocks);
    for (int i = 0; i < num_subsets; ++i) {
        auto spptr = reinterpret_cast<const double*>(subset_totals_thresholds[i]);
        res.subset_totals.emplace_back(spptr, spptr + num_blocks);
    }

    auto buffer = create_adt_buffers(num_cells, num_subsets, detected, subset_totals);
    res.filter_blocked(num_cells, (num_blocks > 1 ? block : NULL), buffer, output);
    return;
}
