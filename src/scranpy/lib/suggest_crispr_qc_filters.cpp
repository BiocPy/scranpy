#include "parallel.h" // must be first, to set all macros.

#include "scran/quality_control/SuggestCrisprQcFilters.hpp"
#include <cstdint>

static auto create_crispr_buffers(
    int num_cells,
    const double* sums,
    const double* max_proportion)
{
    scran::PerCellCrisprQcMetrics::Buffers<double, int32_t> buffer;
    buffer.sums = const_cast<double*>(sums);
    buffer.max_proportion = const_cast<double*>(max_proportion);
    return buffer;
}

//[[export]]
void suggest_crispr_qc_filters(
    int32_t num_cells,
    double* /** numpy */ sums,
    double* /** numpy */ max_proportion,
    int32_t num_blocks,
    const int32_t* /** void_p */ block,
    double* /** numpy */ max_count_out,
    double nmads)
{
    scran::SuggestCrisprQcFilters runner;
    runner.set_num_mads(nmads);

    auto buffer = create_crispr_buffers(num_cells, sums, max_proportion);

    scran::SuggestCrisprQcFilters::Thresholds res;
    if (num_blocks == 1) {
        res = runner.run(num_cells, buffer);
    } else {
        res = runner.run_blocked(num_cells, block, buffer);
    }

    std::copy(res.max_count.begin(), res.max_count.end(), max_count_out);

    return;
}

//[[export]]
void create_crispr_qc_filter(
    int num_cells,
    const double* /** numpy */ sums,
    const double* /** numpy */ max_proportion,
    int num_blocks,
    const int32_t* /** void_p */ block,
    const double* /** numpy */ max_count_thresholds,
    uint8_t* output /** numpy */)
{
    scran::SuggestCrisprQcFilters::Thresholds res;
    res.max_count.insert(res.max_count.end(), max_count_thresholds, max_count_thresholds + num_blocks);
    auto buffer = create_crispr_buffers(num_cells, sums, max_proportion);
    res.filter_blocked(num_cells, (num_blocks > 1 ? block : NULL), buffer, output);
    return;
}
