#include "parallel.h" // must be first, to set all macros.

#include "Mattress.h"
#include "scran/quality_control/PerCellCrisprQcMetrics.hpp"
#include <cstdint>

//[[export]]
void per_cell_crispr_qc_metrics(
    const void* mat,
    double* /** numpy */ sum_output,
    int32_t* /** numpy */ detected_output,
    double* /** numpy */ max_prop_output,
    int32_t* /** numpy */ max_index_output,
    int32_t num_threads)
{
    scran::PerCellCrisprQcMetrics runner;
    runner.set_num_threads(num_threads);

    scran::PerCellCrisprQcMetrics::Buffers<double, int32_t> buffer;
    buffer.sums = sum_output;
    buffer.detected = detected_output;
    buffer.max_proportion = max_prop_output;
    buffer.max_index = max_index_output;

    runner.run(reinterpret_cast<const Mattress*>(mat)->ptr.get(), buffer);
    return;
}
