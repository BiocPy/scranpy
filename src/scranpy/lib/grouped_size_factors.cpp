#include "parallel.h" // before all other headers.

#include "Mattress.h"
#include "scran/normalization/quick_grouped_size_factors.hpp"

#include <cstdint>

//[[export]]
void grouped_size_factors_with_clusters(void* mat, const int32_t* clusters /** numpy */, double* output /** numpy */, int32_t num_threads) {
    scran::GroupedSizeFactors group_runner;
    group_runner.set_num_threads(num_threads);
    group_runner.set_handle_zeros(true);
    group_runner.set_handle_non_finite(true);

    const auto& ptr = reinterpret_cast<const Mattress*>(mat)->ptr;
    group_runner.run(ptr.get(), clusters, output);
}

//[[export]]
void grouped_size_factors_without_clusters(
    void* mat, 
    uint8_t use_block, 
    const int32_t* block /** void_p */, 
    uint8_t use_init_sf,
    const double* initial_size_factors /** void_p */,
    int32_t rank, 
    double* output /** numpy */, 
    int32_t num_threads) 
{
    scran::quick_grouped_size_factors::Options<int32_t> options;
    options.block = block;
    options.rank = rank;
    options.initial_factors = initial_size_factors;
    options.num_threads = num_threads;

    // TODO: turn on the safety handlers.

    const auto& ptr = reinterpret_cast<const Mattress*>(mat)->ptr;
    scran::quick_grouped_size_factors::run(ptr.get(), output, options);
}
