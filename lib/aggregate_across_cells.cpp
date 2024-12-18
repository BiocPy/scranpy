#include "parallel.h" // must be first, to set all macros.

#include "Mattress.h"
#include "scran/aggregation/AggregateAcrossCells.hpp"
#include <cstdint>
#include <algorithm>

//[[export]]
void* combine_factors(int32_t length, int32_t number, const uintptr_t* inputs /** void_p */, int32_t* output_combined /** numpy */) {
    std::vector<const int32_t*> factors;
    factors.reserve(number);
    for (int32_t i = 0; i < number; ++i) {
        factors.push_back(reinterpret_cast<const int32_t*>(inputs[i]));
    }

    auto res = scran::AggregateAcrossCells::combine_factors(length, std::move(factors), output_combined);
    return new decltype(res)(std::move(res));
}

//[[export]]
int32_t get_combined_factors_size(void* ptr) {
    auto combined = reinterpret_cast<scran::AggregateAcrossCells::Combinations<int32_t>*>(ptr);
    return combined->counts.size();
}

//[[export]]
void get_combined_factors_level(void* ptr, int32_t i, int32_t* output /** numpy */) {
    auto combined = reinterpret_cast<const scran::AggregateAcrossCells::Combinations<int32_t>*>(ptr);
    const auto& lev = combined->factors[i];
    std::copy(lev.begin(), lev.end(), output);
}

//[[export]]
void get_combined_factors_count(void* ptr, int32_t* output /** numpy */) {
    auto combined = reinterpret_cast<const scran::AggregateAcrossCells::Combinations<int32_t>*>(ptr);
    std::copy(combined->counts.begin(), combined->counts.end(), output);
}

//[[export]]
void free_combined_factors(void* ptr) {
    delete reinterpret_cast<scran::AggregateAcrossCells::Combinations<int32_t>*>(ptr);
}

//[[export]]
void aggregate_across_cells(
    void* mat,
    const int32_t* groups /** numpy */,
    int32_t ngroups,
    uint8_t do_sums,
    double* output_sums /** void_p */,
    uint8_t do_detected,
    int32_t* output_detected /** void_p */,
    int32_t nthreads)
{
    const auto& ptr = reinterpret_cast<const Mattress*>(mat)->ptr;
    auto NR = ptr->nrow();
    auto NC = ptr->ncol();

    std::vector<double*> output_sums_ptrs;
    if (do_sums) {
        output_sums_ptrs.reserve(ngroups);
        for (int32_t g = 0; g < ngroups; ++g) {
            output_sums_ptrs.push_back(output_sums);
            output_sums += NR;
        }
    }

    std::vector<int32_t*> output_detected_ptrs;
    if (do_detected) {
        output_detected_ptrs.reserve(ngroups);
        for (int32_t g = 0; g < ngroups; ++g) {
            output_detected_ptrs.push_back(output_detected);
            output_detected += NR;
        }
    }

    scran::AggregateAcrossCells aggr;
    aggr.set_num_threads(nthreads);
    aggr.run(ptr.get(), groups, std::move(output_sums_ptrs), std::move(output_detected_ptrs));
    return;
}
