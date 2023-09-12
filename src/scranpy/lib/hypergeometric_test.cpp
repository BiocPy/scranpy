#include "parallel.h"

#include "scran/feature_set_enrichment/HypergeometricTail.hpp"
#include <cstdint>

//[[export]]
void hypergeometric_test(
    int32_t num_genes,
    int32_t de_in_set_size,
    const int32_t* de_in_set /** void_p */,
    int32_t set_size_size,
    const int32_t* set_size /** void_p */,
    int32_t num_de_size,
    const int32_t* num_de /** void_p */,
    int32_t total_genes_size,
    const int32_t* total_genes /** void_p */,
    uint8_t log,
    uint8_t upper_tail,
    int32_t num_threads,
    double* output /** numpy */)
{
    tatami::parallelize([&](int, int32_t start, int32_t len) -> void {
        scran::HypergeometricTail runner;
        runner.set_log(log);
        runner.set_upper_tail(upper_tail);
        for (int32_t i = start, end = start + len; i < end; ++i) {
            auto total = total_genes[(total_genes_size != 1) * i];
            auto inside = set_size[(set_size_size != 1) * i];
            output[i] = runner.run(
                de_in_set[(de_in_set_size != 1) * i],
                inside,
                total - inside,
                num_de[(num_de_size != 1) * i]
            );
        }
    }, num_genes, num_threads);
}
