#include "scran/normalization/LogNormCounts.hpp"
#include "Mattress.h"

#include <cstdint>
#include <memory>

extern "C" {

Mattress* log_norm_counts(const Mattress* mat, uint8_t use_block, const int32_t* block, uint8_t use_size_factors, const double* size_factors, uint8_t center, uint8_t allow_zeros, uint8_t allow_non_finite, int num_threads) {
    scran::LogNormCounts runner;
    runner.set_center(center);
    runner.set_handle_zeros(allow_zeros);
    runner.set_handle_non_finite(allow_non_finite);
    runner.set_num_threads(num_threads);

    if (!use_block) {
        block = NULL;
    }

    std::shared_ptr<tatami::Matrix<double, int> > out;
    if (use_size_factors) {
        out = runner.run_blocked(mat->ptr, std::vector<double>(size_factors, size_factors + mat->ptr->ncol()), block); // TODO: replace with a view.
    } else {
        out = runner.run_blocked(mat->ptr, block);
    }

    return new Mattress(std::move(out));
}

}
