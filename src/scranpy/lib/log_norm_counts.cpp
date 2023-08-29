#include "parallel.h" // must be first, to set all macros.

#include "scran/normalization/LogNormCounts.hpp"
#include "Mattress.h"

#include <memory>

//[[export]]
void* log_norm_counts(const void* mat0, const double* size_factors /** numpy */) {
    auto mat = reinterpret_cast<const Mattress*>(mat0);
    scran::LogNormCounts runner;
    runner.set_center(false);
    return new Mattress(runner.run(mat->ptr, std::vector<double>(size_factors, size_factors + mat->ptr->ncol())));
}
