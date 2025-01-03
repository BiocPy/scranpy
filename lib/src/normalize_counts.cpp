#include <vector>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "scran_norm/scran_norm.hpp"

#include "mattress.h"
#include "utils.h"

uintptr_t normalize_counts(uintptr_t x, pybind11::array size_factors, bool log, double pseudo_count, double log_base, bool preserve_sparsity) {
    scran_norm::NormalizeCountsOptions opt;
    opt.log = log;
    opt.pseudo_count = pseudo_count;
    opt.log_base = log_base;
    opt.preserve_sparsity = preserve_sparsity;

    auto ptr = mattress::cast(x);
    auto sfptr = check_numpy_array<double>(size_factors);
    auto tmp = std::make_unique<mattress::BoundMatrix>();
    tmp->ptr = scran_norm::normalize_counts(ptr->ptr, std::vector<double>(sfptr, sfptr + size_factors.size()), opt);
    tmp->original = ptr->original;
    return mattress::cast(tmp.release());
}

void init_normalize_counts(pybind11::module& m) {
    m.def("normalize_counts", &normalize_counts);
}
