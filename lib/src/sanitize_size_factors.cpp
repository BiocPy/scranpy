#include <vector>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "scran_norm/scran_norm.hpp"

#include "utils.h"

void sanitize_size_factors(const pybind11::array& size_factors, bool handle_zero, bool handle_negative, bool handle_nan, bool handle_infinite) {
    scran_norm::SanitizeSizeFactorsOptions opt;
    opt.handle_zero = (handle_zero ? scran_norm::SanitizeAction::SANITIZE : scran_norm::SanitizeAction::IGNORE);
    opt.handle_negative = (handle_negative ? scran_norm::SanitizeAction::SANITIZE : scran_norm::SanitizeAction::IGNORE);
    opt.handle_infinite = (handle_infinite ? scran_norm::SanitizeAction::SANITIZE : scran_norm::SanitizeAction::IGNORE);
    opt.handle_nan = (handle_nan ? scran_norm::SanitizeAction::SANITIZE : scran_norm::SanitizeAction::IGNORE);

    size_t ncells = size_factors.size();
    double* iptr = const_cast<double*>(check_numpy_array<double>(size_factors));
    scran_norm::sanitize_size_factors(ncells, iptr, opt);
    return; 
}

void init_sanitize_size_factors(pybind11::module& m) {
    m.def("sanitize_size_factors", &sanitize_size_factors);
}
