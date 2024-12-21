#include <vector>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "scran_norm/scran_norm.hpp"

#include "utils.h"

double choose_pseudo_count(const pybind11::array& size_factors, double quantile, double max_bias, double min_value) {
    scran_norm::ChoosePseudoCountOptions opt;
    opt.quantile = quantile;
    opt.max_bias = max_bias;
    opt.min_value = min_value;
    return scran_norm::choose_pseudo_count(size_factors.size(), check_numpy_array<double>(size_factors), opt);
}

void init_choose_pseudo_count(pybind11::module& m) {
    m.def("choose_pseudo_count", &choose_pseudo_count);
}
