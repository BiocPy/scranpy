#include <vector>
#include <stdexcept>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"
#include "scran_norm/scran_norm.hpp"

#include "utils.h"

void center_size_factors(const pybind11::array& size_factors, std::optional<pybind11::array> maybe_block, bool lowest) {
    scran_norm::CenterSizeFactorsOptions opt;
    opt.block_mode = (lowest ? scran_norm::CenterBlockMode::LOWEST : scran_norm::CenterBlockMode::PER_BLOCK);
    opt.ignore_invalid = true;

    size_t ncells = size_factors.size();
    double* iptr = const_cast<double*>(check_numpy_array<double>(size_factors));

    if (maybe_block.has_value()) {
        const auto& block = *maybe_block;
        if (block.size() != ncells) {
            throw std::runtime_error("'block' must be the same length as the number of cells");
        }
        auto bptr = check_numpy_array<uint32_t>(block);
        scran_norm::center_size_factors_blocked(ncells, iptr, bptr, NULL, opt);
    } else {
        scran_norm::center_size_factors(ncells, iptr, NULL, opt);
    }
}

void init_center_size_factors(pybind11::module& m) {
    m.def("center_size_factors", &center_size_factors);
}
