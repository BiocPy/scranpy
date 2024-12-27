#include <vector>
#include <algorithm>
#include <stdexcept>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "scran_aggregate/scran_aggregate.hpp"

#include "utils.h"

static pybind11::tuple convert_to_index_list(const std::vector<std::vector<uint32_t> >& levels) {
    size_t num_fac = levels.size();
    pybind11::tuple combos(num_fac);
    for (size_t f = 0; f < num_fac; ++f) {
        const auto& current = levels[f];
        combos[f] = pybind11::array_t<uint32_t>(current.size(), current.data());
    }
    return combos;
}

pybind11::tuple combine_factors(const pybind11::tuple& factors, bool keep_unused, const pybind11::array& num_levels) {
    size_t num_fac = factors.size();
    if (num_fac == 0) {
        throw std::runtime_error("'factors' must have length greater than zero");
    }

    std::vector<pybind11::array> ibuffers;
    ibuffers.reserve(num_fac);
    for (size_t f = 0; f < num_fac; ++f) {
        ibuffers.emplace_back(factors[f].cast<pybind11::array>());
    }

    size_t ngenes = ibuffers.front().size();
    for (size_t f = 1; f < num_fac; ++f) {
        if (static_cast<size_t>(ibuffers[f].size()) != ngenes) {
            throw std::runtime_error("all elements of 'factors' must have the same length");
        }
    }

    pybind11::tuple output(2);

    if (keep_unused) {
        if (static_cast<size_t>(num_levels.size()) != num_fac) {
            throw std::runtime_error("'num_levels' and 'factors' must have the same length");
        }
        auto lptr = check_numpy_array<uint32_t>(num_levels);
        std::vector<std::pair<const uint32_t*, uint32_t> > buffers;
        buffers.reserve(num_fac);
        for (size_t f = 0; f < num_fac; ++f) {
            buffers.emplace_back(check_numpy_array<uint32_t>(ibuffers[f]), lptr[f]);
        }
        pybind11::array_t<uint32_t> oindices(ngenes);
        auto res = scran_aggregate::combine_factors_unused(ngenes, buffers, static_cast<uint32_t*>(oindices.request().ptr));
        output[0] = oindices;
        output[1] = convert_to_index_list(res);

    } else {
        std::vector<const uint32_t*> buffers;
        buffers.reserve(num_fac);
        for (size_t f = 0; f < num_fac; ++f) {
            buffers.emplace_back(check_numpy_array<uint32_t>(ibuffers[f]));
        }
        pybind11::array_t<uint32_t> oindices(ngenes);
        auto res = scran_aggregate::combine_factors(ngenes, buffers, static_cast<uint32_t*>(oindices.request().ptr));
        output[0] = oindices;
        output[1] = convert_to_index_list(res);
    }

    return output;
}

void init_combine_factors(pybind11::module& m) {
    m.def("combine_factors", &combine_factors);
}
