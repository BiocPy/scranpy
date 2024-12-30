#ifndef UTILS_NEIGHBORS_H
#define UTILS_NEIGHBORS_H

#include <vector>
#include <stdexcept>
#include <cstdint>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

#include "utils.h"

template<typename Index_, class Distance_>
std::vector<std::vector<std::pair<Index_, Distance_> > > unpack_neighbors(const pybind11::array& nnidx, const pybind11::array& nndist) {
    auto ibuffer = nnidx.request();
    size_t nobs = ibuffer.shape[0], nneighbors = ibuffer.shape[1];
    if ((nnidx.flags() & pybind11::array::c_style) == 0) {
        throw std::runtime_error("expected a row-major matrix for the indices");
    }
    const auto& idx_dtype = nnidx.dtype(); // the usual is() doesn't work in a separate process.
    if (idx_dtype.kind() != 'u' || idx_dtype.itemsize() != 4) {
        throw std::runtime_error("unexpected dtype for array of neighbor indices");
    }
    const uint32_t* iptr = get_numpy_array_data<uint32_t>(nnidx);

    auto dbuffer = nndist.request();
    if ((nndist.flags() & pybind11::array::c_style) == 0) {
        throw std::runtime_error("expected a row-major matrix for the distances");
    }
    const auto& dist_dtype = nndist.dtype(); // the usual is() doesn't work in a separate process.
    if (dist_dtype.kind() != 'f' || dist_dtype.itemsize() != 8) {
        throw std::runtime_error("unexpected dtype for array of neighbor distances");
    }
    if (nobs != static_cast<size_t>(dbuffer.shape[0]) || nneighbors != static_cast<size_t>(dbuffer.shape[1])) {
        throw std::runtime_error("neighbor indices and distances should have the same shape");
    }
    const double* dptr = get_numpy_array_data<double>(nndist);

    std::vector<std::vector<std::pair<Index_, Distance_> > > neighbors(nobs);
    for (auto& current : neighbors) {
        current.reserve(nneighbors);
        for (size_t k = 0; k < nneighbors; ++k, ++iptr, ++dptr) {
            current.emplace_back(*iptr, *dptr);
        }
    }

    return neighbors;
}

#endif
