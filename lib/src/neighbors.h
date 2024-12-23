#ifndef UTILS_NEIGHBORS_H
#define UTILS_NEIGHBORS_H

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include <vector>
#include "utils.h"

template<typename Index_, class Distance_>
std::vector<std::vector<std::pair<Index_, Distance_> > > unpack_neighbors(const pybind11::array& nnidx, const pybind11::array& nndist) {
    auto ibuffer = nnidx.request();
    size_t nobs = ibuffer.shape[0], nneighbors = ibuffer.shape[1];
    if ((nnidx.flags() & pybind11::array::c_style) == 0) {
        throw std::runtime_error("expected a row-major matrix for the indices");
    }
    if (!nnidx.dtype().is(pybind11::dtype::of<uint32_t>())) {
        throw std::runtime_error("unexpected dtype for array of neighbor indices");
    }
    const uint32_t* iptr = get_numpy_array_data<uint32_t>(nnidx);

    auto dbuffer = nndist.request();
    if ((nndist.flags() & pybind11::array::c_style) == 0) {
        throw std::runtime_error("expected a row-major matrix for the distances");
    }
    if (!nndist.dtype().is(pybind11::dtype::of<double>())) {
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
