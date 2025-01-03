#include <vector>
#include <stdexcept>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "nenesub/nenesub.hpp"
#include "tatami/tatami.hpp"

#include "utils.h"

pybind11::array subsample_by_neighbors(const pybind11::array& indices, const pybind11::array& distances, int min_remaining) {
    auto ibuffer = indices.request();
    size_t nobs = ibuffer.shape[0], nneighbors = ibuffer.shape[1];
    if ((indices.flags() & pybind11::array::c_style) == 0) {
        throw std::runtime_error("expected a row-major matrix for the indices");
    }
    if (!indices.dtype().is(pybind11::dtype::of<uint32_t>())) {
        throw std::runtime_error("unexpected dtype for array of neighbor indices");
    }
    const uint32_t* iptr = get_numpy_array_data<uint32_t>(indices);

    auto dbuffer = distances.request();
    if ((distances.flags() & pybind11::array::c_style) == 0) {
        throw std::runtime_error("expected a row-major matrix for the distances");
    }
    if (!distances.dtype().is(pybind11::dtype::of<double>())) {
        throw std::runtime_error("unexpected dtype for array of neighbor distances");
    }
    if (nobs != static_cast<size_t>(dbuffer.shape[0]) || nneighbors != static_cast<size_t>(dbuffer.shape[1])) {
        throw std::runtime_error("neighbor indices and distances should have the same shape");
    }
    const double* dptr = get_numpy_array_data<double>(distances);

    if (nneighbors < static_cast<size_t>(min_remaining)) {
        throw std::runtime_error("'min_remaining' should not be greater than the number of neighbors");
    }

    nenesub::Options opt;
    opt.min_remaining = min_remaining;
    std::vector<uint32_t> selected;
    nenesub::compute(
        static_cast<uint32_t>(nobs),
        /* get_neighbors = */ [&](size_t i) -> tatami::ArrayView<uint32_t> {
            return tatami::ArrayView<uint32_t>(iptr + nneighbors * i, nneighbors);
        },
        /* get_index = */ [](const tatami::ArrayView<uint32_t>& neighbors, size_t i) -> uint32_t{
            return neighbors[i];
        },
        /* get_max_distance = */ [&](size_t i) -> double {
            return dptr[nneighbors * i]; // no need to cast, everything is already a size_t.
        },
        opt, 
        selected
    );

    return pybind11::array_t<uint32_t>(selected.size(), selected.data());
}

void init_subsample_by_neighbors(pybind11::module& m) {
    m.def("subsample_by_neighbors", &subsample_by_neighbors);
}
