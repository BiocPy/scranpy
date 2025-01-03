#include <vector>
#include <cstdint>
#include <stdexcept>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"
#include "mnncorrect/mnncorrect.hpp"
#include "knncolle_py.h"

#include "utils.h"

pybind11::tuple correct_mnn(
    const pybind11::array& x, 
    const pybind11::array& block, 
    int num_neighbors, 
    double num_mads, 
    int robust_iterations,
    double robust_trim,
    int num_threads,
    int mass_cap,
    std::optional<pybind11::array> maybe_order, 
    std::string ref_policy, 
    uintptr_t builder_ptr)
{
    mnncorrect::Options<uint32_t, uint32_t, double> opts;
    opts.num_neighbors = num_neighbors;
    opts.num_mads = num_mads;
    opts.robust_iterations = robust_iterations;
    opts.robust_trim = robust_trim;
    opts.mass_cap = mass_cap;
    opts.num_threads = num_threads;

    if (maybe_order.has_value()) {
        const pybind11::array& order = *maybe_order;
        auto optr = check_numpy_array<uint32_t>(order);
        opts.order.insert(opts.order.end(), optr, optr + order.size());
    }

    if (ref_policy == "input") {
        opts.reference_policy = mnncorrect::ReferencePolicy::INPUT;
    } else if (ref_policy == "max-variance") {
        opts.reference_policy = mnncorrect::ReferencePolicy::MAX_VARIANCE;
    } else if (ref_policy == "max-rss") {
        opts.reference_policy = mnncorrect::ReferencePolicy::MAX_RSS;
    } else if (ref_policy == "max-size") {
        opts.reference_policy = mnncorrect::ReferencePolicy::MAX_SIZE;
    } else {
        throw std::runtime_error("unknown reference policy");
    }

    const auto& builder = knncolle_py::cast_builder(builder_ptr)->ptr;
    typedef std::shared_ptr<knncolle::Builder<knncolle_py::SimpleMatrix, knncolle_py::Distance> > BuilderPointer;
    opts.builder = BuilderPointer(BuilderPointer{}, builder.get()); // make a no-op shared pointer.

    auto xbuffer = x.request();
    if (xbuffer.shape.size() != 2) {
        throw std::runtime_error("expected a 2-dimensional array for 'x'");
    }
    if ((x.flags() & pybind11::array::f_style) == 0) {
        throw std::runtime_error("expected Fortran-style storage for 'x'");
    }
    if (!x.dtype().is(pybind11::dtype::of<double>())) {
        throw std::runtime_error("unexpected dtype for 'x'");
    }

    size_t ndim = xbuffer.shape[0];
    size_t nobs = xbuffer.shape[1];
    if (nobs != block.size()) {
        throw std::runtime_error("length of 'block' should equal the number of columns in 'x'");
    }

    pybind11::array_t<double, pybind11::array::f_style> corrected({ ndim, nobs });
    auto res = mnncorrect::compute(
        ndim,
        nobs,
        static_cast<const double*>(xbuffer.ptr),
        check_numpy_array<uint32_t>(block),
        static_cast<double*>(corrected.request().ptr),
        opts
    );

    pybind11::tuple output(3);
    output[0] = corrected;
    output[1] = pybind11::array_t<size_t>(res.merge_order.size(), res.merge_order.data());
    output[2] = pybind11::array_t<size_t>(res.num_pairs.size(), res.num_pairs.data());
    return output;
}

void init_correct_mnn(pybind11::module& m) {
    m.def("correct_mnn", &correct_mnn);
}
