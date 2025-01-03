#include <vector>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "clrm1/clrm1.hpp"
#include "mattress.h"

pybind11::array compute_clrm1_factors(uintptr_t x, int num_threads) {
    const auto& mat = mattress::cast(x)->ptr;

    clrm1::Options opt;
    opt.num_threads = num_threads;
    pybind11::array_t<double> output(mat->ncol());
    clrm1::compute(*mat, opt, static_cast<double*>(output.request().ptr));
    return output;
}

void init_compute_clrm1_factors(pybind11::module& m) {
    m.def("compute_clrm1_factors", &compute_clrm1_factors);
}
