#include <vector>
#include <cstdint>
#include <stdexcept>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "phyper/phyper.hpp"
#include "subpar/subpar.hpp"

#include "utils.h"

pybind11::array test_enrichment(const pybind11::array& overlap, uint32_t num_interest, const pybind11::array& set_sizes, uint32_t universe, bool log, int num_threads) {
    size_t nsets = overlap.size();
    if (nsets != static_cast<size_t>(set_sizes.size())) {
        throw std::runtime_error("'overlap' and 'set_sizes' should have the same length");
    }

    phyper::Options opt;
    opt.upper_tail = true;
    opt.log = log;

    pybind11::array_t<double> output(nsets);
    double* optr = static_cast<double*>(output.request().ptr); // avoid any python references inside the parallel section.
    auto olptr = check_numpy_array<uint32_t>(overlap);
    auto ssptr = check_numpy_array<uint32_t>(set_sizes);

    subpar::parallelize(num_threads, nsets, [&](int, size_t start, size_t length) {
        for (size_t s = start, end = start + length; s < end; ++s) {
            optr[s] = phyper::compute(
                olptr[s],
                ssptr[s],
                universe - ssptr[s],
                num_interest,
                opt
            );
        }
    });

    return output;
}

void init_test_enrichment(pybind11::module& m) {
    m.def("test_enrichment", &test_enrichment);
}
