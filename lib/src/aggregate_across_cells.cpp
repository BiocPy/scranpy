#include <vector>
#include <stdexcept>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "scran_aggregate/scran_aggregate.hpp"
#include "tatami_stats/tatami_stats.hpp"
#include "mattress.h"

#include "utils.h"

pybind11::tuple aggregate_across_cells(uintptr_t x, const pybind11::array& groups, int num_threads) {
    const auto& mat = mattress::cast(x)->ptr;
    size_t NC = mat->ncol();
    size_t NR = mat->nrow();
    if (static_cast<size_t>(groups.size()) != NC) {
        throw std::runtime_error("length of 'groups' should be equal to the number of columns in 'x'");
    }
    auto gptr = check_numpy_array<uint32_t>(groups);

    size_t ncombos = tatami_stats::total_groups<uint32_t>(gptr, NC);
    pybind11::array_t<double, pybind11::array::f_style> sums({ NR, ncombos });
    pybind11::array_t<uint32_t, pybind11::array::f_style> detected({ NR, ncombos });

    scran_aggregate::AggregateAcrossCellsBuffers<double, uint32_t> buffers;
    {
        buffers.sums.reserve(ncombos);
        buffers.detected.reserve(ncombos);
        double* osum = static_cast<double*>(sums.request().ptr);
        uint32_t* odet = static_cast<uint32_t*>(detected.request().ptr);
        for (size_t i = 0; i < ncombos; ++i, osum += NR, odet += NR) {
            buffers.sums.push_back(osum);
            buffers.detected.push_back(odet);
        }
    }

    scran_aggregate::AggregateAcrossCellsOptions opt;
    opt.num_threads = num_threads;
    scran_aggregate::aggregate_across_cells(*mat, gptr, buffers, opt);

    pybind11::tuple output(2);
    output[0] = sums;
    output[1] = detected;
    return output;
}

void init_aggregate_across_cells(pybind11::module& m) {
    m.def("aggregate_across_cells", &aggregate_across_cells);
}
