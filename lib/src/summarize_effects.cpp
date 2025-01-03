#include <vector>
#include <stdexcept>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "scran_markers/scran_markers.hpp"

#include "utils.h"

pybind11::tuple summarize_effects(const pybind11::array& effects, int num_threads) {
    auto ebuffer = effects.request();
    if (ebuffer.shape.size() != 3) {
        throw std::runtime_error("expected a 3-dimensional array for the effects");
    }
    size_t num_groups = ebuffer.shape[0];
    if (num_groups != ebuffer.shape[1]) {
        throw std::runtime_error("first two dimensions of the effects array should have the same extent");
    }
    size_t num_genes = ebuffer.shape[2];
    if ((effects.flags() & pybind11::array::f_style) == 0) {
        throw std::runtime_error("expected Fortran-style storage for the effects");
    }
    if (!effects.dtype().is(pybind11::dtype::of<double>())) {
        throw std::runtime_error("unexpected dtype for the array of effects");
    }
    const double* eptr = get_numpy_array_data<double>(effects);

    std::vector<pybind11::array_t<double> > min, mean, median, max;
    min.reserve(num_groups);
    mean.reserve(num_groups);
    median.reserve(num_groups);
    max.reserve(num_groups);
    std::vector<pybind11::array_t<uint32_t> > min_rank;
    min_rank.reserve(num_groups);

    std::vector<scran_markers::SummaryBuffers<double, uint32_t> > groupwise;
    groupwise.resize(num_groups);
    for (int g = 0; g < num_groups; ++g) {
        min.emplace_back(num_genes);
        groupwise[g].min = static_cast<double*>(min.back().request().ptr);
        mean.emplace_back(num_genes);
        groupwise[g].mean = static_cast<double*>(mean.back().request().ptr);
        median.emplace_back(num_genes);
        groupwise[g].median = static_cast<double*>(median.back().request().ptr);
        max.emplace_back(num_genes);
        groupwise[g].max = static_cast<double*>(max.back().request().ptr);
        min_rank.emplace_back(num_genes);
        groupwise[g].min_rank = static_cast<uint32_t*>(min_rank.back().request().ptr);
    }

    scran_markers::SummarizeEffectsOptions opt;
    opt.num_threads = num_threads;
    scran_markers::summarize_effects(num_genes, num_groups, eptr, groupwise, opt);

    pybind11::tuple output(num_groups);
    for (size_t g = 0; g < num_groups; ++g) {
        pybind11::tuple current(5);
        current[0] = min[g];
        current[1] = mean[g];
        current[2] = median[g];
        current[3] = max[g];
        current[4] = min_rank[g];
        output[g] = current;
    }
    return output;
}

void init_summarize_effects(pybind11::module& m) {
    m.def("summarize_effects", &summarize_effects);
}
