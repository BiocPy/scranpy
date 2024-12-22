#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "umappp/umappp.hpp"

#include "neighbors.h"

pybind11::array run_umap(const pybind11::array& nnidx, const pybind11::array& nndist, int ndim, double min_dist, int seed, int num_epochs, int num_threads, bool parallel_optimization) {
    auto neighbors = unpack_neighbors<uint32_t, float>(nnidx, nndist);
    size_t nobs = neighbors.size();

    umappp::Options opt;
    opt.min_dist = min_dist;
    opt.seed = seed;
    opt.num_epochs = num_epochs;
    opt.num_threads = num_threads;
    opt.parallel_optimization = parallel_optimization;

    std::vector<float> embedding(ndim * nobs);
    auto status = umappp::initialize(std::move(neighbors), ndim, embedding.data(), opt);
    status.run();

    pybind11::array_t<double, pybind11::array::f_style> output({ static_cast<size_t>(ndim), nobs });
    std::copy(embedding.begin(), embedding.end(), static_cast<double*>(output.request().ptr));
    return output;
}

void init_run_umap(pybind11::module& m) {
    m.def("run_umap", &run_umap);
}
