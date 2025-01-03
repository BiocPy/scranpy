#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "qdtsne/qdtsne.hpp"

#include "neighbors.h"
#include <iostream>

pybind11::array run_tsne(const pybind11::array& nnidx, const pybind11::array& nndist, double perplexity, int leaf_approx, int max_depth, int max_iter, int seed, int num_threads) {
    qdtsne::Options opt;
    opt.perplexity = perplexity;
    opt.infer_perplexity = false; // rely on the perplexity supplied by the user.
    opt.leaf_approximation = leaf_approx;
    opt.max_depth = max_depth;
    opt.max_iterations = max_iter;
    opt.num_threads = num_threads;

    auto neighbors = unpack_neighbors<uint32_t, double>(nnidx, nndist);
    size_t nobs = neighbors.size();
    auto status = qdtsne::initialize<2>(std::move(neighbors), opt);

    pybind11::array_t<double, pybind11::array::f_style> output({ static_cast<size_t>(2), nobs });
    auto optr = static_cast<double*>(output.request().ptr);
    qdtsne::initialize_random<2>(optr, nobs, seed);
    status.run(optr);

    return output;
}

int perplexity_to_neighbors(double p) {
    return qdtsne::perplexity_to_k(p);
}

void init_run_tsne(pybind11::module& m) {
    m.def("run_tsne", &run_tsne);
    m.def("perplexity_to_neighbors", &perplexity_to_neighbors);
}
