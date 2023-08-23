#include "parallel.h" // must be first, to set all macros.

#include "knncolle/knncolle.hpp"
#include "scran/clustering/BuildSnnGraph.hpp"
#include <cstring>

scran::BuildSnnGraph::Scheme resolve_weighting_scheme(const char* weight_scheme) {
    if (std::strcmp(weight_scheme, "ranked") == 0) {
        return scran::BuildSnnGraph::RANKED;
    } else if (std::strcmp(weight_scheme, "number") == 0) {
        return scran::BuildSnnGraph::NUMBER;
    } else if (std::strcmp(weight_scheme, "jaccard") == 0) {
        return scran::BuildSnnGraph::JACCARD;
    }

    throw std::runtime_error("unknown weighting scheme '" + std::string(weight_scheme) + "'");
    return scran::BuildSnnGraph::RANKED; // this has no purpose but to avoid compiler warnings.
}

//[[export]]
void* build_snn_graph_from_nn_results(const void* x, const char* weight_scheme, int32_t num_threads) {
    scran::BuildSnnGraph runner;
    runner.set_num_threads(num_threads);
    runner.set_weighting_scheme(resolve_weighting_scheme(weight_scheme));
    auto res = runner.run(*reinterpret_cast<const knncolle::NeighborList<>*>(x));
    return reinterpret_cast<void*>(new scran::BuildSnnGraph::Results(std::move(res)));
}

//[[export]]
void* build_snn_graph_from_nn_index(const void* x, int32_t num_neighbors, const char* weight_scheme, int32_t num_threads) {
    scran::BuildSnnGraph runner;
    runner.set_neighbors(num_neighbors);
    runner.set_num_threads(num_threads);
    runner.set_weighting_scheme(resolve_weighting_scheme(weight_scheme));
    auto res = runner.run(reinterpret_cast<const knncolle::Base<>*>(x));
    return reinterpret_cast<void*>(new scran::BuildSnnGraph::Results(std::move(res)));
}

//[[export]]
int32_t fetch_snn_graph_edges(const void* ptr) {
    return reinterpret_cast<const scran::BuildSnnGraph::Results*>(ptr)->weights.size();
}

//[[export]]
const int* fetch_snn_graph_indices(const void* ptr) {
    return reinterpret_cast<const scran::BuildSnnGraph::Results*>(ptr)->edges.data();
}

//[[export]]
const double* fetch_snn_graph_weights(const void* ptr) {
    return reinterpret_cast<const scran::BuildSnnGraph::Results*>(ptr)->weights.data();
}

//[[export]]
void free_snn_graph(void* ptr) {
    delete reinterpret_cast<const scran::BuildSnnGraph::Results*>(ptr);
}
