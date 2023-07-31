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

extern "C" {

scran::BuildSnnGraph::Results* build_snn_graph_from_nn_results(const knncolle::NeighborList<>* x, const char* weight_scheme, int num_threads) {
    scran::BuildSnnGraph runner;
    runner.set_num_threads(num_threads);
    runner.set_weighting_scheme(resolve_weighting_scheme(weight_scheme));
    auto res = runner.run(*x);
    return new scran::BuildSnnGraph::Results(std::move(res));
}

scran::BuildSnnGraph::Results* build_snn_graph_from_nn_index(const knncolle::Base<>* x, int num_neighbors, const char* weight_scheme, int num_threads) {
    scran::BuildSnnGraph runner;
    runner.set_neighbors(num_neighbors);
    runner.set_num_threads(num_threads);
    runner.set_weighting_scheme(resolve_weighting_scheme(weight_scheme));
    auto res = runner.run(x);
    return new scran::BuildSnnGraph::Results(std::move(res));
}

int fetch_snn_graph_edges(const scran::BuildSnnGraph::Results* ptr) {
    return ptr->weights.size();
}

const int* fetch_snn_graph_indices(const scran::BuildSnnGraph::Results* ptr) {
    return ptr->edges.data();
}

const double* fetch_snn_graph_weights(const scran::BuildSnnGraph::Results* ptr) {
    return ptr->weights.data();
}

void free_snn_graph(scran::BuildSnnGraph::Results* ptr) {
    delete ptr;
}

}
