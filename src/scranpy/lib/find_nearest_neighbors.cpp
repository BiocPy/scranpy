#include "knncolle/knncolle.hpp"
#include <cstdint>

extern "C" {

knncolle::Base<>* build_neighbor_index(int ndim, int nobs, const double* ptr, bool approximate) {
    if (approximate) {
        return new knncolle::AnnoyEuclidean<>(ndim, nobs, ptr);
    } else {
        return new knncolle::VpTreeEuclidean<>(ndim, nobs, ptr);
    }
}

int fetch_neighbor_index_ndim(const knncolle::Base<>* ptr) {
    return ptr->ndim();
}

int fetch_neighbor_index_nobs(const knncolle::Base<>* ptr) {
    return ptr->nobs();
}

void free_neighbor_index(knncolle::Base<>* ptr) {
    delete ptr;
}

}

extern "C" {

knncolle::NeighborList<>* find_nearest_neighbors(const knncolle::Base<>* index, int k, int nthreads) {
    auto output = knncolle::find_nearest_neighbors(index, k, nthreads);
    return new knncolle::NeighborList<>(std::move(output));
}

int fetch_neighbor_results_nobs(const knncolle::NeighborList<>* ptr) {
    return ptr->size();
}

int fetch_neighbor_results_k(const knncolle::NeighborList<>* ptr) {
    if (ptr->empty()) {
        return 0;
    } else {
        return ptr->front().size();
    }
}

void fetch_neighbor_results_indices(const knncolle::NeighborList<>* ptr, int i, int32_t* outdex, double* outdist) {
    const auto& chosen = (*ptr)[i];
    for (const auto& p : chosen) {
        *(outdex++) = p.first;
        *(outdist++) = p.second;
    }
    return;
}

void free_neighbor_results(knncolle::NeighborList<>* ptr) {
    delete ptr;
}

}
