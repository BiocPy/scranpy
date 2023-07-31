#include "parallel.h" // must be first, to set all macros.

#include "knncolle/knncolle.hpp"
#include <cstdint>

//[[export]]
void* build_neighbor_index(int ndim, int nobs, const double* ptr, uint8_t approximate) {
    knncolle::Base<>* output;
    if (approximate) {
        output = new knncolle::AnnoyEuclidean<>(ndim, nobs, ptr);
    } else {
        output = new knncolle::VpTreeEuclidean<>(ndim, nobs, ptr);
    }
    return reinterpret_cast<void*>(output);
}

//[[export]]
int fetch_neighbor_index_ndim(const void* ptr) {
    return reinterpret_cast<const knncolle::Base<>*>(ptr)->ndim();
}

//[[export]]
int fetch_neighbor_index_nobs(const void* ptr) {
    return reinterpret_cast<const knncolle::Base<>*>(ptr)->nobs();
}

//[[export]]
void free_neighbor_index(void* ptr) {
    delete reinterpret_cast<const knncolle::Base<>*>(ptr);
}

//[[export]]
void* find_nearest_neighbors(const void* index, int k, int nthreads) {
    auto output = knncolle::find_nearest_neighbors(reinterpret_cast<const knncolle::Base<>*>(index), k, nthreads);
    return reinterpret_cast<void*>(new knncolle::NeighborList<>(std::move(output)));
}

//[[export]]
int fetch_neighbor_results_nobs(const void* ptr) {
    return reinterpret_cast<knncolle::NeighborList<>*>(ptr)->size();
}

//[[export]]
int fetch_neighbor_results_k(const void* ptr0) {
    auto ptr = reinterpret_cast<const knncolle::NeighborList<>*>(ptr0);
    if (ptr->empty()) {
        return 0;
    } else {
        return ptr->front().size();
    }
}

//[[export]]
void fetch_neighbor_results_single(const void* ptr0, int i, int32_t* outdex, double* outdist) {
    auto ptr = reinterpret_cast<const knncolle::NeighborList<>*>(ptr0);
    const auto& chosen = (*ptr)[i];
    for (const auto& p : chosen) {
        *(outdex++) = p.first;
        *(outdist++) = p.second;
    }
    return;
}

//[[export]]
void free_neighbor_results(void* ptr) {
    delete reinterpret_cast<knncolle::NeighborList<>*>(ptr);
}

//[[export]]
void serialize_neighbor_results(const void* ptr0, int32_t* outdex, double* outdist) {
    auto ptr = reinterpret_cast<const knncolle::NeighborList<>*>(ptr0);
    for (int o = 0, end = ptr->size(); o < end; ++o) {
        const auto& chosen = (*ptr)[o];
        for (const auto& p : chosen) {
            *(outdex++) = p.first;
            *(outdist++) = p.second;
        }
    }
}

//[[export]]
void* unserialize_neighbor_results(int nobs, int k, int32_t* indices, double* distances) {
    knncolle::NeighborList<> output(nobs);
    for (int o = 0; o < nobs; ++o) {
        auto& chosen = output[o];
        for (int i = 0; i < k; ++i) {
            chosen.emplace_back(*(indices++), *(distances++));
        }
    }
    return reinterpret_cast<void*>(new knncolle::NeighborList<>(std::move(output)));
}
