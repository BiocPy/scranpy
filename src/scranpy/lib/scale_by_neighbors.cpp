#include "parallel.h" // must be first, to set all macros.

#include "scran/dimensionality_reduction/ScaleByNeighbors.hpp"
#include <cstdint>

//[[export]]
void scale_by_neighbors(int32_t nembeddings, const uintptr_t* indices /** void_p */, int32_t nneighbors, double* /** numpy */ output, int32_t nthreads) {
    scran::ScaleByNeighbors scaler;
    scaler.set_neighbors(nneighbors);
    scaler.set_num_threads(nthreads);

    std::vector<std::pair<double, double> > collected;
    for (int32_t e = 0; e < nembeddings; ++e) {
        auto curind = reinterpret_cast<const knncolle::Base<>*>(indices[e]);
        collected.push_back(scaler.compute_distance(curind));
    }

    auto scale = scran::ScaleByNeighbors::compute_scale(collected);
    std::copy(scale.begin(), scale.end(), output);
}

//[[export]]
void combine_embeddings(
    int32_t nembeddings,
    const int32_t* ndims /** numpy */,
    int32_t ncells,
    const uintptr_t* embeddings /** void_p */,
    const double* scaling /** numpy */,
    double* output /** numpy */)
{
    std::vector<double> scaling_vec(scaling, scaling + nembeddings);
    std::vector<int> dim_vec(ndims, ndims + nembeddings);

    std::vector<const double*> embed_vec(nembeddings);
    for (int32_t e = 0; e < nembeddings; ++e) {
        embed_vec[e] = reinterpret_cast<const double*>(embeddings[e]);
    }

    scran::ScaleByNeighbors::combine_scaled_embeddings(dim_vec, ncells, embed_vec, scaling_vec, output);
}
