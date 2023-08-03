#include "parallel.h" // must be first, to set all macros.

#include "knncolle/knncolle.hpp"
#include "qdtsne/qdtsne.hpp"

typedef qdtsne::Tsne<>::Status<int> TsneStatus;

//[[export]]
void* initialize_tsne(const void* neighbors, double perplexity, int32_t nthreads) {
    qdtsne::Tsne factory;
    factory.set_perplexity(perplexity).set_num_threads(nthreads);
    factory.set_max_depth(7); // speed up iterations, avoid problems with duplicates.
    return reinterpret_cast<void*>(new TsneStatus(factory.template initialize<>(*reinterpret_cast<const knncolle::NeighborList<>*>(neighbors))));
}

//[[export]]
void randomize_tsne_start(size_t n, double* Y /** numpy */, int32_t seed) {
    qdtsne::initialize_random(Y, n, seed);
    return;
}

//[[export]]
int32_t fetch_tsne_status_iteration(const void* ptr) {
    return reinterpret_cast<const TsneStatus*>(ptr)->iteration();
}

//[[export]]
int32_t fetch_tsne_status_nobs(const void* ptr) {
    return reinterpret_cast<const TsneStatus*>(ptr)->nobs();
}

//[[export]]
void free_tsne_status(void* ptr) {
    delete reinterpret_cast<TsneStatus*>(ptr);
}

//[[export]]
void* clone_tsne_status(const void* ptr) {
    return reinterpret_cast<void*>(new TsneStatus(*reinterpret_cast<const TsneStatus*>(ptr)));
}

//[[export]]
int32_t perplexity_to_k(double perplexity) {
    return std::ceil(perplexity * 3);
}

//[[export]]
void run_tsne(void* status, int32_t maxiter, double* Y /** numpy */) {
    reinterpret_cast<TsneStatus*>(status)->run(Y, maxiter);
    return;
}
