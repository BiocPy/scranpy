#include "parallel.h" // must be first, to set all macros.

#include "knncolle/knncolle.hpp"
#include "qdtsne/qdtsne.hpp"

typedef qdtsne::Tsne<>::Status<int> TsneStatus;

extern "C" {

TsneStatus* initialize_tsne(const knncolle::NeighborList<>* neighbors, double perplexity, int nthreads) {
    qdtsne::Tsne factory;
    factory.set_perplexity(perplexity).set_num_threads(nthreads);
    factory.set_max_depth(7); // speed up iterations, avoid problems with duplicates.
    return new TsneStatus(factory.template initialize<>(*neighbors));
}

void randomize_tsne_start(size_t n, double* Y, int seed) {
    qdtsne::initialize_random(Y, n, seed);
    return;
}

int fetch_tsne_status_iteration(const TsneStatus* ptr) {
    return ptr->iteration();
}

int fetch_tsne_status_nobs(const TsneStatus* ptr) {
    return ptr->nobs();
}

void free_tsne_status(TsneStatus* ptr) {
    delete ptr;
}

TsneStatus* clone_tsne_status(const TsneStatus* ptr) {
    return new TsneStatus(*ptr);
}

int perplexity_to_k(double perplexity) {
    return std::ceil(perplexity * 3);
}

void run_tsne(TsneStatus* status, int maxiter, double* Y) {
    status->run(Y, maxiter);
    return;
}

}
