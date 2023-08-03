#include "parallel.h" // must be first, to set all macros.

#include "knncolle/knncolle.hpp"
#include "umappp/Umap.hpp"

typedef umappp::Umap<>::Status UmapStatus;

//[[export]]
void* initialize_umap(const void* neighbors, int32_t num_epochs, double min_dist, double* Y /** numpy */, int32_t nthreads) {
    umappp::Umap factory;
    factory.set_min_dist(min_dist).set_num_epochs(num_epochs).set_num_threads(nthreads);
    return reinterpret_cast<void*>(new UmapStatus(factory.initialize(*reinterpret_cast<const knncolle::NeighborList<>*>(neighbors), 2, Y)));
}

//[[export]]
int32_t fetch_umap_status_nobs(const void* ptr) {
    return reinterpret_cast<const UmapStatus*>(ptr)->nobs();
}

//[[export]]
int32_t fetch_umap_status_epoch(const void* ptr) {
    return reinterpret_cast<const UmapStatus*>(ptr)->epoch();
}

//[[export]]
int32_t fetch_umap_status_num_epochs(const void* ptr) {
    return reinterpret_cast<const UmapStatus*>(ptr)->num_epochs();
}

//[[export]]
void free_umap_status(void* ptr) {
    delete reinterpret_cast<UmapStatus*>(ptr);
}

//[[export]]
void* clone_umap_status(const void* ptr, double* cloned /** numpy */) {
    auto out = new UmapStatus(*reinterpret_cast<const UmapStatus*>(ptr));
    out->set_embedding(cloned, false);
    return reinterpret_cast<void*>(out);
}

//[[export]]
void run_umap(void* status, int32_t max_epoch) {
    reinterpret_cast<UmapStatus*>(status)->run(max_epoch);
    return;
}
