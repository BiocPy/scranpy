#include "parallel.h" // must be first, to set all macros.

#include "knncolle/knncolle.hpp"
#include "umappp/Umap.hpp"

typedef umappp::Umap<>::Status UmapStatus;

extern "C" {

UmapStatus* initialize_umap(const knncolle::NeighborList<>* neighbors, int num_epochs, double min_dist, double* Y, int nthreads) {
    umappp::Umap factory;
    factory.set_min_dist(min_dist).set_num_epochs(num_epochs).set_num_threads(nthreads);
    return new UmapStatus(factory.initialize(*neighbors, 2, Y));
}

int fetch_umap_status_nobs(const UmapStatus* ptr) {
    return ptr->nobs();
}

int fetch_umap_status_epoch(const UmapStatus* ptr) {
    return ptr->epoch();
}

int fetch_umap_status_num_epochs(const UmapStatus* ptr) {
    return ptr->num_epochs();
}

void free_umap_status(UmapStatus* ptr) {
    delete ptr;
}

UmapStatus* clone_umap_status(const UmapStatus* ptr, double* cloned) {
    auto out = new UmapStatus(*ptr);
    out->set_embedding(cloned, false);
    return out;
}

void run_umap(UmapStatus* status, int max_epoch) {
    status->run(max_epoch);
    return;
}

}
