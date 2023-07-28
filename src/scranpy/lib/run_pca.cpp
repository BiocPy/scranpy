#include "Mattress.h"
#include "scran/dimensionality_reduction/SimplePca.hpp"
#include <cstdint>

static void precheck_inputs(int number, size_t NC, bool use_subset, const uint8_t* subset) {
    if (number < 1) {
        throw std::runtime_error("requested number of PCs should be positive");
    }
    if (NC < number) {
        throw std::runtime_error("fewer cells than the requested number of PCs");
    }
}

/****************************/

extern "C" {

const double* fetch_simple_pca_coordinates(const scran::SimplePca::Results* x) {
    return x->pcs.data();
}

const double* fetch_simple_pca_variance_explained(const scran::SimplePca::Results* x) {
    return x->variance_explained.data();
}

double fetch_simple_pca_total_variance(const scran::SimplePca::Results* x) {
    return x->total_variance;
}

int fetch_simple_pca_num_dims(const scran::SimplePca::Results* x) {
    return x->pcs.rows();
}

void free_simple_pca(scran::SimplePca::Results* x) {
    delete x;
}

scran::SimplePca::Results* run_simple_pca(const Mattress* mat, int number, uint8_t use_subset, const uint8_t* subset, uint8_t scale, int num_threads) {
    auto& ptr = mat->ptr;
    auto NR = ptr->nrow();
    auto NC = ptr->ncol();
    precheck_inputs(number, NC, use_subset, subset);

    scran::SimplePca pca;
    pca.set_rank(number).set_scale(scale).set_num_threads(num_threads);
    auto result = pca.run(ptr.get(), use_subset ? subset : NULL);

    return new scran::SimplePca::Results(std::move(result)); 
}

}

/****************************/

