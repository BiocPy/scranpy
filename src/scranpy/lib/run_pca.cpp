#include "Mattress.h"
#include "scran/dimensionality_reduction/SimplePca.hpp"
#include "scran/dimensionality_reduction/ResidualPca.hpp"
#include "scran/dimensionality_reduction/MultiBatchPca.hpp"
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

extern "C" {

const double* fetch_residual_pca_coordinates(const scran::ResidualPca::Results* x) {
    return x->pcs.data();
}

const double* fetch_residual_pca_variance_explained(const scran::ResidualPca::Results* x) {
    return x->variance_explained.data();
}

double fetch_residual_pca_total_variance(const scran::ResidualPca::Results* x) {
    return x->total_variance;
}

int fetch_residual_pca_num_dims(const scran::ResidualPca::Results* x) {
    return x->pcs.rows();
}

void free_residual_pca(scran::ResidualPca::Results* x) {
    delete x;
}

scran::ResidualPca::Results* run_residual_pca(const Mattress* mat, const int32_t* block, int number, uint8_t use_subset, const uint8_t* subset, uint8_t scale, int num_threads) {
    auto& ptr = mat->ptr;
    auto NR = ptr->nrow();
    auto NC = ptr->ncol();
    precheck_inputs(number, NC, use_subset, subset);

    scran::ResidualPca pca;
    pca.set_rank(number).set_scale(scale).set_num_threads(num_threads);
    auto result = pca.run(ptr.get(), block, use_subset ? subset : NULL);

    return new scran::ResidualPca::Results(std::move(result)); 
}

}

/****************************/

extern "C" {

const double* fetch_multibatch_pca_coordinates(const scran::MultiBatchPca::Results* x) {
    return x->pcs.data();
}

const double* fetch_multibatch_pca_variance_explained(const scran::MultiBatchPca::Results* x) {
    return x->variance_explained.data();
}

double fetch_multibatch_pca_total_variance(const scran::MultiBatchPca::Results* x) {
    return x->total_variance;
}

int fetch_multibatch_pca_num_dims(const scran::MultiBatchPca::Results* x) {
    return x->pcs.rows();
}

void free_multibatch_pca(scran::MultiBatchPca::Results* x) {
    delete x;
}

scran::MultiBatchPca::Results* run_multibatch_pca(const Mattress* mat, const int32_t* block, int number, uint8_t use_subset, const uint8_t* subset, uint8_t scale, int num_threads) {
    auto& ptr = mat->ptr;
    auto NR = ptr->nrow();
    auto NC = ptr->ncol();
    precheck_inputs(number, NC, use_subset, subset);

    scran::MultiBatchPca pca;
    pca.set_rank(number).set_scale(scale).set_num_threads(num_threads);
    auto result = pca.run(ptr.get(), block, use_subset ? subset : NULL);

    return new scran::MultiBatchPca::Results(std::move(result)); 
}

}
