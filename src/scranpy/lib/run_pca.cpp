#include "parallel.h" // must be first, to set all macros.

#include "Mattress.h"
#include "scran/dimensionality_reduction/SimplePca.hpp"
#include "scran/dimensionality_reduction/ResidualPca.hpp"
#include "scran/dimensionality_reduction/MultiBatchPca.hpp"
#include <cstdint>

static void precheck_inputs(int32_t number, size_t NC, uint8_t use_subset, const uint8_t* subset) {
    if (number < 1) {
        throw std::runtime_error("requested number of PCs should be positive");
    }
    if (NC < number) {
        throw std::runtime_error("fewer cells than the requested number of PCs");
    }
}

/****************************/

//[[export]]
const double* fetch_simple_pca_coordinates(const void* x) {
    return reinterpret_cast<const scran::SimplePca::Results*>(x)->pcs.data();
}

//[[export]]
const double* fetch_simple_pca_variance_explained(const void* x) {
    return reinterpret_cast<const scran::SimplePca::Results*>(x)->variance_explained.data();
}

//[[export]]
double fetch_simple_pca_total_variance(const void* x) {
    return reinterpret_cast<const scran::SimplePca::Results*>(x)->total_variance;
}

//[[export]]
int32_t fetch_simple_pca_num_dims(const void* x) {
    return reinterpret_cast<const scran::SimplePca::Results*>(x)->pcs.rows();
}

//[[export]]
void free_simple_pca(void* x) {
    delete reinterpret_cast<scran::SimplePca::Results*>(x);
}

//[[export]]
void* run_simple_pca(const void* mat, int32_t number, uint8_t use_subset, const uint8_t* subset /** void_p */, uint8_t scale, int32_t num_threads) {
    const auto& ptr = reinterpret_cast<const Mattress*>(mat)->ptr;
    auto NR = ptr->nrow();
    auto NC = ptr->ncol();
    precheck_inputs(number, NC, use_subset, subset);

    scran::SimplePca pca;
    pca.set_rank(number).set_scale(scale).set_num_threads(num_threads);
    auto result = pca.run(ptr.get(), use_subset ? subset : NULL);

    return reinterpret_cast<void*>(new scran::SimplePca::Results(std::move(result)));
}

/****************************/

//[[export]]
const double* fetch_residual_pca_coordinates(const void* x) {
    return reinterpret_cast<const scran::ResidualPca::Results*>(x)->pcs.data();
}

//[[export]]
const double* fetch_residual_pca_variance_explained(const void* x) {
    return reinterpret_cast<const scran::ResidualPca::Results*>(x)->variance_explained.data();
}

//[[export]]
double fetch_residual_pca_total_variance(const void* x) {
    return reinterpret_cast<const scran::ResidualPca::Results*>(x)->total_variance;
}

//[[export]]
int32_t fetch_residual_pca_num_dims(const void* x) {
    return reinterpret_cast<const scran::ResidualPca::Results*>(x)->pcs.rows();
}

//[[export]]
void free_residual_pca(void* x) {
    delete reinterpret_cast<scran::ResidualPca::Results*>(x);
}

//[[export]]
void* run_residual_pca(
    const void* mat,
    const int32_t* /** numpy */ block,
    uint8_t equal_weights,
    int32_t number,
    uint8_t use_subset,
    const uint8_t* /** void_p */ subset,
    uint8_t scale,
    int32_t num_threads)
{
    const auto& ptr = reinterpret_cast<const Mattress*>(mat)->ptr;
    auto NR = ptr->nrow();
    auto NC = ptr->ncol();
    precheck_inputs(number, NC, use_subset, subset);

    scran::ResidualPca pca;
    pca.set_rank(number).set_scale(scale).set_num_threads(num_threads);
    pca.set_block_weight_policy(equal_weights ? scran::WeightPolicy::VARIABLE : scran::WeightPolicy::NONE);

    auto result = pca.run(ptr.get(), block, use_subset ? subset : NULL);
    return reinterpret_cast<void*>(new scran::ResidualPca::Results(std::move(result)));
}

/****************************/

//[[export]]
const double* fetch_multibatch_pca_coordinates(const void* x) {
    return reinterpret_cast<const scran::MultiBatchPca::Results*>(x)->pcs.data();
}

//[[export]]
const double* fetch_multibatch_pca_variance_explained(const void* x) {
    return reinterpret_cast<const scran::MultiBatchPca::Results*>(x)->variance_explained.data();
}

//[[export]]
double fetch_multibatch_pca_total_variance(const void* x) {
    return reinterpret_cast<const scran::MultiBatchPca::Results*>(x)->total_variance;
}

//[[export]]
int32_t fetch_multibatch_pca_num_dims(const void* x) {
    return reinterpret_cast<const scran::MultiBatchPca::Results*>(x)->pcs.rows();
}

//[[export]]
void free_multibatch_pca(void* x) {
    delete reinterpret_cast<scran::MultiBatchPca::Results*>(x);
}

//[[export]]
void* run_multibatch_pca(
    const void* mat,
    const int32_t* /** numpy */ block,
    uint8_t use_residuals,
    uint8_t equal_weights,
    int32_t number,
    uint8_t use_subset,
    const uint8_t* /** void_p */ subset,
    uint8_t scale,
    int32_t num_threads)
{
    const auto& ptr = reinterpret_cast<const Mattress*>(mat)->ptr;
    auto NR = ptr->nrow();
    auto NC = ptr->ncol();
    precheck_inputs(number, NC, use_subset, subset);

    scran::MultiBatchPca pca;
    pca.set_rank(number).set_scale(scale).set_num_threads(num_threads);
    pca.set_use_residuals(use_residuals);
    pca.set_block_weight_policy(equal_weights ? scran::WeightPolicy::VARIABLE : scran::WeightPolicy::NONE);

    auto result = pca.run(ptr.get(), block, use_subset ? subset : NULL);
    return reinterpret_cast<void*>(new scran::MultiBatchPca::Results(std::move(result)));
}
