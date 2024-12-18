#include "parallel.h" // must be first, to set all macros.

#include "Mattress.h"
#include "scran/feature_selection/ModelGeneVariances.hpp"
#include <cstdint>

//[[export]]
void model_gene_variances(
    const void* mat,
    double* /** numpy */ means,
    double* /** numpy */ variances,
    double* /** numpy */ fitted,
    double* /** numpy */ residuals,
    double span,
    int32_t num_threads)
{
    scran::ModelGeneVariances runner;
    runner.set_num_threads(num_threads);
    runner.set_span(span);
    runner.run(reinterpret_cast<const Mattress*>(mat)->ptr.get(), means, variances, fitted, residuals);
}

//[[export]]
void model_gene_variances_blocked(
    const void* mat,
    double* /** numpy */ ave_means,
    double* /** numpy */ ave_detected,
    double* /** numpy */ ave_fitted,
    double* /** numpy */ ave_residuals,
    int32_t num_blocks,
    const int32_t* /** void_p */ block,
    uintptr_t* /** void_p */ block_means,
    uintptr_t* /** void_p */ block_variances,
    uintptr_t* /** void_p */ block_fitted,
    uintptr_t* /** void_p */ block_residuals,
    double span,
    int32_t num_threads)
{
    scran::ModelGeneVariances runner;
    runner.set_num_threads(num_threads);
    runner.set_span(span);

    std::vector<double*> mean_ptrs(num_blocks), variance_ptrs(num_blocks), fitted_ptrs(num_blocks), residual_ptrs(num_blocks);
    for (int32_t b = 0; b < num_blocks; ++b) {
        mean_ptrs[b] = reinterpret_cast<double*>(block_means[b]);
        variance_ptrs[b] = reinterpret_cast<double*>(block_variances[b]);
        fitted_ptrs[b] = reinterpret_cast<double*>(block_fitted[b]);
        residual_ptrs[b] = reinterpret_cast<double*>(block_residuals[b]);
    }

    runner.run_blocked(reinterpret_cast<const Mattress*>(mat)->ptr.get(), block, mean_ptrs, variance_ptrs, fitted_ptrs, residual_ptrs, ave_means, ave_detected, ave_fitted, ave_residuals);
    return;
}
