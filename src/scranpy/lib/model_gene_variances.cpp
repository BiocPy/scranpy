#include "Mattress.h"
#include "scran/feature_selection/ModelGeneVariances.hpp"
#include <cstdint>

extern "C" {

void model_gene_variances(const Mattress* mat, double* means, double* variances, double* fitted, double* residuals, double span, int num_threads) {
    scran::ModelGeneVariances runner;
    runner.set_num_threads(num_threads);
    runner.set_span(span);
    runner.run(mat->ptr.get(), means, variances, fitted, residuals);
}
    
void model_gene_variances_blocked(
    const Mattress* mat, 
    double* ave_means,
    double* ave_detected,
    double* ave_fitted,
    double* ave_residuals,
    int num_blocks, 
    const int32_t* block, 
    uintptr_t* block_means,
    uintptr_t* block_variances,
    uintptr_t* block_fitted,
    uintptr_t* block_residuals,
    double span, 
    int num_threads)
{
    scran::ModelGeneVariances runner;
    runner.set_num_threads(num_threads);
    runner.set_span(span);

    std::vector<double*> mean_ptrs(num_blocks), variance_ptrs(num_blocks), fitted_ptrs(num_blocks), residual_ptrs(num_blocks);
    for (int b = 0; b < num_blocks; ++b) {
        mean_ptrs[b] = reinterpret_cast<double*>(block_means[b]);
        variance_ptrs[b] = reinterpret_cast<double*>(block_variances[b]);
        fitted_ptrs[b] = reinterpret_cast<double*>(block_fitted[b]);
        residual_ptrs[b] = reinterpret_cast<double*>(block_residuals[b]);
    }

    runner.run_blocked(mat->ptr.get(), block, mean_ptrs, variance_ptrs, fitted_ptrs, residual_ptrs, ave_means, ave_detected, ave_fitted, ave_residuals);
    return;
}

}
