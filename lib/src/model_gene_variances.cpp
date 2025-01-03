#include <vector>
#include <stdexcept>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "scran_variances/scran_variances.hpp"
#include "mattress.h"

#include "utils.h"
#include "block.h"

pybind11::tuple model_gene_variances(
    uintptr_t x,
    std::optional<pybind11::array> maybe_block,
    size_t nblocks,
    std::string block_weight_policy,
    pybind11::tuple variable_block_weight,
    bool mean_filter,
    double min_mean,
    bool transform,
    double span,
    bool use_min_width,
    double min_width,
    int min_window_count,
    int num_threads)
{
    scran_variances::ModelGeneVariancesOptions opt;
    opt.fit_variance_trend_options.mean_filter = mean_filter;
    opt.fit_variance_trend_options.minimum_mean = min_mean;
    opt.fit_variance_trend_options.transform = transform;
    opt.fit_variance_trend_options.span = span;
    opt.fit_variance_trend_options.use_minimum_width = use_min_width;
    opt.fit_variance_trend_options.minimum_width = min_width;
    opt.fit_variance_trend_options.minimum_window_count = min_window_count;
    opt.num_threads = num_threads;

    opt.block_weight_policy = parse_block_weight_policy(block_weight_policy);
    opt.variable_block_weight_parameters = parse_variable_block_weight(variable_block_weight);

    const auto& mat = mattress::cast(x)->ptr;
    size_t nc = mat->ncol();
    size_t nr = mat->nrow();

    pybind11::array_t<double> means(nr), variances(nr), fitted(nr), residuals(nr);
    scran_variances::ModelGeneVariancesBuffers<double> buffers;
    buffers.means = static_cast<double*>(means.request().ptr);
    buffers.variances = static_cast<double*>(variances.request().ptr);
    buffers.fitted = static_cast<double*>(fitted.request().ptr);
    buffers.residuals = static_cast<double*>(residuals.request().ptr);

    pybind11::tuple output(5);

    if (maybe_block.has_value()) {
        const auto& block = *maybe_block;
        if (block.size() != nc) {
            throw std::runtime_error("'block' must be the same length as the number of cells");
        }
        auto bptr = check_numpy_array<uint32_t>(block);

        scran_variances::ModelGeneVariancesBlockedBuffers<double> bbuffers;
        bbuffers.average = buffers;
        bbuffers.per_block.resize(nblocks);

        std::vector<pybind11::array_t<double> > block_mean, block_var, block_fit, block_res;
        block_mean.reserve(nblocks);
        block_var.reserve(nblocks);
        block_fit.reserve(nblocks);
        block_res.reserve(nblocks);

        for (size_t b = 0; b < nblocks; ++b) {
            block_mean.emplace_back(nr);
            bbuffers.per_block[b].means = static_cast<double*>(block_mean.back().request().ptr);
            block_var.emplace_back(nr);
            bbuffers.per_block[b].variances = static_cast<double*>(block_var.back().request().ptr);
            block_fit.emplace_back(nr);
            bbuffers.per_block[b].fitted = static_cast<double*>(block_fit.back().request().ptr);
            block_res.emplace_back(nr);
            bbuffers.per_block[b].residuals = static_cast<double*>(block_res.back().request().ptr);
        }

        scran_variances::model_gene_variances_blocked(*mat, bptr, bbuffers, opt);

        pybind11::tuple pb(nblocks);
        for (size_t b = 0; b < nblocks; ++b) {
            pybind11::tuple current(4);
            current[0] = block_mean[b];
            current[1] = block_var[b];
            current[2] = block_fit[b];
            current[3] = block_res[b];
            pb[b] = current;
        }
        output[4] = pb;

    } else {
        scran_variances::model_gene_variances(*mat, buffers, opt);
        output[4] = pybind11::none();
    }

    output[0] = means;
    output[1] = variances;
    output[2] = fitted;
    output[3] = residuals;
    return output;
}

void init_model_gene_variances(pybind11::module& m) {
    m.def("model_gene_variances", &model_gene_variances);
}
