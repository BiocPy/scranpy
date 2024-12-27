#include <vector>
#include <stdexcept>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"
#include "scran_markers/scran_markers.hpp"
#include "mattress.h"

#include "utils.h"
#include "block.h"

pybind11::tuple score_markers_summary(
    uintptr_t x,
    const pybind11::array& groups,
    size_t num_groups,
    std::optional<pybind11::array> maybe_block,
    std::string block_weight_policy,
    const pybind11::tuple& variable_block_weight,
    double threshold,
    int num_threads,
    bool compute_cohens_d,
    bool compute_auc,
    bool compute_delta_mean,
    bool compute_delta_detected)
{
    const auto& mat = mattress::cast(x)->ptr;
    size_t NC = mat->ncol();
    size_t NR = mat->nrow();
    if (static_cast<size_t>(groups.size()) != NC) {
        throw std::runtime_error("'groups' must have length equal to the number of cells");
    }

    scran_markers::ScoreMarkersSummaryOptions opt;
    opt.threshold = threshold;
    opt.num_threads = num_threads;
    opt.block_weight_policy = parse_block_weight_policy(block_weight_policy);
    opt.variable_block_weight_parameters = parse_variable_block_weight(variable_block_weight);

    scran_markers::ScoreMarkersSummaryBuffers<double, uint32_t> buffers;

    pybind11::array_t<double, pybind11::array::f_style> means({ NR, num_groups });
    pybind11::array_t<double, pybind11::array::f_style> detected({ NR, num_groups });
    {
        auto mptr = static_cast<double*>(means.request().ptr);
        auto dptr = static_cast<double*>(detected.request().ptr);
        buffers.mean.reserve(num_groups);
        buffers.detected.reserve(num_groups);
        size_t out_offset = 0;
        for (size_t g = 0; g < num_groups; ++g) {
            buffers.mean.emplace_back(mptr + out_offset);
            buffers.detected.emplace_back(dptr + out_offset);
            out_offset += NR;
        }
    }

    std::vector<pybind11::array_t<double> > cohens_min, cohens_mean, cohens_median, cohens_max;
    std::vector<pybind11::array_t<double> > auc_min, auc_mean, auc_median, auc_max;
    std::vector<pybind11::array_t<double> > dm_min, dm_mean, dm_median, dm_max;
    std::vector<pybind11::array_t<double> > dd_min, dd_mean, dd_median, dd_max;
    std::vector<pybind11::array_t<uint32_t> > cohens_mr, auc_mr, dm_mr, dd_mr;

    auto initialize = [&](
        std::vector<scran_markers::SummaryBuffers<double, uint32_t> >& ptrs,
        std::vector<pybind11::array_t<double> >& min,
        std::vector<pybind11::array_t<double> >& mean,
        std::vector<pybind11::array_t<double> >& median,
        std::vector<pybind11::array_t<double> >& max,
        std::vector<pybind11::array_t<uint32_t> >& min_rank
    ) {
        ptrs.resize(num_groups);
        min.reserve(num_groups);
        mean.reserve(num_groups);
        median.reserve(num_groups);
        max.reserve(num_groups);
        min_rank.reserve(num_groups);
        for (int g = 0; g < num_groups; ++g) {
            min.emplace_back(NR);
            ptrs[g].min = static_cast<double*>(min.back().request().ptr);
            mean.emplace_back(NR);
            ptrs[g].mean = static_cast<double*>(mean.back().request().ptr);
            median.emplace_back(NR);
            ptrs[g].median = static_cast<double*>(median.back().request().ptr);
            max.emplace_back(NR);
            ptrs[g].max = static_cast<double*>(max.back().request().ptr);
            min_rank.emplace_back(NR);
            ptrs[g].min_rank = static_cast<uint32_t*>(min_rank.back().request().ptr);
        }
    };

    if (compute_cohens_d) {
        initialize(buffers.cohens_d, cohens_min, cohens_mean, cohens_median, cohens_max, cohens_mr);
    }
    if (compute_delta_mean) {
        initialize(buffers.delta_mean, dm_min, dm_mean, dm_median, dm_max, dm_mr);
    }
    if (compute_delta_detected) {
        initialize(buffers.delta_detected, dd_min, dd_mean, dd_median, dd_max, dd_mr);
    }
    if (compute_auc) {
        initialize(buffers.auc, auc_min, auc_mean, auc_median, auc_max, auc_mr);
    }

    auto gptr = check_numpy_array<uint32_t>(groups);
    if (maybe_block.has_value()) {
        const auto& block = *maybe_block;
        if (block.size() != NC) {
            throw std::runtime_error("'block' must be the same length as the number of cells");
        }
        auto bptr = check_numpy_array<uint32_t>(block);
        scran_markers::score_markers_summary_blocked(*mat, gptr, bptr, opt, buffers);
    } else {
        scran_markers::score_markers_summary(*mat, gptr, opt, buffers);
    }

    auto format = [&](
        const std::vector<pybind11::array_t<double> >& min,
        const std::vector<pybind11::array_t<double> >& mean,
        const std::vector<pybind11::array_t<double> >& median,
        const std::vector<pybind11::array_t<double> >& max,
        const std::vector<pybind11::array_t<uint32_t> >& min_rank
    ) -> pybind11::tuple {
        size_t ngroups = min.size();
        pybind11::tuple output(ngroups);
        for (size_t g = 0; g < ngroups; ++g) {
            pybind11::tuple current(5);
            current[0] = min[g];
            current[1] = mean[g];
            current[2] = median[g];
            current[3] = max[g];
            current[4] = min_rank[g];
            output[g] = current;
        }
        return output;
    };

    pybind11::tuple output(6);
    output[0] = means;
    output[1] = detected;
    output[2] = format(cohens_min, cohens_mean, cohens_median, cohens_max, cohens_mr);
    output[3] = format(auc_min, auc_mean, auc_median, auc_max, auc_mr);
    output[4] = format(dm_min, dm_mean, dm_median, dm_max, dm_mr);
    output[5] = format(dd_min, dd_mean, dd_median, dd_max, dd_mr);
    return output;
}

pybind11::tuple score_markers_pairwise(
    uintptr_t x,
    const pybind11::array& groups,
    size_t num_groups,
    std::optional<pybind11::array> maybe_block,
    std::string block_weight_policy,
    const pybind11::tuple& variable_block_weight,
    double threshold,
    int num_threads,
    bool compute_cohens_d,
    bool compute_auc,
    bool compute_delta_mean,
    bool compute_delta_detected)
{
    const auto& mat = mattress::cast(x)->ptr;
    size_t NC = mat->ncol();
    size_t NR = mat->nrow();
    if (static_cast<size_t>(groups.size()) != NC) {
        throw std::runtime_error("'groups' must have length equal to the number of cells");
    }

    scran_markers::ScoreMarkersPairwiseOptions opt;
    opt.threshold = threshold;
    opt.num_threads = num_threads;
    opt.block_weight_policy = parse_block_weight_policy(block_weight_policy);
    opt.variable_block_weight_parameters = parse_variable_block_weight(variable_block_weight);

    scran_markers::ScoreMarkersPairwiseBuffers<double> buffers;

    pybind11::array_t<double, pybind11::array::f_style> means({ NR, num_groups });
    pybind11::array_t<double, pybind11::array::f_style> detected({ NR, num_groups });
    {
        auto mptr = static_cast<double*>(means.request().ptr);
        auto dptr = static_cast<double*>(detected.request().ptr);
        buffers.mean.reserve(num_groups);
        buffers.detected.reserve(num_groups);
        size_t out_offset = 0;
        for (size_t g = 0; g < num_groups; ++g) {
            buffers.mean.emplace_back(mptr + out_offset);
            buffers.detected.emplace_back(dptr + out_offset);
            out_offset += NR;
        }
    }

    pybind11::array_t<double, pybind11::array::f_style> cohen, auc, delta_mean, delta_detected;
    if (compute_cohens_d) {
        cohen = pybind11::array_t<double, pybind11::array::f_style>({ num_groups, num_groups, NR });
        buffers.cohens_d = static_cast<double*>(cohen.request().ptr);
    }
    if (compute_delta_mean) {
        delta_mean = pybind11::array_t<double, pybind11::array::f_style>({ num_groups, num_groups, NR });
        buffers.delta_mean = static_cast<double*>(delta_mean.request().ptr);
    }
    if (compute_delta_detected) {
        delta_detected = pybind11::array_t<double, pybind11::array::f_style>({ num_groups, num_groups, NR });
        buffers.delta_detected = static_cast<double*>(delta_detected.request().ptr);
    }
    if (compute_auc) {
        auc = pybind11::array_t<double, pybind11::array::f_style>({ num_groups, num_groups, NR });
        buffers.auc = static_cast<double*>(auc.request().ptr);
    }

    auto gptr = check_numpy_array<uint32_t>(groups);
    if (maybe_block.has_value()) {
        const auto& block = *maybe_block;
        if (block.size() != NC) {
            throw std::runtime_error("'block' must be the same length as the number of cells");
        }
        auto bptr = check_numpy_array<uint32_t>(block);
        scran_markers::score_markers_pairwise_blocked(*mat, gptr, bptr, opt, buffers);
    } else {
        scran_markers::score_markers_pairwise(*mat, gptr, opt, buffers);
    }

    pybind11::tuple output(6);
    output[0] = means;
    output[1] = detected;
    output[2] = cohen;
    output[3] = auc;
    output[4] = delta_mean;
    output[5] = delta_detected;
    return output;
}

void init_score_markers(pybind11::module& m) {
    m.def("score_markers_pairwise", &score_markers_pairwise);
    m.def("score_markers_summary", &score_markers_summary);
}
