#include <vector>
#include <stdexcept>
#include <cstdint>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "scran_qc/scran_qc.hpp"
#include "mattress.h"

#include "utils.h"
#include "qc.h"

pybind11::tuple compute_rna_qc_metrics(uintptr_t x, pybind11::list subsets, int num_threads) {
    const auto& mat = mattress::cast(x)->ptr;
    size_t nc = mat->ncol();
    size_t nr = mat->nrow();

    size_t nsub = subsets.size();
    auto in_subsets = configure_qc_subsets(nr, subsets);

    // Creating output containers.
    scran_qc::ComputeRnaQcMetricsBuffers<double, uint32_t> buffers;
    pybind11::array_t<double> sum(nc);
    buffers.sum = static_cast<double*>(sum.request().ptr);
    pybind11::array_t<uint32_t> detected(nc);
    buffers.detected = static_cast<uint32_t*>(detected.request().ptr);
    pybind11::list out_subsets = prepare_subset_metrics(nc, nsub, buffers.subset_proportion);

    // Running QC code.
    scran_qc::ComputeRnaQcMetricsOptions opt;
    opt.num_threads = num_threads;
    scran_qc::compute_rna_qc_metrics(*mat, in_subsets, buffers, opt);

    pybind11::tuple output(3);
    output[0] = sum;
    output[1] = detected;
    output[2] = out_subsets;
    return output;
}

class ConvertedRnaQcMetrics {
public:
    ConvertedRnaQcMetrics(pybind11::tuple metrics) {
        if (metrics.size() != 3) {
            throw std::runtime_error("'metrics' should have the same format as the output of 'compute_rna_qc_metrics'");
        }

        sum = metrics[0].cast<pybind11::array>();
        check_numpy_array<double>(sum);
        size_t ncells = sum.size();

        detected = metrics[1].cast<pybind11::array>();
        check_numpy_array<uint32_t>(detected);
        if (ncells != static_cast<size_t>(detected.size())) {
            throw std::runtime_error("all 'metrics' vectors should have the same length");
        }

        auto tmp = metrics[2].cast<pybind11::list>();
        check_subset_metrics(ncells, tmp, subsets);
    }

private:
    pybind11::array sum;
    pybind11::array detected;
    std::vector<pybind11::array> subsets;

public:
    size_t size() const {
        return sum.size();
    }

    size_t num_subsets() const {
        return subsets.size();
    }

    auto to_buffer() const {
        scran_qc::ComputeRnaQcMetricsBuffers<const double, const uint32_t, const double> buffers;
        buffers.sum = get_numpy_array_data<double>(sum);
        buffers.detected = get_numpy_array_data<uint32_t>(detected);
        buffers.subset_proportion.reserve(subsets.size());
        for (auto& s : subsets) {
            buffers.subset_proportion.push_back(get_numpy_array_data<double>(s));
        }
        return buffers;
    }
};

pybind11::tuple suggest_rna_qc_thresholds(pybind11::tuple metrics, std::optional<pybind11::array> maybe_block, double num_mads) {
    ConvertedRnaQcMetrics all_metrics(metrics);
    auto buffers = all_metrics.to_buffer();
    size_t ncells = all_metrics.size();

    scran_qc::ComputeRnaQcFiltersOptions opt;
    opt.sum_num_mads = num_mads;
    opt.detected_num_mads = num_mads;
    opt.subset_proportion_num_mads = num_mads;

    pybind11::tuple output(3);

    if (maybe_block.has_value()) {
        const auto& block = *maybe_block;
        if (block.size() != ncells) {
            throw std::runtime_error("'block' must be the same length as the number of cells");
        }
        auto bptr = check_numpy_array<uint32_t>(block);

        auto filt = scran_qc::compute_rna_qc_filters_blocked(ncells, buffers, bptr, opt);

        const auto& sout = filt.get_sum();
        output[0] = pybind11::array_t<double>(sout.size(), sout.data());
        const auto& dout = filt.get_detected();
        output[1] = pybind11::array_t<double>(dout.size(), dout.data());
        const auto& ssout = filt.get_subset_proportion();
        output[2] = create_subset_filters(ssout);

    } else {
        auto filt = scran_qc::compute_rna_qc_filters(ncells, buffers, opt);
        output[0] = filt.get_sum();
        output[1] = filt.get_detected();
        const auto& ssout = filt.get_subset_proportion();
        output[2] = pybind11::array_t<double>(ssout.size(), ssout.data());
    }

    return output;
}

pybind11::array filter_rna_qc_metrics(pybind11::tuple filters, pybind11::tuple metrics, std::optional<pybind11::array> maybe_block) {
    ConvertedRnaQcMetrics all_metrics(metrics);
    auto mbuffers = all_metrics.to_buffer();
    size_t ncells = all_metrics.size();
    size_t nsubs = all_metrics.num_subsets();

    if (filters.size() != 3) {
        throw std::runtime_error("'filters' should have the same format as the output of 'suggestRnaQcFilters'");
    }

    pybind11::array_t<bool> keep(ncells);
    bool* kptr = static_cast<bool*>(keep.request().ptr);

    if (maybe_block.has_value()) {
        const auto& block = *maybe_block;
        if (block.size() != ncells) {
            throw std::runtime_error("'block' must be the same length as the number of cells");
        }
        auto bptr = check_numpy_array<uint32_t>(block);

        scran_qc::RnaQcBlockedFilters filt;
        const auto& sum = filters[0].cast<pybind11::array>();
        size_t nblocks = sum.size();
        copy_filters_blocked(nblocks, sum, filt.get_sum());
        const auto& detected = filters[1].cast<pybind11::array>();
        copy_filters_blocked(nblocks, detected, filt.get_detected());
        const auto& subsets = filters[2].cast<pybind11::list>();
        copy_subset_filters_blocked(nsubs, nblocks, subsets, filt.get_subset_proportion());

        filt.filter(ncells, mbuffers, bptr, kptr);

    } else {
        scran_qc::RnaQcFilters filt;
        filt.get_sum() = filters[0].cast<double>();
        filt.get_detected() = filters[1].cast<double>();
        const auto& subsets = filters[2].cast<pybind11::array>();
        copy_subset_filters_unblocked(nsubs, subsets, filt.get_subset_proportion());
        filt.filter(ncells, mbuffers, kptr);
    }

    return keep;
}

void init_rna_quality_control(pybind11::module& m) {
    m.def("compute_rna_qc_metrics", &compute_rna_qc_metrics);
    m.def("suggest_rna_qc_thresholds", &suggest_rna_qc_thresholds);
    m.def("filter_rna_qc_metrics", &filter_rna_qc_metrics);
}
