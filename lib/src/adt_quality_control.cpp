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

pybind11::tuple compute_adt_qc_metrics(uintptr_t x, pybind11::list subsets, int num_threads) {
    const auto& mat = mattress::cast(x)->ptr;
    size_t nc = mat->ncol();
    size_t nr = mat->nrow();

    size_t nsub = subsets.size();
    auto in_subsets = configure_qc_subsets(nr, subsets);

    // Creating output containers.
    scran_qc::ComputeAdtQcMetricsBuffers<double, uint32_t> buffers;
    pybind11::array_t<double> sum(nc);
    buffers.sum = static_cast<double*>(sum.request().ptr);
    pybind11::array_t<uint32_t> detected(nc);
    buffers.detected = static_cast<uint32_t*>(detected.request().ptr);
    pybind11::list out_subsets = prepare_subset_metrics(nc, nsub, buffers.subset_sum);

    // Running QC code.
    scran_qc::ComputeAdtQcMetricsOptions opt;
    opt.num_threads = num_threads;
    scran_qc::compute_adt_qc_metrics(*mat, in_subsets, buffers, opt);

    pybind11::tuple output(3);
    output[0] = sum;
    output[1] = detected;
    output[2] = out_subsets;
    return output;
}

class ConvertedAdtQcMetrics {
public:
    ConvertedAdtQcMetrics(pybind11::tuple metrics) {
        if (metrics.size() != 3) {
            throw std::runtime_error("'metrics' should have the same format as the output of 'compute_adt_qc_metrics'");
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
        scran_qc::ComputeAdtQcMetricsBuffers<const double, const uint32_t> buffers;
        buffers.sum = get_numpy_array_data<double>(sum);
        buffers.detected = get_numpy_array_data<uint32_t>(detected);
        buffers.subset_sum.reserve(subsets.size());
        for (auto& s : subsets) {
            buffers.subset_sum.push_back(get_numpy_array_data<double>(s));
        }
        return buffers;
    }
};

pybind11::tuple suggest_adt_qc_thresholds(pybind11::tuple metrics, std::optional<pybind11::array> maybe_block, double min_detected_drop, double num_mads) {
    ConvertedAdtQcMetrics all_metrics(metrics);
    auto buffers = all_metrics.to_buffer();
    size_t ncells = all_metrics.size();
    size_t nsubs = all_metrics.num_subsets();

    scran_qc::ComputeAdtQcFiltersOptions opt;
    opt.detected_num_mads = num_mads;
    opt.subset_sum_num_mads = num_mads;
    opt.detected_min_drop = min_detected_drop;

    pybind11::tuple output(2);

    if (maybe_block.has_value()) {
        const auto& block = *maybe_block;
        if (block.size() != ncells) {
            throw std::runtime_error("'block' must be the same length as the number of cells");
        }
        auto bptr = check_numpy_array<uint32_t>(block);

        auto filt = scran_qc::compute_adt_qc_filters_blocked(ncells, buffers, bptr, opt);

        const auto& dout = filt.get_detected();
        output[0] = pybind11::array_t<double>(dout.size(), dout.data());
        const auto& ssout = filt.get_subset_sum();
        output[1] = create_subset_filters(ssout);

    } else {
        auto filt = scran_qc::compute_adt_qc_filters(ncells, buffers, opt);
        output[0] = filt.get_detected();
        const auto& ssout = filt.get_subset_sum();
        output[1] = pybind11::array_t<double>(ssout.size(), ssout.data());
    }

    return output;
}

pybind11::array filter_adt_qc_metrics(pybind11::tuple filters, pybind11::tuple metrics, std::optional<pybind11::array> maybe_block) {
    ConvertedAdtQcMetrics all_metrics(metrics);
    auto mbuffers = all_metrics.to_buffer();
    size_t ncells = all_metrics.size();
    size_t nsubs = all_metrics.num_subsets();

    if (filters.size() != 2) {
        throw std::runtime_error("'filters' should have the same format as the output of 'suggest_adt_qc_filters'");
    }

    pybind11::array_t<bool> keep(ncells);
    bool* kptr = static_cast<bool*>(keep.request().ptr);

    if (maybe_block.has_value()) {
        const auto& block = *maybe_block;
        if (block.size() != ncells) {
            throw std::runtime_error("'block' must be the same length as the number of cells");
        }
        auto bptr = check_numpy_array<uint32_t>(block);

        scran_qc::AdtQcBlockedFilters filt;
        const auto& detected = filters[0].cast<pybind11::array>();
        size_t nblocks = detected.size();
        copy_filters_blocked(nblocks, detected, filt.get_detected());
        const auto& subsets = filters[1].cast<pybind11::list>();
        copy_subset_filters_blocked(nsubs, nblocks, subsets, filt.get_subset_sum());

        filt.filter(ncells, mbuffers, bptr, kptr);

    } else {
        scran_qc::AdtQcFilters filt;
        filt.get_detected() = filters[0].cast<double>();
        const auto& subsets = filters[1].cast<pybind11::array>();
        copy_subset_filters_unblocked(nsubs, subsets, filt.get_subset_sum());
        filt.filter(ncells, mbuffers, kptr);
    }

    return keep;
}

void init_adt_quality_control(pybind11::module& m) {
    m.def("compute_adt_qc_metrics", &compute_adt_qc_metrics);
    m.def("suggest_adt_qc_thresholds", &suggest_adt_qc_thresholds);
    m.def("filter_adt_qc_metrics", &filter_adt_qc_metrics);
}
