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

pybind11::tuple compute_crispr_qc_metrics(uintptr_t x, int num_threads) {
    const auto& mat = mattress::cast(x)->ptr;
    size_t nc = mat->ncol();
    size_t nr = mat->nrow();

    // Creating output containers.
    scran_qc::ComputeCrisprQcMetricsBuffers<double, uint32_t, double, uint32_t> buffers;
    pybind11::array_t<double> sum(nc);
    buffers.sum = static_cast<double*>(sum.request().ptr);
    pybind11::array_t<uint32_t> detected(nc);
    buffers.detected = static_cast<uint32_t*>(detected.request().ptr);
    pybind11::array_t<double> max_value(nc);
    buffers.max_value = static_cast<double*>(max_value.request().ptr);
    pybind11::array_t<uint32_t> max_index(nc);
    buffers.max_index = static_cast<uint32_t*>(max_index.request().ptr);

    // Running QC code.
    scran_qc::ComputeCrisprQcMetricsOptions opt;
    opt.num_threads = num_threads;
    scran_qc::compute_crispr_qc_metrics(*mat, buffers, opt);

    pybind11::tuple output(4);
    output[0] = sum;
    output[1] = detected;
    output[2] = max_value;
    output[3] = max_index;
    return output;
}

class ConvertedCrisprQcMetrics {
public:
    ConvertedCrisprQcMetrics(pybind11::tuple metrics) {
        if (metrics.size() != 4) {
            throw std::runtime_error("'metrics' should have the same format as the output of 'compute_crispr_qc_metrics'");
        }

        sum = metrics[0].cast<pybind11::array>();
        check_numpy_array<double>(sum);
        size_t ncells = sum.size();

        detected = metrics[1].cast<pybind11::array>();
        check_numpy_array<uint32_t>(detected);
        if (ncells != static_cast<size_t>(detected.size())) {
            throw std::runtime_error("all 'metrics' vectors should have the same length");
        }

        max_value = metrics[2].cast<pybind11::array>();
        check_numpy_array<double>(max_value);
        if (ncells != static_cast<size_t>(max_value.size())) {
            throw std::runtime_error("all 'metrics' vectors should have the same length");
        }

        max_index = metrics[3].cast<pybind11::array>();
        check_numpy_array<uint32_t>(max_index);
        if (ncells != static_cast<size_t>(max_index.size())) {
            throw std::runtime_error("all 'metrics' vectors should have the same length");
        }
    }

private:
    pybind11::array sum;
    pybind11::array detected;
    pybind11::array max_value;
    pybind11::array max_index;

public:
    size_t size() const {
        return sum.size();
    }

    auto to_buffer() const {
        scran_qc::ComputeCrisprQcMetricsBuffers<const double, const uint32_t, const double, const uint32_t> buffers;
        buffers.sum = get_numpy_array_data<double>(sum);
        buffers.detected = get_numpy_array_data<uint32_t>(detected);
        buffers.max_value = get_numpy_array_data<double>(max_value);
        buffers.max_index = get_numpy_array_data<uint32_t>(max_index);
        return buffers;
    }
};

pybind11::tuple suggest_crispr_qc_thresholds(pybind11::tuple metrics, std::optional<pybind11::array> maybe_block, double num_mads) {
    ConvertedCrisprQcMetrics all_metrics(metrics);
    auto buffers = all_metrics.to_buffer();
    size_t ncells = all_metrics.size();

    scran_qc::ComputeCrisprQcFiltersOptions opt;
    opt.max_value_num_mads = num_mads;

    pybind11::tuple output(1);

    if (maybe_block.has_value()) {
        const auto& block = *maybe_block;
        if (block.size() != ncells) {
            throw std::runtime_error("'block' must be the same length as the number of cells");
        }
        auto bptr = check_numpy_array<uint32_t>(block);

        auto filt = scran_qc::compute_crispr_qc_filters_blocked(ncells, buffers, bptr, opt);
        const auto& mout = filt.get_max_value();
        output[0] = pybind11::array_t<double>(mout.size(), mout.data());

    } else {
        auto filt = scran_qc::compute_crispr_qc_filters(ncells, buffers, opt);
        output[0] = filt.get_max_value();
    }

    return output;
}

pybind11::array filter_crispr_qc_metrics(pybind11::tuple filters, pybind11::tuple metrics, std::optional<pybind11::array> maybe_block) {
    ConvertedCrisprQcMetrics all_metrics(metrics);
    auto mbuffers = all_metrics.to_buffer();
    size_t ncells = all_metrics.size();

    if (filters.size() != 1) {
        throw std::runtime_error("'filters' should have the same format as the output of 'suggest_crispr_qc_thresholds'");
    }

    pybind11::array_t<bool> keep(ncells);
    bool* kptr = static_cast<bool*>(keep.request().ptr);

    if (maybe_block.has_value()) {
        const auto& block = *maybe_block;
        if (block.size() != ncells) {
            throw std::runtime_error("'block' must be the same length as the number of cells");
        }
        auto bptr = check_numpy_array<uint32_t>(block);

        scran_qc::CrisprQcBlockedFilters filt;
        auto max_value = filters[0].cast<pybind11::array>();
        size_t nblocks = max_value.size();
        copy_filters_blocked(nblocks, max_value, filt.get_max_value());

        filt.filter(ncells, mbuffers, bptr, kptr);

    } else {
        scran_qc::CrisprQcFilters filt;
        filt.get_max_value() = filters[0].cast<double>();
        filt.filter(ncells, mbuffers, kptr);
    }

    return keep;
}

void init_crispr_quality_control(pybind11::module& m) {
    m.def("compute_crispr_qc_metrics", &compute_crispr_qc_metrics);
    m.def("suggest_crispr_qc_thresholds", &suggest_crispr_qc_thresholds);
    m.def("filter_crispr_qc_metrics", &filter_crispr_qc_metrics);
}
