#include <vector>
#include <stdexcept>
#include <cstdint>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "scran_qc/scran_qc.hpp"
#include "mattress.h"

#include "utils.h"

pybind11::tuple compute_adt_qc_metrics(uintptr_t x, pybind11::list subsets, int num_threads) {
    const auto& mat = mattress::cast(x)->ptr;
    size_t nc = mat->ncol();
    size_t nr = mat->nrow();

    // Setting up the subsets.
    size_t nsub = subsets.size();
    std::vector<const bool*> in_subsets;
    in_subsets.reserve(nsub);
    for (size_t s = 0; s < nsub; ++s) {
        const auto& cursub = subsets[s].cast<pybind11::array>();
        if (nr != static_cast<size_t>(cursub.size())) {
            throw std::runtime_error("each entry of 'subsets' should have the same length as 'x.shape[0]'");
        }
        in_subsets.emplace_back(check_numpy_array<bool>(cursub));
    }

    // Creating output containers.
    scran_qc::ComputeAdtQcMetricsBuffers<double, uint32_t> buffers;
    pybind11::array_t<double> sum(nc);
    buffers.sum = static_cast<double*>(sum.request().ptr);

    pybind11::array_t<uint32_t> detected(nc);
    buffers.detected = static_cast<uint32_t*>(detected.request().ptr);

    pybind11::list out_subsets(nsub);
    for (size_t s = 0; s < nsub; ++s) {
        pybind11::array_t<double> sub(nc);
        buffers.subset_sum.push_back(static_cast<double*>(sub.request().ptr));
        out_subsets[s] = std::move(sub);
    }

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
        size_t nsubs = tmp.size();
        subsets.reserve(nsubs);
        for (size_t s = 0; s < nsubs; ++s) {
            auto cursub = tmp[s].cast<pybind11::array>();
            if (static_cast<size_t>(cursub.size()) != ncells) {
                throw std::runtime_error("all 'metrics' vectors should have the same length");
            }
            check_numpy_array<double>(cursub);
            subsets.emplace_back(std::move(cursub));
        }
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
        pybind11::list subs(nsubs);
        for (size_t s = 0; s < nsubs; ++s) {
            const auto& cursub = ssout[s];
            subs[s] = pybind11::array_t<double>(cursub.size(), cursub.data());
        }
        output[1] = subs;

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

    if (maybe_block.has_value()) {
        const auto& block = *maybe_block;
        if (block.size() != ncells) {
            throw std::runtime_error("'block' must be the same length as the number of cells");
        }
        auto bptr = check_numpy_array<uint32_t>(block);

        scran_qc::AdtQcBlockedFilters filt;
        const auto& detected = filters[0].cast<pybind11::array>();
        size_t nblocks = detected.size();
        auto& df = filt.get_detected();
        auto dptr = check_numpy_array<double>(detected);
        df.insert(df.end(), dptr, dptr + nblocks);

        const auto& subsets = filters[1].cast<pybind11::list>();
        if (static_cast<size_t>(subsets.size()) != nsubs) {
            throw std::runtime_error("'filters.subsets' should have the same length as the number of subsets in 'metrics'");
        }
        auto& ssf = filt.get_subset_sum();
        ssf.reserve(nsubs);
        for (size_t s = 0; s < nsubs; ++s) {
            const auto& cursub = subsets[s].cast<pybind11::array>();
            if (static_cast<size_t>(cursub.size()) != nblocks) {
                throw std::runtime_error("each entry of 'filters.subsets' should have the same length as 'filters.detected'");
            }
            auto ptr = check_numpy_array<double>(cursub);
            ssf.emplace_back(ptr, ptr + nblocks);
        }

        for (size_t c = 0; c < ncells; ++c) {
            if (bptr[c] >= nblocks) {
                throw std::runtime_error("'block' contains out-of-range indices");
            }
        }
        filt.filter(ncells, mbuffers, bptr, static_cast<bool*>(keep.request().ptr));

    } else {
        scran_qc::AdtQcFilters filt;
        filt.get_detected() = filters[0].cast<double>();

        const auto& subsets = filters[1].cast<pybind11::array>();
        if (static_cast<size_t>(subsets.size()) != nsubs) {
            throw std::runtime_error("'filters.subsets' should have the same length as the number of subsets in 'metrics'");
        }
        auto subptr = check_numpy_array<double>(subsets);
        auto& ssf = filt.get_subset_sum();
        ssf.insert(ssf.end(), subptr, subptr + nsubs);

        filt.filter(ncells, mbuffers, static_cast<bool*>(keep.request().ptr));
    }

    return keep;
}

void init_adt_quality_control(pybind11::module& m) {
    m.def("compute_adt_qc_metrics", &compute_adt_qc_metrics);
    m.def("suggest_adt_qc_thresholds", &suggest_adt_qc_thresholds);
    m.def("filter_adt_qc_metrics", &filter_adt_qc_metrics);
}
