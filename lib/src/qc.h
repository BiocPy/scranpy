#ifndef UTILS_QC_H
#define UTILS_QC_H

#include <vector>
#include <cstdint>
#include <string>
#include <stdexcept>

#include "pybind11/pybind11.h"

#include "utils.h"

inline std::vector<const bool*> configure_qc_subsets(size_t ngenes, const pybind11::list& subsets) {
    size_t nsub = subsets.size();
    std::vector<const bool*> in_subsets;
    in_subsets.reserve(nsub);
    for (size_t s = 0; s < nsub; ++s) {
        const auto& cursub = subsets[s].cast<pybind11::array>();
        if (ngenes != static_cast<size_t>(cursub.size())) {
            throw std::runtime_error("each entry of 'subsets' should have the same length as 'x.shape[0]'");
        }
        in_subsets.emplace_back(check_numpy_array<bool>(cursub));
    }
    return in_subsets;
}

inline pybind11::list prepare_subset_metrics(size_t ncells, size_t nsub, std::vector<double*>& ptrs) {
    pybind11::list out_subsets(nsub);
    ptrs.reserve(nsub);
    for (size_t s = 0; s < nsub; ++s) {
        pybind11::array_t<double> sub(ncells);
        ptrs.push_back(static_cast<double*>(sub.request().ptr));
        out_subsets[s] = std::move(sub);
    }
    return out_subsets;
}

inline void check_subset_metrics(size_t ncells, const pybind11::list& input, std::vector<pybind11::array>& store) {
    size_t nsubs = input.size();
    store.reserve(nsubs);
    for (size_t s = 0; s < nsubs; ++s) {
        auto cursub = input[s].cast<pybind11::array>();
        if (static_cast<size_t>(cursub.size()) != ncells) {
            throw std::runtime_error("all 'metrics' vectors should have the same length");
        }
        check_numpy_array<double>(cursub);
        store.emplace_back(std::move(cursub));
    }
}

inline pybind11::list create_subset_filters(const std::vector<std::vector<double> >& input) {
    size_t nsubs = input.size();
    pybind11::list subs(nsubs);
    for (size_t s = 0; s < nsubs; ++s) {
        const auto& cursub = input[s];
        subs[s] = pybind11::array_t<double>(cursub.size(), cursub.data());
    }
    return subs;
}

inline void copy_filters_blocked(size_t nblocks, const pybind11::array& input, std::vector<double>& store) {
    if (static_cast<size_t>(input.size()) != nblocks) {
        throw std::runtime_error("each array of thresholds in 'filters' should have length equal to the number of blocks");
    }
    auto ptr = check_numpy_array<double>(input);
    store.insert(store.end(), ptr, ptr + nblocks);
}

inline void copy_subset_filters_blocked(size_t nsubs, size_t nblocks, const pybind11::list& subsets, std::vector<std::vector<double> >& store) {
    if (static_cast<size_t>(subsets.size()) != nsubs) {
        throw std::runtime_error("'filters.subset_*' should have the same length as the number of subsets in 'metrics'");
    }
    store.resize(nsubs);
    for (size_t s = 0; s < nsubs; ++s) {
        const auto& cursub = subsets[s].cast<pybind11::array>();
        copy_filters_blocked(nblocks, cursub, store[s]);
    }
}

inline void copy_subset_filters_unblocked(size_t nsubs, const pybind11::array& subsets, std::vector<double>& store) {
    if (static_cast<size_t>(subsets.size()) != nsubs) {
        throw std::runtime_error("'filters.subset_*' should have the same length as the number of subsets in 'metrics'");
    }
    auto subptr = check_numpy_array<double>(subsets);
    store.insert(store.end(), subptr, subptr + nsubs);
}

#endif
