#include <vector>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "scran_variances/scran_variances.hpp"

#include "utils.h"

pybind11::array choose_highly_variable_genes(const pybind11::array& stats, int top, bool larger, bool keep_ties, std::optional<double> bound) {
    scran_variances::ChooseHighlyVariableGenesOptions opt;
    opt.top = top;
    opt.larger = larger;
    opt.keep_ties = keep_ties;

    opt.use_bound = bound.has_value();
    if (opt.use_bound) {
        opt.bound = *bound;
    }

    auto res = scran_variances::choose_highly_variable_genes_index<uint32_t>(stats.size(), check_numpy_array<double>(stats), opt);
    return pybind11::array_t<uint32_t>(res.size(), res.data());
}

void init_choose_highly_variable_genes(pybind11::module& m) {
    m.def("choose_highly_variable_genes", &choose_highly_variable_genes);
}
