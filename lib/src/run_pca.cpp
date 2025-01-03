#include <vector>
#include <algorithm>
#include <string>
#include <stdexcept>
#include <cstdint>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "scran_pca/scran_pca.hpp"
#include "mattress.h"

#include "utils.h"
#include "block.h"

static pybind11::array transfer(const Eigen::MatrixXd& x) {
    pybind11::array_t<double, pybind11::array::f_style> output({ x.rows(), x.cols() });
    static_assert(!Eigen::MatrixXd::IsRowMajor);
    std::copy_n(x.data(), output.size(), static_cast<double*>(output.request().ptr));
    return output;
}

static pybind11::array transfer(const Eigen::VectorXd& x) {
    return pybind11::array_t<double>(x.size(), x.data());
}

pybind11::tuple run_pca(
    uintptr_t x,
    int number,
    std::optional<pybind11::array> maybe_block, 
    std::string block_weight_policy,
    const pybind11::tuple& variable_block_weight,
    bool components_from_residuals,
    bool scale,
    bool realized,
    int irlba_work,
    int irlba_iterations,
    int irlba_seed,
    int num_threads)
{
    const auto& mat = mattress::cast(x)->ptr;

    irlba::Options iopt;
    iopt.extra_work = irlba_work;
    iopt.max_iterations = irlba_iterations;
    iopt.seed = irlba_seed;
    iopt.cap_number = true;

    if (maybe_block.has_value()) {
        const auto& block = *maybe_block;
        if (block.size() != static_cast<size_t>(mat->ncol())) {
            throw std::runtime_error("'block' must be the same length as the number of cells");
        }
        const auto* bptr = check_numpy_array<uint32_t>(block);

        scran_pca::BlockedPcaOptions opt;
        opt.number = number;
        opt.scale = scale;
        opt.block_weight_policy = parse_block_weight_policy(block_weight_policy);
        opt.variable_block_weight_parameters = parse_variable_block_weight(variable_block_weight);
        opt.components_from_residuals = components_from_residuals;
        opt.realize_matrix = realized;
        opt.irlba_options = iopt;
        opt.num_threads = num_threads;

        auto out = scran_pca::blocked_pca(*mat, bptr, opt);

        pybind11::tuple output(6);
        output[0] = transfer(out.components);
        output[1] = transfer(out.rotation);
        output[2] = transfer(out.variance_explained);
        output[3] = out.total_variance;
        output[4] = transfer(out.center);
        output[5] = transfer(out.scale);
        return output;

    } else {
        scran_pca::SimplePcaOptions opt;
        opt.number = number;
        opt.scale = scale;
        opt.realize_matrix = realized;
        opt.irlba_options = iopt;
        opt.num_threads = num_threads;

        auto out = scran_pca::simple_pca(*mat, opt);

        pybind11::tuple output(6);
        output[0] = transfer(out.components);
        output[1] = transfer(out.rotation);
        output[2] = transfer(out.variance_explained);
        output[3] = out.total_variance;
        output[4] = transfer(out.center);
        output[5] = transfer(out.scale);
        return output;
    }
}

void init_run_pca(pybind11::module& m) {
    m.def("run_pca", &run_pca);
}
