#include <vector>
#include <stdexcept>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "scran_variances/scran_variances.hpp"

#include "utils.h"

pybind11::tuple fit_variance_trend(
    const pybind11::array& means,
    const pybind11::array& variances,
    bool mean_filter,
    double min_mean,
    bool transform,
    double span,
    bool use_min_width,
    double min_width,
    int min_window_count,
    int num_threads)
{
    scran_variances::FitVarianceTrendOptions opt;
    opt.mean_filter = mean_filter;
    opt.minimum_mean = min_mean;
    opt.transform = transform;
    opt.span = span;
    opt.use_minimum_width = use_min_width;
    opt.minimum_width = min_width;
    opt.minimum_window_count = min_window_count;
    opt.num_threads = num_threads;

    size_t n = means.size();
    if (n != static_cast<size_t>(variances.size())) {
        throw std::runtime_error("'means' and 'variances' should have the same length");
    }

    pybind11::array_t<double> fitted(n), residuals(n);
    scran_variances::FitVarianceTrendWorkspace<double> wrk; 
    scran_variances::fit_variance_trend(
        n,
        check_numpy_array<double>(means),
        check_numpy_array<double>(variances),
        static_cast<double*>(fitted.request().ptr),
        static_cast<double*>(residuals.request().ptr),
        wrk,
        opt
    );

    pybind11::tuple output(2);
    output[0] = fitted;
    output[1] = residuals;
    return output;
}

void init_fit_variance_trend(pybind11::module& m) {
    m.def("fit_variance_trend", &fit_variance_trend);
}
