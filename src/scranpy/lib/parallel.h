#ifndef PARALLEL_H
#define PARALLEL_H

#include "tatami/tatami.hpp"
#include <thread>
#include <cmath>
#include <vector>
#include <stdexcept>

template<typename Index_, class Function_>
void run_parallel_old(Index_ njobs, Function_ fun, size_t nthreads) {
    if (nthreads == 1) {
        fun(0, njobs);
        return;
    }

    Index_ jobs_per_worker = njobs/nthreads + (njobs % nthreads > 0);
    std::vector<std::thread> workers;
    workers.reserve(nthreads);
    std::vector<std::string> errors(nthreads);

    Index_ first = 0;
    for (int w = 0; w < nthreads && first < njobs; ++w) {
        int last = first + std::min(jobs_per_worker, njobs - first);
        workers.emplace_back([&fun,&errors](int w, Index_ first, Index_ last) -> void {
            try {
                fun(first, last);
            } catch (std::exception& e) {
                errors[w] = e.what();
            } catch (...) {
                errors[w] = "unknown C++ error";
            }
        }, w, first, last);
        first = last;
    }

    for (auto& wrk : workers) {
        wrk.join();
    }

    for (const auto& e : errors) {
        if (!e.empty()) {
            throw std::runtime_error(e);
        }
    }
}

template<class Function_>
void run_parallel_simple(int nthreads, Function_ fun) {
    if (nthreads == 1) {
        fun(0);
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(nthreads);
    std::vector<std::string> errors(nthreads);

    for (int w = 0; w < nthreads; ++w) {
        workers.emplace_back([&fun,&errors](int w) -> void {
            try {
                fun(w);
            } catch (std::exception& e) {
                errors[w] = e.what();
            } catch (...) {
                errors[w] = "unknown C++ error";
            }
        }, w);
    }

    for (auto& wrk : workers) {
        wrk.join();
    }

    for (const auto& e : errors) {
        if (!e.empty()) {
            throw std::runtime_error(e);
        }
    }
}

/*
 * All of these macros are explicitly defined for safety's sake.
 *
 * In theory, this would not be necessary as macros for libraries like scran
 * propagate to scran's dependencies like tatami, irlba, and kmeans.
 * Unfortunately, this assumes that scran is included in each compilation unit,
 * which is not currently the case. This means that certain compilation units
 * may have propagation and others may not, resulting in ODR errors.
 *
 * You can figure out which macros need to be defined by checking the
 * dependencies in build_main/_deps; many of Aaron's libraries will support
 * some form of *_CUSTOM_PARALLEL macro.
 */

#define IRLBA_CUSTOM_PARALLEL run_parallel_simple
#define KNNCOLLE_CUSTOM_PARALLEL run_parallel_old
#define KMEANS_CUSTOM_PARALLEL run_parallel_old
#define SCRAN_CUSTOM_PARALLEL tatami::parallelize
#define MNNCORRECT_CUSTOM_PARALLEL run_parallel_old
#define QDTSNE_CUSTOM_PARALLEL run_parallel_old
#define UMAPPP_CUSTOM_PARALLEL run_parallel_old

#endif
