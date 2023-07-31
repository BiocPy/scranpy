#include "Mattress.h"
#include "scran/differential_analysis/ScoreMarkers.hpp"
#include <vector>

extern "C" {

void score_markers(
    const Mattress* mat, 
    int num_clusters, 
    const int32_t* clusters, 
    int num_blocks,
    const int32_t* block,
    uint8_t do_auc,
    double threshold,
    uintptr_t* raw_means,
    uintptr_t* raw_detected,
    uintptr_t* raw_cohen,
    uintptr_t* raw_auc,
    uintptr_t* raw_lfc,
    uintptr_t* raw_delta_detected,
    int num_threads)
{
    std::vector<double*> means;
    means.reserve(num_clusters);
    std::vector<double*> detected;
    detected.reserve(num_clusters);
    for (int c = 0; c < num_clusters; ++c) {
        means.push_back(reinterpret_cast<double*>(raw_means[c]));
        detected.push_back(reinterpret_cast<double*>(raw_detected[c]));
    }

    std::vector<std::vector<double*> > cohen(scran::differential_analysis::summary::n_summaries);
    auto lfc = cohen;
    auto delta_detected = cohen;
    std::vector<std::vector<double*> > auc(do_auc ? scran::differential_analysis::summary::n_summaries : 0);

    for (int s = 0; s < 3; ++s) {
        scran::differential_analysis::summary which_summary;
        if (s == 0) {
            which_summary = scran::differential_analysis::MIN;
        } else if (s == 1) {
            which_summary = scran::differential_analysis::MEAN;
        } else if (s == 2) {
            which_summary = scran::differential_analysis::MIN_RANK;
        }

        auto cptr = reinterpret_cast<uintptr_t*>(raw_cohen[s]);
        auto lptr = reinterpret_cast<uintptr_t*>(raw_lfc[s]);
        auto dptr = reinterpret_cast<uintptr_t*>(raw_delta_detected[s]);
        auto aptr = (do_auc ? reinterpret_cast<uintptr_t*>(raw_auc[s]) : NULL);

        cohen[which_summary].resize(num_clusters, NULL);
        lfc[which_summary].resize(num_clusters, NULL);
        delta_detected[which_summary].resize(num_clusters, NULL);
        if (do_auc) {
            auc[which_summary].resize(num_clusters, NULL);
        }

        for (int c = 0; c < num_clusters; ++c) {
            cohen[which_summary][c] = reinterpret_cast<double*>(cptr[c]);
            lfc[which_summary][c] = reinterpret_cast<double*>(lptr[c]);
            delta_detected[which_summary][c] = reinterpret_cast<double*>(dptr[c]);
            if (do_auc) {
                auc[which_summary][c] = reinterpret_cast<double*>(aptr[c]);
            }
        }
    }

    scran::ScoreMarkers runner;
    runner.set_threshold(threshold);
    runner.set_num_threads(num_threads);

    // TODO: check this in run_blocked.
    if (num_blocks > 1) {
        runner.run_blocked(
            mat->ptr.get(), 
            clusters, 
            block,
            std::move(means), 
            std::move(detected), 
            std::move(cohen), 
            std::move(auc), 
            std::move(lfc), 
            std::move(delta_detected)
        );
    } else {
        runner.run(
            mat->ptr.get(), 
            clusters, 
            std::move(means), 
            std::move(detected), 
            std::move(cohen), 
            std::move(auc), 
            std::move(lfc), 
            std::move(delta_detected)
        );
    }

    return;
}

}
