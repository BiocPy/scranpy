#include "parallel.h" // must be first, to set all macros.

#include "Mattress.h"
#include "scran/feature_set_enrichment/ScoreFeatureSet.hpp"
#include <cstdint>
#include <algorithm>

//[[export]]
void score_feature_set(
    void* mat,
    const uint8_t* features /** numpy */,
    uint8_t use_block,
    const int32_t* block /** void_p */,
    double* output_scores /** numpy */,
    double* output_weights /** numpy */,
    uint8_t scale,
    int32_t nthreads)
{
    const auto& ptr = reinterpret_cast<const Mattress*>(mat)->ptr;
    if (!use_block) {
        block = NULL;
    }

    scran::ScoreFeatureSet runner;
    runner.set_num_threads(nthreads);
    runner.set_scale(scale);
    auto res = runner.run_blocked(ptr.get(), features, block);

    std::copy(res.scores.begin(), res.scores.end(), output_scores);
    std::copy(res.weights.begin(), res.weights.end(), output_weights);
}
