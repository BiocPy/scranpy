#include "parallel.h"
#include "Mattress.h"
#include "mnncorrect/MnnCorrect.hpp"
#include <cstdint>
#include <cstring>
#include <algorithm>

//[[export]]
void mnn_correct(
    int32_t ndim,
    int32_t ncells,
    const double* x /** numpy */,
    int32_t nbatches,
    const int32_t* batch /** numpy */,
    int32_t k,
    double nmads,
    int32_t nthreads,
    int32_t mass_cap,
    uint8_t use_order,
    const int32_t* order /** void_p */,
    const char* ref_policy,
    uint8_t approximate,
    double* corrected_output /** numpy */,
    int32_t* merge_order_output /** numpy */,
    int32_t* num_pairs_output /** numpy */)
{
    mnncorrect::MnnCorrect<> runner;
    runner
        .set_approximate(approximate)
        .set_num_neighbors(k)
        .set_num_mads(nmads)
        .set_mass_cap(mass_cap)
        .set_num_threads(nthreads);

    std::vector<int> ordering;
    const int* optr = NULL;
    if (use_order) {
        ordering.insert(ordering.end(), order, order + nbatches);
        optr = ordering.data();
    }

    if (std::strcmp(ref_policy, "input") == 0) {
        runner.set_reference_policy(mnncorrect::Input);
    } else if (std::strcmp(ref_policy, "max-variance") == 0) {
        runner.set_reference_policy(mnncorrect::MaxVariance);
    } else if (std::strcmp(ref_policy, "max-rss") == 0) {
        runner.set_reference_policy(mnncorrect::MaxRss);
    } else if (std::strcmp(ref_policy, "max-size") == 0) {
        runner.set_reference_policy(mnncorrect::MaxSize);
    } else {
        throw std::runtime_error("unknown reference policy");
    }

    auto res = runner.run(ndim, ncells, x, batch, corrected_output, optr);
    std::copy(res.merge_order.begin(), res.merge_order.end(), merge_order_output);
    std::copy(res.num_pairs.begin(), res.num_pairs.end(), num_pairs_output);
    return;
}
