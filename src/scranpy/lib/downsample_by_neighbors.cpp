#include "parallel.h" // must be first, to set all macros.

#include "Mattress.h"
#include "scran/aggregation/DownsampleByNeighbors.hpp"
#include <cstdint>
#include <algorithm>

//[[export]]
void downsample_by_neighbors(void* ptr, int32_t* output /** numpy */, int32_t num_threads) {
    auto res = reinterpret_cast<const knncolle::NeighborList<>*>(ptr);
    scran::DownsampleByNeighbors downer;
    downer.set_num_threads(num_threads);
    std::vector<int> tmp_output(res->size()); // not sure if we can assume int = int32_t
    downer.run(*res, tmp_output.data());
    std::copy(tmp_output.begin(), tmp_output.end(), output);
}
