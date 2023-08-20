#include "parallel.h" // must be first, to set all macros.

#include "scran/normalization/CenterSizeFactors.hpp"
#include "Mattress.h"

#include <cstdint>
#include <memory>

//[[export]]
void center_size_factors(
    int32_t num,
    double* size_factors /** numpy */, 
    uint8_t use_block, 
    const int32_t* block /** void_p */)
{
    scran::CenterSizeFactors runner;
    if (!use_block) {
        block = NULL;
    }
    runner.run_blocked(num, size_factors, block);
    return; 
}
