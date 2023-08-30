#include "parallel.h" // must be first, to set all macros.

#include "scran/normalization/CenterSizeFactors.hpp"
#include "scran/normalization/SanitizeSizeFactors.hpp"
#include "Mattress.h"

#include <cstdint>
#include <memory>

//[[export]]
void center_size_factors(
    int32_t num,
    double* size_factors /** numpy */,
    uint8_t allow_zeros,
    uint8_t allow_non_finite,
    uint8_t use_block,
    const int32_t* block /** void_p */)
{
    if (!use_block) {
        block = NULL;
    }

    scran::CenterSizeFactors runner;
    auto res = runner.run_blocked(num, size_factors, block);

    scran::SanitizeSizeFactors san;
    san.set_handle_zero(allow_zeros ? scran::SanitizeSizeFactors::HandlerAction::SANITIZE : scran::SanitizeSizeFactors::HandlerAction::ERROR);
    san.set_handle_non_finite(allow_non_finite ? scran::SanitizeSizeFactors::HandlerAction::SANITIZE : scran::SanitizeSizeFactors::HandlerAction::ERROR);
    san.run(num, size_factors, res);

    return;
}
