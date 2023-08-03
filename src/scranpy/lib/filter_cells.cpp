#include "parallel.h" // must be first, to set all macros.

#include "Mattress.h"
#include "scran/quality_control/FilterCells.hpp"
#include <cstdint>

//[[export]]
void* filter_cells(const void* mat0, const uint8_t* filter /** numpy */, uint8_t discard) {
    scran::FilterCells runner;
    if (discard) {
        runner.set_discard();
    } else {
        runner.set_retain();
    }
    auto mat = reinterpret_cast<const Mattress*>(mat0);
    return new Mattress(runner.run(mat->ptr, filter));
}
