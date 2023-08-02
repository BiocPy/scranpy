#include "parallel.h" // must be first, to set all macros.

#include "Mattress.h"
#include "scran/quality_control/FilterCells.hpp"

//[[export]]
void* filter_cells(const void* mat0, const bool* filter, bool discard) {
    scran::FilterCells runner;
    if (discard) {
        runner.set_discard();
    } else {
        runner.set_retain();
    }

    auto mat = reinterpret_cast<const Mattress*>(mat0);
    return new Mattress(runner.run(mat->ptr, filter));
}
