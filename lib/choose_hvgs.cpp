#include "parallel.h" // must be first, to set all macros.

#include "scran/feature_selection/ChooseHvgs.hpp"
#include <cstdint>

//[[export]]
void choose_hvgs(int32_t len, const double* stat /** numpy */, int32_t top, uint8_t* /** numpy */ output) {
    scran::ChooseHvgs runner;
    runner.set_top(top);
    runner.run(len, stat, output);
    return;
}
