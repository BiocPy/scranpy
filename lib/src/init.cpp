#include "pybind11/pybind11.h"

void init_adt_quality_control(pybind11::module&);

PYBIND11_MODULE(lib_scranpy, m) {
    init_adt_quality_control(m);
}
