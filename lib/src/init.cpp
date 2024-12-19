#include "pybind11/pybind11.h"

void init_adt_quality_control(pybind11::module&);
void init_rna_quality_control(pybind11::module&);
void init_crispr_quality_control(pybind11::module&);

PYBIND11_MODULE(lib_scranpy, m) {
    init_adt_quality_control(m);
    init_rna_quality_control(m);
    init_crispr_quality_control(m);
}
