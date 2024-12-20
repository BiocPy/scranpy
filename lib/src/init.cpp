#include "pybind11/pybind11.h"

void init_adt_quality_control(pybind11::module&);
void init_rna_quality_control(pybind11::module&);
void init_crispr_quality_control(pybind11::module&);
void init_normalize_counts(pybind11::module&);
void init_center_size_factors(pybind11::module&);
void init_sanitize_size_factors(pybind11::module&);
void init_compute_clrm1_factors(pybind11::module&);

PYBIND11_MODULE(lib_scranpy, m) {
    init_adt_quality_control(m);
    init_rna_quality_control(m);
    init_crispr_quality_control(m);
    init_normalize_counts(m);
    init_center_size_factors(m);
    init_sanitize_size_factors(m);
    init_compute_clrm1_factors(m);
}
