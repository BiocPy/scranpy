cmake_minimum_required(VERSION 3.24)

project(scranpy
    VERSION 1.0.0
    DESCRIPTION "Building the scranpy shared library"
    LANGUAGES CXX)

# Defining the targets.
find_package(pybind11 CONFIG)

# pybind11 method:
pybind11_add_module(scranpy
    src/adt_quality_control.cpp
    src/rna_quality_control.cpp
    src/crispr_quality_control.cpp
    src/normalize_counts.cpp
    src/center_size_factors.cpp
    src/sanitize_size_factors.cpp
    src/compute_clrm1_factors.cpp
    src/model_gene_variances.cpp
    src/init.cpp
)

target_include_directories(scranpy PRIVATE "${ASSORTHEAD_INCLUDE_DIR}")
target_include_directories(scranpy PRIVATE "${MATTRESS_INCLUDE_DIR}")
target_include_directories(scranpy PRIVATE "${KNNCOLLE_INCLUDE_DIR}")

set_property(TARGET scranpy PROPERTY CXX_STANDARD 17)

target_link_libraries(scranpy PRIVATE pybind11::pybind11)

set_target_properties(scranpy PROPERTIES
    OUTPUT_NAME lib_scranpy
    PREFIX ""
)
