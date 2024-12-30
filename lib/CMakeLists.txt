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
    src/choose_pseudo_count.cpp
    src/model_gene_variances.cpp
    src/fit_variance_trend.cpp
    src/choose_highly_variable_genes.cpp
    src/run_pca.cpp
    src/run_tsne.cpp
    src/run_umap.cpp
    src/build_snn_graph.cpp
    src/cluster_graph.cpp
    src/cluster_kmeans.cpp
    src/score_markers.cpp
    src/summarize_effects.cpp
    src/aggregate_across_cells.cpp
    src/aggregate_across_genes.cpp
    src/combine_factors.cpp
    src/correct_mnn.cpp
    src/subsample_by_neighbors.cpp
    src/scale_by_neighbors.cpp
    src/score_gene_set.cpp
    src/test_enrichment.cpp
    src/init.cpp
)

target_include_directories(scranpy PRIVATE "${ASSORTHEAD_INCLUDE_DIR}")
target_include_directories(scranpy PRIVATE "${MATTRESS_INCLUDE_DIR}")
target_include_directories(scranpy PRIVATE "${KNNCOLLE_INCLUDE_DIR}")

set_property(TARGET scranpy PROPERTY CXX_STANDARD 17)

target_link_libraries(scranpy PRIVATE pybind11::pybind11)

find_package(igraph CONFIG REQUIRED)
target_link_libraries(scranpy PRIVATE igraph::igraph)

set_target_properties(scranpy PROPERTIES
    OUTPUT_NAME lib_scranpy
    PREFIX ""
)
