#include "pybind11/pybind11.h"

void init_adt_quality_control(pybind11::module&);
void init_rna_quality_control(pybind11::module&);
void init_crispr_quality_control(pybind11::module&);
void init_normalize_counts(pybind11::module&);
void init_center_size_factors(pybind11::module&);
void init_sanitize_size_factors(pybind11::module&);
void init_compute_clrm1_factors(pybind11::module&);
void init_choose_pseudo_count(pybind11::module&);
void init_model_gene_variances(pybind11::module&);
void init_fit_variance_trend(pybind11::module&);
void init_choose_highly_variable_genes(pybind11::module&);
void init_run_pca(pybind11::module&);
void init_run_tsne(pybind11::module&);
void init_run_umap(pybind11::module&);
void init_build_snn_graph(pybind11::module&);
void init_cluster_graph(pybind11::module&);
void init_cluster_kmeans(pybind11::module&);
void init_score_markers(pybind11::module&);
void init_summarize_effects(pybind11::module&);
void init_aggregate_across_cells(pybind11::module&);
void init_aggregate_across_genes(pybind11::module&);
void init_combine_factors(pybind11::module&);
void init_correct_mnn(pybind11::module&);
void init_subsample_by_neighbors(pybind11::module&);
void init_scale_by_neighbors(pybind11::module&);
void init_score_gene_set(pybind11::module&);
void init_test_enrichment(pybind11::module&);

PYBIND11_MODULE(lib_scranpy, m) {
    init_adt_quality_control(m);
    init_rna_quality_control(m);
    init_crispr_quality_control(m);
    init_normalize_counts(m);
    init_center_size_factors(m);
    init_sanitize_size_factors(m);
    init_compute_clrm1_factors(m);
    init_choose_pseudo_count(m);
    init_model_gene_variances(m);
    init_fit_variance_trend(m);
    init_choose_highly_variable_genes(m);
    init_run_pca(m);
    init_run_tsne(m);
    init_run_umap(m);
    init_build_snn_graph(m);
    init_cluster_graph(m);
    init_cluster_kmeans(m);
    init_score_markers(m);
    init_summarize_effects(m);
    init_aggregate_across_cells(m);
    init_aggregate_across_genes(m);
    init_combine_factors(m);
    init_correct_mnn(m);
    init_subsample_by_neighbors(m);
    init_scale_by_neighbors(m);
    init_score_gene_set(m);
    init_test_enrichment(m);
}
