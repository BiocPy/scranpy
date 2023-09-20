"""Setup file for scranpy. Use setup.cfg to configure your project.

This file was generated with PyScaffold 4.5.
PyScaffold helps you to put up the scaffold of your new Python project.
Learn more under: https://pyscaffold.org/
"""
import mattress
from setuptools import setup
from setuptools.extension import Extension
import assorthead

if __name__ == "__main__":
    try:
        setup(
            use_scm_version={"version_scheme": "no-guess-dev"},
            ext_modules=[
                Extension(
                    "scranpy._core",
                    [
                        "src/scranpy/lib/per_cell_rna_qc_metrics.cpp",
                        "src/scranpy/lib/per_cell_adt_qc_metrics.cpp",
                        "src/scranpy/lib/per_cell_crispr_qc_metrics.cpp",
                        "src/scranpy/lib/log_norm_counts.cpp",
                        "src/scranpy/lib/center_size_factors.cpp",
                        "src/scranpy/lib/suggest_rna_qc_filters.cpp",
                        "src/scranpy/lib/suggest_adt_qc_filters.cpp",
                        "src/scranpy/lib/suggest_crispr_qc_filters.cpp",
                        "src/scranpy/lib/filter_cells.cpp",
                        "src/scranpy/lib/model_gene_variances.cpp",
                        "src/scranpy/lib/choose_hvgs.cpp",
                        "src/scranpy/lib/run_pca.cpp",
                        "src/scranpy/lib/find_nearest_neighbors.cpp",
                        "src/scranpy/lib/build_snn_graph.cpp",
                        "src/scranpy/lib/run_tsne.cpp",
                        "src/scranpy/lib/run_umap.cpp",
                        "src/scranpy/lib/score_markers.cpp",
                        "src/scranpy/lib/mnn_correct.cpp",
                        "src/scranpy/lib/aggregate_across_cells.cpp",
                        "src/scranpy/lib/downsample_by_neighbors.cpp",
                        "src/scranpy/lib/scale_by_neighbors.cpp",
                        "src/scranpy/lib/hypergeometric_test.cpp",
                        "src/scranpy/lib/score_feature_set.cpp",
                        "src/scranpy/lib/grouped_size_factors.cpp",
                        "src/scranpy/lib/bindings.cpp",
                    ],
                    include_dirs=[assorthead.includes()] + mattress.includes(),
                    language="c++",
                    extra_compile_args=[
                        "-std=c++17",
                    ],
                )
            ],
        )
    except:  # noqa
        print(
            "\n\nAn error occurred while building the project, "
            "please ensure you have the most updated version of setuptools, "
            "setuptools_scm and wheel with:\n"
            "   pip install -U setuptools setuptools_scm wheel\n\n"
        )
        raise
