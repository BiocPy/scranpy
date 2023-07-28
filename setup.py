"""
    Setup file for scranpy.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 4.5.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""
import mattress
from setuptools import setup
from setuptools.extension import Extension

if __name__ == "__main__":
    try:
        setup(
            use_scm_version={"version_scheme": "no-guess-dev"},
            ext_modules=[
                Extension(
                    "scranpy.core",
                    [
                        "src/scranpy/lib/per_cell_rna_qc_metrics.cpp",
                        "src/scranpy/lib/log_norm_counts.cpp",
                        "src/scranpy/lib/suggest_rna_qc_filters.cpp",
                        "src/scranpy/lib/model_gene_variances.cpp",
                    ],
                    include_dirs=[
                        "extern/libscran/include",
                        "extern/weightedlowess/include",
                    ]
                    + mattress.includes(),
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
