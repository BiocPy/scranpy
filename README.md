<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Built Status](https://api.cirrus-ci.com/github/<USER>/scranpy.svg?branch=main)](https://cirrus-ci.com/github/<USER>/scranpy)
[![ReadTheDocs](https://readthedocs.org/projects/scranpy/badge/?version=latest)](https://scranpy.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/scranpy/main.svg)](https://coveralls.io/r/<USER>/scranpy)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/scranpy.svg)](https://anaconda.org/conda-forge/scranpy)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/scranpy)
-->

[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)
[![PyPI-Server](https://img.shields.io/pypi/v/scranpy.svg)](https://pypi.org/project/scranpy/)
[![Downloads](https://static.pepy.tech/badge/scranpy/month)](https://pepy.tech/project/scranpy)
![Unit tests](https://github.com/BiocPy/scranpy/actions/workflows/pypi-test.yml/badge.svg)

# scran, in Python

## Overview

The **scranpy** package provides Python bindings to the single-cell analysis methods in [**libscran**](https://github.com/LTLA/libscran) and related C++ libraries.
It performs the standard steps in a typical single-cell analysis including quality control, normalization, feature selection, dimensionality reduction, clustering and marker detection.
**scranpy** makes heavy use of the [BiocPy](https://github.com/BiocPy) data structures in its user interface,
while it uses the [**mattress**](https://pypi.org/project/mattress) package to provide a C++ representation of the underlying matrix data.
This package is effectively a mirror of its counterparts in Javascript ([**scran.js**](https://npmjs.com/package/scran.js)) and R ([**scran.chan**](https://github.com/LTLA/scran.chan)),
which are based on the same underlying C++ libraries and concepts.

## Quick start

🚧🚧🚧 **Under construction** 🚧🚧🚧

Currently, it's anything but quick... at least it runs.
More to come soon.

```python
# TODO: streamline the loader:
path = "pbmc4k-tenx.h5"
import h5py as h5
fhandle = h5.File(path)
import scipy.sparse as sp
mat = sp.csc_matrix(
    (fhandle["matrix"]["data"], fhandle["matrix"]["indices"], fhandle["matrix"]["indptr"]),
    fhandle["matrix"]["shape"]
)
features = [x.decode("ascii") for x in fhandle["matrix"]["features"]["name"]]

import scranpy
options = scranpy.AnalyzeOptions()
options.per_cell_rna_qc_metrics_options.subsets = {
    "mito": scranpy.quality_control.guess_mito_from_symbols(features, "mt-")
}

results = scranpy.analyze(mat)
```

## Developer Notes

Steps to setup dependencies -

- initialize git submodules in `extern/libscran`.

- run `cmake .` inside the `extern/knncolle` to download the annoy library. a future version of this will use a cmake to setup the extern directory.

First one needs to build the extern library, this would generate a shared object file to `src/scranpy/core-[*].so`

```shell
python setup.py build_ext --inplace
```

For typical development workflows, run this for tests

```shell
python setup.py build_ext --inplace && tox
```

To rebuild the **ctypes** bindings [**cpptypes**](https://github.com/BiocPy/ctypes-wrapper):

```shell
cpptypes src/scranpy/lib --py src/scranpy/cpphelpers.py --cpp src/scranpy/lib/bindings.cpp
```

To rebuild the [dry run analysis source code](src/scranpy/analysis_dry.py):

```shell
./scripts/dryrun.py src/scranpy/analyze/live_analyze.py > src/scranpy/analyze/dry_analyze.py
```

<!-- pyscaffold-notes -->

## Note

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.
