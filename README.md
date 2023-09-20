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

Let's load in the famous PBMC 4k dataset from 10X Genomics (available [here](https://github.com/kanaverse/random-test-files/releases/tag/10x-pbmc-v1.0.0)):

```python
import singlecellexperiment
sce = singlecellexperiment.read_tenx_h5("pbmc4k-tenx.h5")
```

Then we just need to call one of **scranpy**'s `analyze()` functions.
(We do have to tell it what the mitochondrial genes are, though.)

```python
import scranpy
options = scranpy.AnalyzeOptions()
options.per_cell_rna_qc_metrics_options.subsets = {
    "mito": scranpy.guess_mito_from_symbols(sce.row_data["name"], "mt-")
}
results = scranpy.analyze_sce(sce, options=options)
```

This will perform all of the usual steps for a routine single-cell analysis, 
as described in Bioconductor's [Orchestrating single cell analysis](https://bioconductor.org/books/OSCA) book.
It returns an object containing clusters, t-SNEs, UMAPs, marker genes, and so on:

```python
results.clusters
results.tsne
results.umap
results.rna_markers
```

We won't go over the theory here as it's explained more thoroughly in the book.
Check out the [reference documentation](https://biocpy.github.io/scranpy) for more details.

## Multiple batches

To demonstrate, let's grab two batches of PBMC datasets from 10X Genomics (again, available [here](https://github.com/kanaverse/random-test-files/releases/tag/10x-pbmc-v1.0.0)):

```python
import singlecellexperiment
sce3k = singlecellexperiment.read_tenx_h5("pbmc3k-tenx.h5")
sce4k = singlecellexperiment.read_tenx_h5("pbmc4k-tenx.h5")
```

They don't have the same features, so we'll just take the intersection of their Ensembl IDs before combining them:

```python
import biocutils
common = biocutils.intersect(sce3k.row_data["id"], sce4k.row_data["id"])
sce3k_common = sce3k[biocutils.match(common, sce3k.row_data["id"]), :]
sce4k_common = sce4k[biocutils.match(common, sce4k.row_data["id"]), :]

import scipy.sparse
combined = scipy.sparse.hstack((sce3k_common.assay(0), sce4k_common.assay(0)))
batch = ["3k"] * sce3k_common.shape[1] + ["4k"] * sce4k_common.shape[1]
```

We can now perform a batch-aware analysis:

```python
import scranpy
options = scranpy.AnalyzeOptions()
options.per_cell_rna_qc_metrics_options.subsets = {
    "mito": scranpy.guess_mito_from_symbols(sce3k_common.row_data["name"], "mt-")
}
options.miscellaneous_options.block = batch
results = scranpy.analyze(combined, options=options)
```

This yields mostly the same set of results as before, but with an extra MNN-corrected embedding for clustering, visualization, etc.

```python
results.mnn
```

## Multiple modalities

Let's grab a 10X Genomics immune profiling dataset (see [here](https://github.com/kanaverse/random-test-files/releases/download/10x-immune-v1.0.0/immune_3.0.0-tenx.h5)):

```python
import singlecellexperiment
sce = singlecellexperiment.read_tenx_h5("immune_3.0.0-tenx.h5")
```

We need to split it to genes and ADTs:

```python
is_gene = [x == "Gene Expression" for x in sce.row_data["feature_type"]]
gene_data = sce[is_gene,:]
is_adt = [x == "Antibody Capture" for x in sce.row_data["feature_type"]]
adt_data = sce[is_adt,:]
```

And now we can run the analysis:

```python
import scranpy
options = scranpy.AnalyzeOptions()
options.per_cell_rna_qc_metrics_options.subsets = {
    "mito": scranpy.guess_mito_from_symbols(gene_data.row_data["name"], "mt-")
}
options.per_cell_adt_qc_metrics_options.subsets = {
    "igg": [n.lower().startswith("igg") for n in adt_data.row_data["name"]]
}
results = scranpy.analyze_se(gene_data, adt_se = adt_data, options=options)
```

This returns ADT-specific results in the relevant fields, as well as a set of combined PCs for use in clustering, visualization, etc. 

```python
results.adt_size_factors
results.adt_markers
results.combined_pcs
```

## Customizing the analysis

Most parameters can be changed by setting the relevant fields in the `AnalyzeOptions` object.
For example, we can modify the number of neighbors and resolution used for graph-based clustering:

```python
options.build_snn_graph_options.num_neighbors = 10
options.miscellaneous_options.snn_graph_multilevel_resolution = 2
```

Or we can fiddle the the various dimensionality reduction parameters:

```python
options.run_pca_options.rank = 50
options.run_tsne_options.perplexity = 20
options.run_umap_options.min_dist = 0.5
```

The `AnalyzeOptions` has a few convenience methods to easily set the same parameter across multiple `*_options` attributes.
For example, to enable parallel processing in every step:

```python
options.set_threads(5)
```

Advanced users can even obtain the sequence of steps used internally by `analyze()` by calling it with `dry_run = True`:

```python
commands = scranpy.analyze(sce, dry_run = True)
print(commands)
## import scranpy
## import numpy
## 
## results = AnalyzeResults()
## ...
```

Users can then add, remove or replace steps as desired.

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
cpptypes src/scranpy/lib --py src/scranpy/_cpphelpers.py --cpp src/scranpy/lib/bindings.cpp --dll _core
```

To rebuild the [dry run analysis source code](src/scranpy/analysis_dry.py):

```shell
./scripts/dryrun.py src/scranpy/analyze/live_analyze.py > src/scranpy/analyze/dry_analyze.py
```
