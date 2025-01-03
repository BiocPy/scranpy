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

The **scranpy** package provides Python bindings to the single-cell analysis methods in the [**libscran**](https://github.com/libscran) C++ libraries.
It performs the standard steps in a typical single-cell analysis including quality control, normalization, feature selection, dimensionality reduction, clustering and marker detection.
This package is effectively a mirror of its counterparts in Javascript ([**scran.js**](https://npmjs.com/package/scran.js)) and R ([**scrapper**](https://github.com/libscran/scrapper)),
which are based on the same underlying C++ libraries and concepts.

## Quick start

Let's fetch a dataset from the [**scrnaseq**](https://github.com/BiocPy/scrnaseq) package:

```python
import scrnaseq 
sce = scrnaseq.fetch_dataset("zeisel-brain-2015", "2023-12-14", realize_assays=True)
```

Then we call **scranpy**'s `analyze()` functions, with some additional information about the mitochondrial subset for quality control purposes.

```python
import scranpy
results = scranpy.analyze(
    sce,
    rna_subsets = {
        "mito": [name.startswith("mt-") for name in sce.get_row_names()]
    }
)
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

Users can also convert this into other [**BiocPy**](https://github.com/BiocPy) classes for easier manipulation:

```python
results.to_singlecellexperiment()
results.rna_qc_metrics.to_biocframe()
results.rna_markers.to_biocframes()
```

We won't go over the theory here as it's explained more thoroughly in the book.
Also check out the [reference documentation](https://biocpy.github.io/scranpy) for more details.

## Multiple batches

To demonstrate, let's grab two pancreas datasets from the **scrnaseq** package.
Each dataset represents a separate batch of cells generated in different studies.

```python
import scrnaseq 
gsce = scrnaseq.fetch_dataset("grun-pancreas-2016", "2023-12-14", realize_assays=True)
msce = scrnaseq.fetch_dataset("muraro-pancreas-2016", "2023-12-19", realize_assays=True)
```

They don't have the same features, so we'll just take the intersection of their row names before combining them into a single `SingleCellExperiment` object:

```python
import biocutils
common = biocutils.intersect(gsce.get_row_names(), msce.get_row_names())
combined = biocutils.relaxed_combine_columns(
    gsce[biocutils.match(common, gsce.get_row_names()), :],
    msce[biocutils.match(common, msce.get_row_names()), :]
)
block = ["grun"] * gsce.shape[1] + ["muraro"] * msce.shape[1]
```

We can now perform a batch-aware analysis:

```python
import scranpy
results = scranpy.analyze(combined, block=block) # no mitochondrial genes in this case...
```

This yields mostly the same set of results as before, but with an extra MNN-corrected embedding for clustering, visualization, etc.
The blocking factor is also used in relevant functions to avoid problems with batch effects.

```python
results.mnn_corrected
```

## Multiple modalities

Let's grab a 10X Genomics immune profiling dataset (see [here](https://github.com/kanaverse/random-test-files/releases/download/10x-immune-v1.0.0/immune_3.0.0-tenx.h5)),
which contains count data for the entire transcriptome and targeted proteins:

```python
import singlecellexperiment
sce = singlecellexperiment.read_tenx_h5("immune_3.0.0-tenx.h5", realize_assays=True)
sce.set_row_names(sce.get_row_data().get_column("id"), in_place=True)
```

We split it to genes and ADTs:

```python
feattypes = sce.get_row_data().get_column("feature_type")
gene_data = sce[[x == "Gene Expression" for x in feattypes],:]
adt_data = sce[[x == "Antibody Capture" for x in feattypes],:]
```

And now we can run the analysis:

```python
import scranpy
results = scranpy.analyze(
    gene_data,
    adt_x = adt_data, 
    rna_subsets = { 
        "mito": [n.startswith("MT-") for n in gene_data.get_row_data().get_column("name")]
    },
    adt_subsets = {
        "igg": [n.startswith("IgG") for n in adt_data.get_row_data().get_column("name")]
    }
)
```

This returns ADT-specific results in the relevant fields, as well as a set of combined PCs for use in clustering, visualization, etc. 

```python
results.adt_size_factors
results.adt_markers
results.combined_pca
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
