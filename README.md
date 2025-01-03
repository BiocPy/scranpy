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
![Unit tests](https://github.com/libscran/scranpy/actions/workflows/run-tests.yml/badge.svg)

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
print(sce)
## class: SingleCellExperiment
## dimensions: (20006, 3005)
## assays(1): ['counts']
## row_data columns(1): ['featureType']
## row_names(20006): ['Tspan12', 'Tshz1', 'Fnbp1l', ..., 'mt-Rnr2', 'mt-Rnr1', 'mt-Nd4l']
## column_data columns(9): ['tissue', 'group #', 'total mRNA mol', 'well', 'sex', 'age', 'diameter', 'level1class', 'level2class']
## column_names(3005): ['1772071015_C02', '1772071017_G12', '1772071017_A05', ..., '1772063068_D01', '1772066098_A12', '1772058148_F03']
## main_experiment_name: gene
## reduced_dims(0): []
## alternative_experiments(2): ['repeat', 'ERCC']
## row_pairs(0): []
## column_pairs(0): []
## metadata(0):
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
print(results.clusters)
## [ 0  0  0 ...  6  6 13]

print(results.tsne)
## [[  6.24189264   6.12559936   5.41776875 ...  20.07822751  18.25022123
##    14.78338538]
##  [-28.82249121 -28.18510674 -28.92849565 ...   7.73694327   3.70750309
##     7.13103324]]

print(results.umap)
## [[ 9.84396648  9.73148155  9.83376408 ... -6.64551735 -5.74155378
##   -4.41887522]
##  [-1.26350224 -1.16540933 -1.13979638 ... -5.63315582 -4.83151293
##   -6.02484226]]

first_markers = results.rna_markers.to_biocframes(summaries=["median"])[0]
first_markers.set_row_names(results.rna_row_names, in_place=True)
print(first_markers)
## BiocFrame with 20006 rows and 6 columns
##                        mean            detected     cohens_d_median          auc_median   delta_mean_median delta_detected_median
##          <ndarray[float64]>  <ndarray[float64]>  <ndarray[float64]>  <ndarray[float64]>  <ndarray[float64]>    <ndarray[float64]>
## Tspan12 0.35759151503119846  0.3157894736842105 0.31138667468315545  0.5989624247185128 0.31138667468315545   0.31138667468315545
##   Tshz1  0.5997779968274406 0.41776315789473684 0.36865087228075244  0.6031352215283973 0.36865087228075244   0.36865087228075244
##  Fnbp1l  1.1660581606996154              0.6875  0.7644031115934953  0.6905056759545924  0.7644031115934953    0.7644031115934953
##                         ...                 ...                 ...                 ...                 ...                   ...
## mt-Rnr2   6.966227511583628  0.9967105263157895 -0.7666238430581961  0.2928277982073087 -0.7666238430581961   -0.7666238430581961
## mt-Rnr1   4.914541788016454  0.9769736842105263 -0.4847704371628273  0.3834708208344696 -0.4847704371628273   -0.4847704371628273
## mt-Nd4l  3.2901199968427246  0.9342105263157894 -0.5903983282435646 0.30724666142969365 -0.5903983282435646   -0.5903983282435646
```

Users can also convert the results into a `SingleCellExperiment` for easier manipulation:

```python
print(results.to_singlecellexperiment())
## class: SingleCellExperiment
## dimensions: (20006, 2874)
## assays(2): ['filtered', 'normalized']
## row_data columns(5): ['mean', 'variance', 'fitted', 'residual', 'is_highly_variable']
## row_names(20006): ['Tspan12', 'Tshz1', 'Fnbp1l', ..., 'mt-Rnr2', 'mt-Rnr1', 'mt-Nd4l']
## column_data columns(5): ['sum', 'detected', 'subset_proportion_mito', 'size_factors', 'clusters']
## column_names(2874): ['1772071015_C02', '1772071017_G12', '1772071017_A05', ..., '1772066097_D04', '1772063068_D01', '1772066098_A12']
## main_experiment_name:
## reduced_dims(3): ['pca', 'tsne', 'umap']
## alternative_experiments(0): []
## row_pairs(0): []
## column_pairs(0): []
## metadata(0):
```

Check out the [reference documentation](https://libscran.github.io/scranpy) for more details.

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
print(combined)
## class: SingleCellExperiment
## dimensions: (18499, 4800)
## assays(1): ['counts']
## row_data columns(2): ['symbol', 'chr']
## row_names(18499): ['A1BG-AS1__chr19', 'A1BG__chr19', 'A1CF__chr10', ..., 'ZYX__chr7', 'ZZEF1__chr17', 'ZZZ3__chr1']
## column_data columns(4): ['donor', 'sample', 'label', 'plate']
## column_names(4800): ['D2ex_1', 'D2ex_2', 'D2ex_3', ..., 'D30-8_94', 'D30-8_95', 'D30-8_96']
## main_experiment_name: endogenous
## reduced_dims(0): []
## alternative_experiments(0): []
## row_pairs(0): []
## column_pairs(0): []
## metadata(0):
```

We can now perform a batch-aware analysis, where the blocking factor is also used in relevant functions to avoid problems with batch effects.

```python
import scranpy
block = ["grun"] * gsce.shape[1] + ["muraro"] * msce.shape[1]
results = scranpy.analyze(combined, block=block) # no mitochondrial genes in this case...
```

This yields mostly the same set of results as before, but with an extra MNN-corrected embedding for clustering, visualization, etc.

```python
results.mnn_corrected.corrected
## array([[-1.87690275e+01, -2.20133721e+01, -2.01364711e+01, ...,
##          1.60988874e+01, -2.10494187e+01, -9.41325421e+00],
##        [ 9.95069366e+00,  1.12168142e+01,  1.40745981e+01, ...,
##         -5.63689417e+00, -1.46003926e+01, -4.02325382e+00],
##        [ 1.17917046e+01,  8.40756681e+00,  1.24557851e+01, ...,
##          3.65281722e+00, -1.13280613e+01, -1.12939448e+01],
##        ...,
##        [-4.20177077e+00,  3.64443391e-01,  1.13834851e+00, ...,
##          1.43898885e-02, -2.24228270e+00, -5.89749453e-01],
##        [-2.49456306e+00,  6.82624452e-01,  2.30363317e+00, ...,
##          1.09145269e+00,  3.17776365e+00,  8.27058276e-01],
##        [-2.03562222e+00,  2.04701389e+00,  5.64774034e-01, ...,
##          4.31078606e-01, -4.02375136e-01,  8.52493315e-01]],
##       shape=(25, 3984))
```

## Multiple modalities

Let's grab a 10X Genomics immune profiling dataset (see [here](https://github.com/kanaverse/random-test-files/releases/download/10x-immune-v1.0.0/immune_3.0.0-tenx.h5)),
which contains count data for the entire transcriptome and targeted proteins:

```python
import singlecellexperiment
sce = singlecellexperiment.read_tenx_h5("immune_3.0.0-tenx.h5", realize_assays=True)
sce.set_row_names(sce.get_row_data().get_column("id"), in_place=True)
## class: SingleCellExperiment
## dimensions: (33555, 8258)
## assays(1): ['counts']
## row_data columns(7): ['feature_type', 'genome', 'id', 'name', 'pattern', 'read', 'sequence']
## row_names(33555): ['ENSG00000243485', 'ENSG00000237613', 'ENSG00000186092', ..., 'IgG2b', 'CD127', 'CD15']
## column_data columns(1): ['barcodes']
## column_names(0):
## main_experiment_name:
## reduced_dims(0): []
## alternative_experiments(0): []
## row_pairs(0): []
## column_pairs(0): []
## metadata(0):
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
print(results.adt_size_factors)
## [0.79359408 0.79410332 0.89536413 ... 0.79207839 0.66492723 0.76847637]

print(results.combined_pca.combined)
## [[-9.97603155e+00 -1.04045057e+01 -1.26408576e+01 ... -1.29361354e+01
##   -1.09887392e+01 -1.08070608e+01]
##  [ 7.47726554e+00  6.77629373e+00  1.78091509e+00 ...  2.22256539e+00
##    5.96667219e+00  7.10437993e+00]
##  [-2.63898263e+00 -1.24485522e+00  5.51002546e+00 ...  5.21037673e+00
##   -5.54233035e+00 -3.38828724e+00]
##  ...
##  [-2.04699441e-01 -4.38991650e-01 -2.87170731e+00 ...  2.36527395e+00
##    7.05969255e-01 -2.46180209e-01]
##  [ 4.75688909e-01 -1.54557081e-01 -1.30053159e+00 ...  2.81492567e+00
##    1.21607502e+00 -3.12194853e-01]
##  [ 8.56575012e-02  8.74924626e-03 -7.17362957e-04 ...  1.65769854e-01
##    1.73927253e-01  5.04057044e-02]]

second_markers = results.adt_markers.to_biocframes(summaries=["min_rank"])[1]
second_markers.set_row_names(results.adt_row_names, in_place=True)
print(second_markers)
## BiocFrame with 17 rows and 6 columns
##                      mean           detected cohens_d_min_rank      auc_min_rank delta_mean_min_rank delta_detected_min_rank
##        <ndarray[float64]> <ndarray[float64]> <ndarray[uint32]> <ndarray[uint32]>   <ndarray[uint32]>       <ndarray[uint32]>
##    CD3  11.04397358391642                1.0                 1                 1                   1                       1
##   CD19  4.072383130863625                1.0                 4                 4                   4                       4
## CD45RA 10.481785289114054                1.0                 1                 1                   1                       1
##                       ...                ...               ...               ...                 ...                     ...
##  IgG2b 2.8690172565558263 0.9911190053285968                 6                 5                   6                       6
##  CD127  6.258223461814724                1.0                 2                 2                   2                       2
##   CD15  5.366264191077669                1.0                 4                 4                   4                       4
```

## Customizing the analysis

Most parameters can be changed by modifying the relevant arguments in `analyze()`.
For example:

```python
import scrnaseq 
sce = scrnaseq.fetch_dataset("zeisel-brain-2015", "2023-12-14", realize_assays=True)
is_mito = [name.startswith("mt-") for name in sce.get_row_names()]

import scranpy
results = scranpy.analyze(
    sce,
    rna_subsets = {
        "mito": is_mito
    },
    build_snn_graph_options = {
        "num_neighbors": 10
    },
    cluster_graph_options = {
        "multilevel_resolution": 2
    },
    run_pca_options = {
        "number": 15
    },
    run_tsne_options = {
        "perplexity": 25
    },
    run_umap_options = {
        "min_dist": 0.05
    }
)
```

For finer control, users can call each step individually via lower-level functions.
A typical RNA analysis might be implemented as:

```python
counts = sce.assay(0)
qcmetrics = scranpy.compute_rna_qc_metrics(counts, subsets=is_mito)
thresholds = scranpy.suggest_rna_qc_thresholds(qcmetrics)
filter = scranpy.filter_rna_qc_metrics(thresholds, metrics)

import delayedarray # avoid an actual copy of the matrix.
filtered = delayedarray.DelayedArray(rna_x)[:,filter]

sf = scranpy.center_size_factors(qcmetrics.sum[filter])
normalized = scranpy.normalize_counts(filtered, sf)

vardf = scranpy.model_gene_variances(normalized)
hvgs = scranpy.choose_highly_variable_genes(vardf.residual)
pca = scranpy.run_pca(normalized[hvgs,:])

nn_out = scranpy.run_all_neighbor_steps(pca.components)
clusters = nn_out.cluster_graph.membership
markers = scranpy.score_markers(normalized, groups=clusters)
```

Check out [`analyze.py`](src/scranpy/analyze.py) for more details.
