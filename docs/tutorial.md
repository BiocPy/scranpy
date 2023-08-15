# Getting started

## Setting up

To use **scranpy**, you'll need Python 3.9 or higher.
You'll also need to install **scranpy** itself (duh), which is available from [PyPi](https://pypi.org/project/scranpy) via the usual commands:

```sh
pip install scranpy
```

## Loading a dataset

The full **scranpy** workflow starts from a feature-by-cell count matrix.
To illustrate, let's load one of the famous peripheral blood mononuclear cell (PBMC) datasets from 10X Genomics.
A copy of this dataset is available [here](https://github.com/kanaverse/random-test-files/releases/download/10x-pbmc-v1.0.0/pbmc4k-tenx.h5).

```python
# TODO: streamline the loader:
path = "pbmc4k-tenx.h5"
import h5py as h5
fhandle = h5.File(path)
import scipy.sparse as sp
mat = sp.csc_matrix(
    (
        fhandle["matrix"]["data"],
        fhandle["matrix"]["indices"],
        fhandle["matrix"]["indptr"]
    ),
    fhandle["matrix"]["shape"]
)
features = [x.decode("ascii") for x in fhandle["matrix"]["features"]["name"]]
```

Obviously, you'll be using different code for your own dataset.
The important thing is that you get a count matrix where features are in the rows and cells are in the columns.
Oh, and an array of feature names.

## Running the analysis

Now we draw the rest of the owl.

![Draw the rest of the owl](https://i.kym-cdn.com/photos/images/original/000/572/078/d6d.jpg)

It's as simple as calling **scranpy**'s `analyze()` function:

```python
import scranpy
results = scranpy.analyze(mat, features)
```

This will perform all of the usual steps for a routine single-cell analysis, as described in Bioconductor's [Orchestrating single cell analysis](https://bioconductor.org/books/OSCA) book.
Specifically:

1. Quality control ([`scranpy.quality_control`](api/scranpy.quality_control.rst))
2. Normalization and log-transformation ([`scranpy.normalization`](api/scranpy.normalization.rst))
3. Feature selection ([`scranpy.feature_selection`](api/scranpy.feature_selection.rst))
4. Dimensionality reduction with PCA, UMAP and t-SNE ([`scranpy.dimensionality_reduction`](api/scranpy.dimensionality_reduction.rst))
5. Clustering ([`scranpy.clustering`](api/scranpy.clustering.rst))
6. Marker detection ([`scranpy.marker_detection`](api/scranpy.clustering.rst))

We won't go over the theory here as it's explained more thoroughly in the book.
But you can also check out each module's reference documentation for more details.

# Customizing the analysis

Stay tuned!
