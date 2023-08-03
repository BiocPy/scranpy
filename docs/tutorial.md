# Tutorial

This is a WIP!


# Load a dataset
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
```

# Perform QC
```python
import scranpy
metrics = scranpy.quality_control.per_cell_rna_qc_metrics(
    mat, 
    { "mito": scranpy.quality_control.guess_mito_from_symbols(features) }
)
thresholds = scranpy.quality_control.suggest_rna_qc_filters(metrics)
filter = scranpy.quality_control.create_rna_qc_filter(metrics, thresholds)
```

# Filter cells
```python
import mattress
ptr = mattress.tatamize(mat)
filtered = qc.filter_cells(ptr, filter)
```
# Log-normalize counts
```python
import scranpy.normalization as norm
normed = norm.log_norm_counts(filtered)
```

# Feature selection
```python
varstats = scranpy.feature_selection.model_gene_variances(normed)
selected = scranpy.feature_selection.choose_hvgs(varstats.column("residuals"))
pca = scranpy.dimensionality_reduction.run_pca(normed, rank=20, subset=selected)
```

# Identify clusters
```python
g = scranpy.clustering.build_snn_graph(pca.principal_components)
clusters = g.community_multilevel().membership
```

# Find markers

```python
markers = scranpy.marker_detection.score_markers(normed, clusters)
```

# Compute embeddings

```python
# TODO: run these all at once.
tsne = scranpy.dimensionality_reduction.run_tsne(pca.principal_components)
umap = scranpy.dimensionality_reduction.run_umap(pca.principal_components)
```