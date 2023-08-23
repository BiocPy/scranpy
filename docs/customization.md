# Customizing the analysis

## Setting options

Advanced users can customize the behavior of the `analyze()` function by setting options for specific function calls:

```python
options = scranpy.AnalyzeOptions()
options.choose_hvgs_options.number = 2000
options.run_pca_options.rank = 50
options.run_tsne_options.perplexity = 20
options.run_umap_options.min_dist = 0.5
results = scranpy.analyze(mat, features, options = options)
```

The options available for each step can be found by following the [`AnalyzeOptions` documentation](api/scranpy.rst#scranpy.analyze_live.AnalyzeOptions).

## Getting individual commands

If you need even more control, you can extract the function calls for the individual steps in `analyze()` by setting `dry_run = True`.
This performs a dry run of the analysis, capturing the calls and reporting them in a string.

```python
# Matrix and features aren't actually required here, as they don't get used.
print(scranpy.analyze(None, None, options = options, dry_run = True))
```

You can then copy-paste this string into your own scripts for further customization, e.g., to swap out algorithms for various steps.
Note that the stringified commands assume that your matrix is named `matrix` and your vector of features is named `features`.
