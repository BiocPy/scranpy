# scranpy

The **scranpy** package provides Python bindings to the single-cell analysis methods in [**libscran**](https://github.com/LTLA/libscran) and related C++ libraries.
It performs the standard steps in a typical single-cell analysis including quality control, normalization, feature selection, dimensionality reduction, clustering and marker detection.
**scranpy** makes heavy use of the [BiocPy](https://github.com/BiocPy) data structures in its user interface,
and uses the [**mattress**](https://pypi.org/project/mattress) package to provide a C++ representation of the underlying matrix data.
This package is effectively a mirror of its counterparts in Javascript ([**scran.js**](https://npmjs.com/package/scran.js)) and R ([**scran.chan**](https://github.com/LTLA/scran.chan)),
which are based on the same underlying C++ libraries and concepts.

## Contents

```{toctree}
:maxdepth: 2

Overview <readme>
Contributions <contributing>
License <license>
Authors <authors>
Changelog <changelog>
Module Reference <api/modules>
```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`

[Sphinx]: http://www.sphinx-doc.org/
[Markdown]: https://daringfireball.net/projects/markdown/
[reStructuredText]: http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
[MyST]: https://myst-parser.readthedocs.io/en/latest/
