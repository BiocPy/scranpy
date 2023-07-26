<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Built Status](https://api.cirrus-ci.com/github/<USER>/scranpy.svg?branch=main)](https://cirrus-ci.com/github/<USER>/scranpy)
[![ReadTheDocs](https://readthedocs.org/projects/scranpy/badge/?version=latest)](https://scranpy.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/scranpy/main.svg)](https://coveralls.io/r/<USER>/scranpy)
[![PyPI-Server](https://img.shields.io/pypi/v/scranpy.svg)](https://pypi.org/project/scranpy/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/scranpy.svg)](https://anaconda.org/conda-forge/scranpy)
[![Monthly Downloads](https://pepy.tech/badge/scranpy/month)](https://pepy.tech/project/scranpy)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/scranpy)
-->

[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)

# scranpy

Fast multi-modal single-cell data analysis! stay tuned...

## Developer Notes


Steps to setup dependencies - 

- initialize git submodules in `extern/libscran`.

First one needs to build the extern library, this would generate a shared object file to `src/scranpy/core-[*].so`

```shell
python setup.py build_ext --inplace
```

For typical development workflows, run this for tests

```shell
python setup.py build_ext --inplace && tox
```



<!-- pyscaffold-notes -->

## Note

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.
