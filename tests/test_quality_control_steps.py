import numpy as np
from scranpy.quality_control import per_cell_rna_qc_metrics, suggest_rna_qc_filters

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def test_quality_control_numpy():
    x = np.random.rand(1000, 100)
    result = per_cell_rna_qc_metrics(x, subsets={"foo": [1, 10, 100]})

    assert result is not None
    assert result.dims[0] == 100
    assert result.column("sums") is not None
    assert result.column("detected") is not None
    assert result.column("subset_proportions") is not None
    assert result.column("subset_proportions").column("foo") is not None

    # Works without any subsets.
    result0 = per_cell_rna_qc_metrics(x)
    assert result0.column("sums") is not None
    assert result0.column("detected") is not None
    assert result0.column("subset_proportions").shape[1] == 0

    # Same results when running in parallel.
    resultp = per_cell_rna_qc_metrics(x, subsets={"BAR": [1, 10, 100]}, num_threads=3)
    assert np.array_equal(result.column("sums"), resultp.column("sums"))
    assert np.array_equal(result.column("detected"), resultp.column("detected"))
    assert np.array_equal(
        result.column("subset_proportions").column(0),
        resultp.column("subset_proportions").column(0),
    )


def test_suggest_rna_qc_filters():
    x = np.random.rand(1000, 100)
    result = per_cell_rna_qc_metrics(x, subsets={"foo": [1, 10, 100]})
    filters = suggest_rna_qc_filters(result)

    assert filters is not None
    assert filters.dims[0] == 1
    assert filters.column("sums") is not None
    assert filters.column("detected") is not None
    assert filters.column("subset_proportions") is not None
    assert filters.column("subset_proportions").column("foo") is not None

    # Adding blocks
    # TODO: factor out random block generation into a separate function!)
    x = np.random.rand(1000, 100) * 20
    block_levels = ["A", "B", "C"]
    block = []
    for i in range(x.shape[1]):
        block.append(block_levels[i % len(block_levels)])

    filters_blocked = suggest_rna_qc_filters(result, block=block)

    assert filters_blocked.shape[0] == 3
    assert len(list(set(filters_blocked.rowNames).difference(["A", "B", "C"]))) == 0

    subfilters = filters_blocked.column("subset_proportions")
    assert len(list(set(subfilters.rowNames).difference(["A", "B", "C"]))) == 0
