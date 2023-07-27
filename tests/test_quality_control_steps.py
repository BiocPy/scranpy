import numpy as np
import scranpy as scr
from scranpy.quality_control import per_cell_rna_qc_metrics

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"

def test_quality_control_numpy():
    x = np.random.rand(1000, 100)
    result = per_cell_rna_qc_metrics(x, subsets={ "foo": [1, 10, 100] })

    assert result is not None
    assert result.dims[0] == 100
    assert result.column("sums") is not None
    assert result.column("detected") is not None
    assert result.column("subset_proportions") is not None
    assert result.column("subset_proportions").column("foo") is not None

    # Works without any subsets.
    result0 = per_cell_rna_qc_metrics(x)
    assert result.column("sums") is not None
    assert result.column("detected") is not None
    assert result.column("subset_proportions").shape[1] == 0

    # Same results when running in parallel.
    resultp = per_cell_rna_qc_metrics(x, subsets=[[1, 10, 100]], num_threads = 3)
    assert np.array_equal(result.column("sums"), resultp.column("sums"))
    assert np.array_equal(result.column("detected"), resultp.column("detected"))
    assert np.array_equal(result.column("subset_proportions").column(0), resultp.column("subset_proportions").column(0))

