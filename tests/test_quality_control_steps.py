import numpy as np
import scranpy as scr
from scranpy.quality_control import per_cell_rna_qc_metrics

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"

def test_quality_control_numpy():
    x = np.random.rand(1000, 100)
    result = per_cell_rna_qc_metrics(x, subsets=[[1, 10, 100]])

    assert result is not None
    assert isinstance(result, scr.types.RnaQcResult)
    assert result.sums is not None
    assert result.detected is not None
    assert result.subset_proportions is not None

    # Same results when running in parallel.
    resultp = per_cell_rna_qc_metrics(x, subsets=[[1, 10, 100]], num_threads = 3)
    assert np.array_equal(result["sums"], resultp["sums"])
    assert np.array_equal(result["detected"], resultp["detected"])
    assert np.array_equal(result["subset_proportions"][0], resultp["subset_proportions"][0])

