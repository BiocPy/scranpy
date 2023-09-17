import numpy as np
from scranpy import (
    PerCellCrisprQcMetricsOptions,
    per_cell_crispr_qc_metrics,
)


def test_per_cell_crispr_qc_metrics(mock_data):
    x = mock_data.x
    result = per_cell_crispr_qc_metrics(x)

    assert result is not None
    assert result.dims[0] == 100
    assert result.column("sums") is not None
    assert result.column("detected") is not None
    assert result.column("max_index") is not None
    assert result.column("max_proportion") is not None

    # Same results when running in parallel.
    resultp = per_cell_crispr_qc_metrics(
        x,
        options=PerCellCrisprQcMetricsOptions(num_threads=3),
    )
    assert np.array_equal(result.column("sums"), resultp.column("sums"))
    assert np.array_equal(result.column("detected"), resultp.column("detected"))
    assert np.array_equal(result.column("max_index"), resultp.column("max_index"))
    assert np.array_equal(
        result.column("max_proportion"), resultp.column("max_proportion")
    )
