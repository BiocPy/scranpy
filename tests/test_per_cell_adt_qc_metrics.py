import numpy as np
from scranpy import PerCellAdtQcMetricsOptions, per_cell_adt_qc_metrics


def test_per_cell_adt_qc_metrics(mock_data):
    x = mock_data.x
    result = per_cell_adt_qc_metrics(
        x, options=PerCellAdtQcMetricsOptions(subsets={"foo": [1, 10, 100]})
    )

    assert result is not None
    assert result.dims[0] == 100
    assert result.column("sums") is not None
    assert result.column("detected") is not None
    assert result.column("subset_totals") is not None
    assert result.column("subset_totals").column("foo") is not None

    # Works without any subsets.
    result0 = per_cell_adt_qc_metrics(x)
    assert result0.column("sums") is not None
    assert result0.column("detected") is not None
    assert result0.column("subset_totals").shape[1] == 0

    # Same results when running in parallel.
    resultp = per_cell_adt_qc_metrics(
        x,
        options=PerCellAdtQcMetricsOptions(
            subsets={"BAR": [1, 10, 100]}, num_threads=3
        ),
    )
    assert np.array_equal(result.column("sums"), resultp.column("sums"))
    assert np.array_equal(result.column("detected"), resultp.column("detected"))
    assert np.array_equal(
        result.column("subset_totals").column(0),
        resultp.column("subset_totals").column(0),
    )
