from scranpy import (
    CreateRnaQcFilterOptions,
    PerCellRnaQcMetricsOptions,
    SuggestRnaQcFiltersOptions,
    create_adt_qc_filter,
    per_cell_adt_qc_metrics,
    suggest_adt_qc_filters,
)

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def test_create_adt_qc_filter(mock_data):
    x = mock_data.x
    result = per_cell_adt_qc_metrics(
        x, options=PerCellRnaQcMetricsOptions(subsets={"foo": [1, 10, 100]})
    )

    # No blocks.
    filters = suggest_adt_qc_filters(result)
    keep = create_adt_qc_filter(result, filters)
    assert len(keep) == result.shape[0]

    #  with blocks
    filters_blocked = suggest_adt_qc_filters(
        result, options=SuggestRnaQcFiltersOptions(block=mock_data.block)
    )
    keep_blocked = create_adt_qc_filter(
        result, filters_blocked, options=CreateRnaQcFilterOptions(block=mock_data.block)
    )
    assert len(keep_blocked) == result.shape[0]
