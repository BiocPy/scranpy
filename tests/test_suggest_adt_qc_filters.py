from scranpy import (
    PerCellAdtQcMetricsOptions,
    SuggestAdtQcFiltersOptions,
    per_cell_adt_qc_metrics,
    suggest_adt_qc_filters,
)
from biocframe import BiocFrame

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def test_suggest_adt_qc_filters(mock_data):
    x = mock_data.x
    result = per_cell_adt_qc_metrics(
        x, options=PerCellAdtQcMetricsOptions(subsets={"foo": [1, 10, 100]})
    )
    filters = suggest_adt_qc_filters(result)

    assert filters is not None
    assert filters.dims[0] == 1
    assert filters.column("detected") is not None
    assert filters.column("subset_totals") is not None
    assert filters.column("subset_totals").column("foo") is not None

    #  with blocks
    filters_blocked = suggest_adt_qc_filters(
        result, options=SuggestAdtQcFiltersOptions(block=mock_data.block)
    )

    assert filters_blocked.shape[0] == 3
    assert len(list(set(filters_blocked.row_names).difference(["A", "B", "C"]))) == 0

    subfilters = filters_blocked.column("subset_totals")
    assert len(list(set(subfilters.row_names).difference(["A", "B", "C"]))) == 0


def test_suggest_adt_qc_filters_custom(mock_data):
    x = mock_data.x
    result = per_cell_adt_qc_metrics(
        x, options=PerCellAdtQcMetricsOptions(subsets={"foo": [1, 10, 100]})
    )

    # First without blocks.
    filters_custom = suggest_adt_qc_filters(
        result,
        options=SuggestAdtQcFiltersOptions(
            custom_thresholds=BiocFrame(
                {
                    "detected": [2],
                    "subset_totals": BiocFrame({"foo": [3]}),
                }
            )
        ),
    )

    assert filters_custom.shape[0] == 1
    assert filters_custom.column("detected")[0] == 2
    assert filters_custom.column("subset_totals").column("foo")[0] == 3

    # Now with some blocks, in order.
    filters_custom = suggest_adt_qc_filters(
        result,
        options=SuggestAdtQcFiltersOptions(
            block=mock_data.block,
            custom_thresholds=BiocFrame(
                {
                    "detected": [4, 5, 6],
                    "subset_totals": BiocFrame({"foo": [7, 8, 9]}),
                },
                row_names=["A", "B", "C"],
            ),
        ),
    )

    assert filters_custom.shape[0] == 3
    assert list(filters_custom.column("detected")) == [4, 5, 6]
    assert list(filters_custom.column("subset_totals").column("foo")) == [7, 8, 9]

    print("HEREEEEE")

    # Now with some blocks, out of order.
    filters_custom = suggest_adt_qc_filters(
        result,
        options=SuggestAdtQcFiltersOptions(
            block=mock_data.block,
            custom_thresholds=BiocFrame(
                {
                    "detected": [4, 5, 6],
                    "subset_totals": BiocFrame({"foo": [7, 8, 9]}),
                },
                row_names=["C", "B", "A"],
            ),
        ),
    )

    print("filters_custom", filters_custom)

    assert filters_custom.shape[0] == 3
    assert list(filters_custom.column("detected")) == [6, 5, 4]
    assert list(filters_custom.column("subset_totals").column("foo")) == [9, 8, 7]
