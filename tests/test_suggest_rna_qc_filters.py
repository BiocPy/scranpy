from scranpy import (
    PerCellRnaQcMetricsOptions,
    SuggestRnaQcFiltersOptions,
    per_cell_rna_qc_metrics,
    suggest_rna_qc_filters,
)
from biocframe import BiocFrame

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def test_suggest_rna_qc_filters(mock_data):
    x = mock_data.x
    result = per_cell_rna_qc_metrics(
        x, options=PerCellRnaQcMetricsOptions(subsets={"foo": [1, 10, 100]})
    )
    filters = suggest_rna_qc_filters(result)

    assert filters is not None
    assert filters.dims[0] == 1
    assert filters.column("sums") is not None
    assert filters.column("detected") is not None
    assert filters.column("subset_proportions") is not None
    assert filters.column("subset_proportions").column("foo") is not None

    #  with blocks
    filters_blocked = suggest_rna_qc_filters(
        result, options=SuggestRnaQcFiltersOptions(block=mock_data.block)
    )

    assert filters_blocked.shape[0] == 3
    assert len(list(set(filters_blocked.row_names).difference(["A", "B", "C"]))) == 0

    subfilters = filters_blocked.column("subset_proportions")
    assert len(list(set(subfilters.row_names).difference(["A", "B", "C"]))) == 0


def test_suggest_rna_qc_filters_custom(mock_data):
    x = mock_data.x
    result = per_cell_rna_qc_metrics(
        x, options=PerCellRnaQcMetricsOptions(subsets={"foo": [1, 10, 100]})
    )

    # First without blocks.
    filters_custom = suggest_rna_qc_filters(
        result,
        options=SuggestRnaQcFiltersOptions(
            custom_thresholds=BiocFrame(
                {
                    "sums": [1],
                    "detected": [2],
                    "subset_proportions": BiocFrame({"foo": [3]}),
                }
            )
        ),
    )

    assert filters_custom.shape[0] == 1
    assert filters_custom.column("sums")[0] == 1
    assert filters_custom.column("detected")[0] == 2
    assert filters_custom.column("subset_proportions").column("foo")[0] == 3

    # Now with some blocks, in order.
    filters_custom = suggest_rna_qc_filters(
        result,
        options=SuggestRnaQcFiltersOptions(
            block=mock_data.block,
            custom_thresholds=BiocFrame(
                {
                    "sums": [1, 2, 3],
                    "detected": [4, 5, 6],
                    "subset_proportions": BiocFrame({"foo": [7, 8, 9]}),
                },
                row_names=["A", "B", "C"],
            ),
        ),
    )

    assert filters_custom.shape[0] == 3
    assert list(filters_custom.column("sums")) == [1, 2, 3]
    assert list(filters_custom.column("detected")) == [4, 5, 6]
    assert list(filters_custom.column("subset_proportions").column("foo")) == [7, 8, 9]

    # Now with some blocks, out of order.
    filters_custom = suggest_rna_qc_filters(
        result,
        options=SuggestRnaQcFiltersOptions(
            block=mock_data.block,
            custom_thresholds=BiocFrame(
                {
                    "sums": [1, 2, 3],
                    "detected": [4, 5, 6],
                    "subset_proportions": BiocFrame({"foo": [7, 8, 9]}),
                },
                row_names=["C", "B", "A"],
            ),
        ),
    )

    assert filters_custom.shape[0] == 3
    assert list(filters_custom.column("sums")) == [3, 2, 1]
    assert list(filters_custom.column("detected")) == [6, 5, 4]
    assert list(filters_custom.column("subset_proportions").column("foo")) == [9, 8, 7]
