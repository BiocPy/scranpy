from scranpy import (
    PerCellCrisprQcMetricsOptions,
    SuggestCrisprQcFiltersOptions,
    per_cell_crispr_qc_metrics,
    suggest_crispr_qc_filters,
)
from biocframe import BiocFrame

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def test_suggest_crispr_qc_filters(mock_data):
    x = mock_data.x
    result = per_cell_crispr_qc_metrics(x)
    filters = suggest_crispr_qc_filters(result)

    assert filters is not None
    assert filters.dims[0] == 1
    assert filters.column("max_count") is not None

    #  with blocks
    filters_blocked = suggest_crispr_qc_filters(
        result, options=SuggestCrisprQcFiltersOptions(block=mock_data.block)
    )

    assert filters_blocked.shape[0] == 3
    assert len(list(set(filters_blocked.row_names).difference(["A", "B", "C"]))) == 0


def test_suggest_crispr_qc_filters_custom(mock_data):
    x = mock_data.x
    result = per_cell_crispr_qc_metrics(x)

    # First without blocks.
    filters_custom = suggest_crispr_qc_filters(
        result,
        options=SuggestCrisprQcFiltersOptions(
            custom_thresholds=BiocFrame(
                {
                    "max_count": [2],
                }
            )
        ),
    )

    assert filters_custom.shape[0] == 1
    assert filters_custom.column("max_count")[0] == 2

    # Now with some blocks, in order.
    filters_custom = suggest_crispr_qc_filters(
        result,
        options=SuggestCrisprQcFiltersOptions(
            block=mock_data.block,
            custom_thresholds=BiocFrame(
                {
                    "max_count": [4, 5, 6],
                },
                row_names=["A", "B", "C"],
            ),
        ),
    )

    assert filters_custom.shape[0] == 3
    assert list(filters_custom.column("max_count")) == [4, 5, 6]

    # Now with some blocks, out of order.
    filters_custom = suggest_crispr_qc_filters(
        result,
        options=SuggestCrisprQcFiltersOptions(
            block=mock_data.block,
            custom_thresholds=BiocFrame(
                {
                    "max_count": [4, 5, 6],
                },
                row_names=["C", "B", "A"],
            ),
        ),
    )

    assert filters_custom.shape[0] == 3
    assert list(filters_custom.column("max_count")) == [6, 5, 4]
