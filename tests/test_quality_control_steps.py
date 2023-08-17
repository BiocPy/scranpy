import numpy as np
from scranpy.quality_control import (
    CreateRnaQcFilterOptions,
    PerCellRnaQcMetricsOptions,
    SuggestRnaQcFiltersOptions,
    create_rna_qc_filter,
    guess_mito_from_symbols,
    per_cell_rna_qc_metrics,
    suggest_rna_qc_filters,
)
from biocframe import BiocFrame

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def test_quality_control_numpy(mock_data):
    x = mock_data.x
    result = per_cell_rna_qc_metrics(
        x, options=PerCellRnaQcMetricsOptions(subsets={"foo": [1, 10, 100]})
    )

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
    resultp = per_cell_rna_qc_metrics(
        x,
        options=PerCellRnaQcMetricsOptions(
            subsets={"BAR": [1, 10, 100]}, num_threads=3
        ),
    )
    assert np.array_equal(result.column("sums"), resultp.column("sums"))
    assert np.array_equal(result.column("detected"), resultp.column("detected"))
    assert np.array_equal(
        result.column("subset_proportions").column(0),
        resultp.column("subset_proportions").column(0),
    )


def test_guess_mito_from_symbols():
    out = guess_mito_from_symbols(["asdasd", "mt-asdas", "sadasd", "MT-asdasd"])
    assert out == [1, 3]


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
    assert len(list(set(filters_blocked.rowNames).difference(["A", "B", "C"]))) == 0

    subfilters = filters_blocked.column("subset_proportions")
    assert len(list(set(subfilters.rowNames).difference(["A", "B", "C"]))) == 0

    # checking that the filters get applied.
    keep = create_rna_qc_filter(result, filters)
    assert len(keep) == result.shape[0]

    keep_blocked = create_rna_qc_filter(
        result, filters_blocked, options=CreateRnaQcFilterOptions(block=mock_data.block)
    )
    assert len(keep_blocked) == result.shape[0]

def test_suggest_rna_qc_filters_custom(mock_data):
    x = mock_data.x
    result = per_cell_rna_qc_metrics(
        x, options=PerCellRnaQcMetricsOptions(subsets={"foo": [1, 10, 100]})
    )

    # First without blocks.
    filters_custom = suggest_rna_qc_filters(
        result, 
        options=SuggestRnaQcFiltersOptions(
            custom_thresholds=BiocFrame({
                "sums": [1],
                "detected": [2],
                "subset_proportions": BiocFrame({
                    "foo": [3]                    
                })
            })
        )
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
            custom_thresholds=BiocFrame({
                "sums": [1, 2, 3],
                "detected": [4, 5, 6],
                "subset_proportions": BiocFrame({
                    "foo": [7, 8, 9]
                }),
            }, rowNames = [ "A", "B", "C" ])
        )
    )

    assert filters_custom.shape[0] == 3
    assert list(filters_custom.column("sums")) == [1,2,3]
    assert list(filters_custom.column("detected")) == [4,5,6]
    assert list(filters_custom.column("subset_proportions").column("foo")) == [7,8,9]

    # Now with some blocks, out of order.
    filters_custom = suggest_rna_qc_filters(
        result, 
        options=SuggestRnaQcFiltersOptions(
            block=mock_data.block,
            custom_thresholds=BiocFrame({
                "sums": [1, 2, 3],
                "detected": [4, 5, 6],
                "subset_proportions": BiocFrame({
                    "foo": [7, 8, 9]
                }),
            }, rowNames = [ "C", "B", "A" ])
        )
    )

    assert filters_custom.shape[0] == 3
    assert list(filters_custom.column("sums")) == [3,2,1]
    assert list(filters_custom.column("detected")) == [6,5,4]
    assert list(filters_custom.column("subset_proportions").column("foo")) == [9,8,7]
