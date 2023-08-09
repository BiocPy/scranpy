from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


@dataclass
class FilterCellsArgs:
    """Arguments to filter cells.

    Attributes:
        discard (bool): Whether to discard the cells listed in `filter`.
            If `false`, the specified cells are retained instead, and all
            other cells are discarded. Defaults to True.
        verbose (bool, optional): Display logs?. Defaults to False.
    """

    discard: bool = True
    verbose: bool = False


@dataclass
class PerCellRnaQcMetricsArgs:
    """Arguments to compute per cell QC metrics (RNA).

    Attributes:
        subsets (Mapping, optional): Dictionary of feature subsets.
            Each key is the name of the subset and each value is an array of
            integer indices or booleans, specifying the rows of `x` belonging to the
            subset. Defaults to {}.
        num_threads (int, optional): Number of threads to use. Defaults to 1.
        verbose (bool, optional): Display logs?. Defaults to False.
    """

    subsets: Optional[Mapping] = None
    num_threads: int = 1
    verbose: bool = False


@dataclass
class SuggestRnaQcFilters:
    """Arguments to suggest QC Filters (RNA).

    Attributes:
        block (Sequence, optional): block assignment for each cell.
            This is used to segregate cells in order to perform comparisons within
            each block. Defaults to None, indicating all cells are part of the same
            block.
        num_mads (int, optional): Number of median absolute deviations to
            filter low-quality cells. Defaults to 3.
        verbose (bool, optional): Display logs?. Defaults to False.

    """

    block: Optional[Sequence] = None
    num_mads: int = 3
    verbose: bool = False


@dataclass
class CreateRnaQcFilter:
    """Arguments to create an RNA QC Filter.

    Attributes:
        block (Sequence, optional): Block assignment for each cell.
            This is used to segregate cells in order to perform comparisons within
            each block. Defaults to None, indicating all cells are part of the same
            block.
    """

    block: Optional[Sequence] = None
