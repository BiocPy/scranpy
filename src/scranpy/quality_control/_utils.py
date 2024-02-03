from typing import Optional, Tuple, Union, List

from biocframe import BiocFrame
from numpy import array, float64, ndarray, zeros

from .._utils import create_pointer_array, match_lists


def process_subset_columns(subsets: BiocFrame) -> Tuple[List[ndarray], ndarray]:
    skeys = subsets.column_names
    subset_in = []
    for s in skeys:
        subset_in.append(array(subsets.column(s), dtype=float64, copy=False))
    return subset_in, create_pointer_array(subset_in)


def create_subset_buffers(
    length: int, num_subsets: int
) -> Tuple[List[ndarray], ndarray]:
    subset_out = []
    for i in range(num_subsets):
        subset_out.append(zeros((length,), dtype=float64))
    return subset_out, create_pointer_array(subset_out)


def create_subset_frame(
    column_names: list,
    columns: List[ndarray],
    num_rows: int,
    row_names: Optional[list] = None,
) -> BiocFrame:
    output = BiocFrame({}, number_of_rows=num_rows, row_names=row_names)
    for i, n in enumerate(column_names):
        output[n] = columns[i]
    return output


def check_custom_thresholds(
    num_blocks: int, block_names: list, custom_thresholds: BiocFrame
) -> Union[None, BiocFrame]:
    if custom_thresholds is None:
        return None

    if num_blocks != custom_thresholds.shape[0]:
        raise ValueError(
            "number of rows in 'custom_thresholds' should equal the number of blocks"
        )

    if num_blocks > 1 and custom_thresholds.row_names != block_names:
        m = match_lists(block_names, custom_thresholds.row_names)
        if m is None:
            raise ValueError(
                "row names of 'custom_thresholds' should equal the unique values of 'block'"
            )
        custom_thresholds = custom_thresholds[list(m), :]
    return custom_thresholds
