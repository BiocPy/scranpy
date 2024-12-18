from typing import Sequence, Tuple, Union
from collections.abc import Mapping

import numpy
import biocutils


def _sanitize_subsets(x: Union[Sequence, Mapping], extent: int) -> Tuple:
    if isinstance(x, biocutils.NamedList):
        keys = x.names().as_list()
        vals = list(x.as_list())
    elif isinstance(x, Mapping):
        keys = x.keys()
        vals = list(x.values())
    else:
        keys = None
        vals = list(x)
    for i, s in enumerate(vals):
        vals[i] = _to_logical(s, extent, dtype=numpy.bool)
    return keys, vals


def _to_logical(selection: Sequence, length: int, dtype=numpy.bool) -> numpy.ndarray:
    output = numpy.zeros((length,), dtype=dtype)

    if isinstance(selection, range) or isinstance(selection, slice):
        output[selection] = 1
        return output

    if isinstance(selection, numpy.ndarray):
        if selection.dtype == numpy.bool_:
            if len(selection) != length:
                raise ValueError("length of 'selection' is not equal to 'length'.")
            output[selection] = 1
            return output
        elif selection.dtype == numpy.int_:
            output[selection] = 1
            return output
        else:
            raise TypeError(
                "'selection`s' dtype not supported, must be 'boolean' or 'int',"
                f"provided {selection.dtype}"
            )

    if len(selection) == 0:
        has_bool = False
        has_number = True
    else:
        has_bool = False
        has_number = False
        for ss in selection:
            if isinstance(ss, bool):
                has_bool = True
            elif isinstance(ss, int):
                has_number = True

    if (has_number and has_bool) or (not has_bool and not has_number):
        raise TypeError("'selection' should only contain booleans or numbers")

    if has_bool:
        if len(selection) != length:
            raise ValueError("length of 'selection' is not equal to 'length'.")
        output[:] = selection
    elif has_number:
        output[selection] = 1

    return output
