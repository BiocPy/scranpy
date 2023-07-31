import numpy as np
from scranpy.nearest_neighbors import (
    NeighborResults,
    NNResult,
    build_neighbor_index,
    find_nearest_neighbors,
)

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def test_neighbors(mock_data):
    y = mock_data.pcs
    idx = build_neighbor_index(y, approximate=False)
    assert idx.num_cells() == 1000
    assert idx.num_dimensions() == 50

    res = find_nearest_neighbors(idx, k=10)
    assert res.num_cells() == 1000
    assert res.num_neighbors() == 10
    assert isinstance(res.get(2), NNResult)

    stuff = res.serialize()
    res2 = NeighborResults.unserialize(stuff)
    assert res2 is not None
