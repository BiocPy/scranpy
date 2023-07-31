import numpy as np
from scranpy.nearest_neighbors import (
    NeighborResults,
    build_neighbor_index,
    find_nearest_neighbors,
)

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def test_neighbors():
    y = np.random.rand(50, 1000)  # PCs in rows, cells in dimensions
    idx = build_neighbor_index(y, approximate=False)
    assert idx.num_cells() == 1000
    assert idx.num_dimensions() == 50

    res = find_nearest_neighbors(idx, k=10)
    assert res.num_cells() == 1000
    assert res.num_neighbors() == 10
    assert len(res.get(2).keys()) == 2

    stuff = res.serialize()
    res2 = NeighborResults.unserialize(stuff)
    assert res2 is not None
