from scranpy import (
    BuildNeighborIndexOptions,
    FindNearestNeighborsOptions,
    NeighborResults,
    SingleNeighborResults,
    build_neighbor_index,
    find_nearest_neighbors,
)

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def test_neighbors(mock_data):
    y = mock_data.pcs
    idx = build_neighbor_index(y, BuildNeighborIndexOptions(approximate=False))
    assert idx.num_cells() == y.shape[0]
    assert idx.num_dimensions() == y.shape[1]

    res = find_nearest_neighbors(idx, k=10)
    assert res.num_cells() == y.shape[0]
    assert res.num_neighbors() == 10
    assert isinstance(res.get(2), SingleNeighborResults)

    stuff = res.serialize()
    res2 = NeighborResults.unserialize(stuff)
    assert res2 is not None

    # Same results after parallelization.
    resp = find_nearest_neighbors(
        idx, k=10, options=FindNearestNeighborsOptions(num_threads=3)
    )
    nn_old = res.get(0)
    nn_new = resp.get(0)
    assert (nn_old.index == nn_new.index).all()
    assert (nn_old.distance == nn_new.distance).all()

    nn_old = res.get(idx.num_cells() - 1)
    nn_new = resp.get(idx.num_cells() - 1)
    assert (nn_old.index == nn_new.index).all()
    assert (nn_old.distance == nn_new.distance).all()
