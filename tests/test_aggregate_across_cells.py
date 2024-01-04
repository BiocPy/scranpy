import scranpy
import numpy
from biocframe import BiocFrame


def test_aggregate_across_cells_simple():
    groups = ["A", "B", "C", "D", "A", "B", "C", "A"]
    x = numpy.round(numpy.random.rand(1000, 8) * 5)
    y = scranpy.aggregate_across_cells(x, groups)

    assert list(y.col_data.column("factor_1")) == ["A", "B", "C", "D"]
    assert list(y.col_data.column("counts")) == [3, 2, 2, 1]

    obs_a = y.assay("sums")[:, 0]
    exp_a = x[:, [0, 4, 7]].sum(axis=1)
    assert (obs_a == exp_a).all()

    obs_c = y.assay("detected")[:, 2]
    exp_c = (x[:, [2, 6]] != 0).sum(axis=1)
    assert (obs_c == exp_c).all()


def test_aggregate_across_cells_combinations():
    groups = ["A", "B", "A", "B", "A", "B", "A", "B"]
    batches = [1, 1, 1, 1, 2, 2, 2, 2]
    x = numpy.round(numpy.random.rand(1000, 8) * 5)
    y = scranpy.aggregate_across_cells(x, {"group": groups, "batch": batches})

    assert y.col_data.column("group") == ["A", "A", "B", "B"]
    assert y.col_data.column("batch") == ["1", "2", "1", "2"]
    assert list(y.col_data.column("counts")) == [2, 2, 2, 2]

    obs_a = y.assay("sums")[:, 1]
    exp_a = x[:, [4, 6]].sum(axis=1)
    assert (obs_a == exp_a).all()

    obs_c = y.assay("detected")[:, 2]
    exp_c = (x[:, [1, 3]] != 0).sum(axis=1)
    assert (obs_c == exp_c).all()

    # Try out different input types.
    y2 = scranpy.aggregate_across_cells(x, (groups, batches))
    assert y2.col_data.column("factor_1") == ["A", "A", "B", "B"]
    assert y2.col_data.column("factor_2") == ["1", "2", "1", "2"]
    assert (y2.assay("sums") == y.assay("sums")).all()

    y2 = scranpy.aggregate_across_cells(x, BiocFrame({"g": groups, "b": batches}))
    assert y2.col_data.column("g") == ["A", "A", "B", "B"]
    assert y2.col_data.column("b") == ["1", "2", "1", "2"]
    assert (y2.assay("detected") == y.assay("detected")).all()
