import scranpy
import numpy


def test_aggregate_across_cells_single():
    numpy.random.seed(69)
    x = (numpy.random.rand(1000, 100) * 4).astype(numpy.int32)
    levels = ["A", "B", "C", "D", "E"]
    clusters = [levels[i] for i in (numpy.random.rand(x.shape[1]) * len(levels)).astype(numpy.int32)]
    agg = scranpy.aggregate_across_cells(x, (clusters,))

    assert len(agg.combinations) == 1
    assert agg.combinations[0] == levels
    assert [levels[i] for i in agg.index] == clusters
    assert agg.sum.shape == (1000, 5)
    assert agg.detected.shape == (1000, 5)

    for u, lev in enumerate(levels):
        chosen = [lev == c for c in clusters]
        assert numpy.array(chosen).sum() == agg.counts[u]
        submat = x[:,chosen]
        sum_expected = submat.sum(axis=1)
        assert numpy.allclose(sum_expected, agg.sum[:,u])
        detected_expected = (submat > 0).sum(axis=1)
        assert numpy.allclose(detected_expected, agg.detected[:,u])


def test_aggregate_across_cells_multiple():
    numpy.random.seed(692)
    x = (numpy.random.rand(1000, 500) * 4).astype(numpy.int32)
    levels = ["a", "b", "c", "d"]
    clusters = [levels[i] for i in (numpy.random.rand(x.shape[1]) * len(levels)).astype(numpy.int32)]
    samples = (numpy.random.rand(x.shape[1]) * 3).astype(numpy.int32)
    agg = scranpy.aggregate_across_cells(x, (clusters, samples))

    assert len(agg.combinations) == 2
    assert agg.combinations[0] == ["a"] * 3 + ["b"] * 3 + ["c"] * 3 + ["d"] * 3
    assert agg.combinations[1] == [0, 1, 2] * 4
    assert [agg.combinations[0][i] for i in agg.index] == clusters
    assert [agg.combinations[1][i] for i in agg.index] == list(samples)
    assert agg.sum.shape == (1000, 12)
    assert agg.detected.shape == (1000, 12)

    for u in range(len(agg.combinations[0])):
        curclust = agg.combinations[0][u]
        cursamp = agg.combinations[1][u]
        chosen = [clusters[i] == curclust and samples[i] == cursamp for i in range(x.shape[1])]
        assert numpy.array(chosen).sum() == agg.counts[u]
        submat = x[:,chosen]
        sum_expected = submat.sum(axis=1)
        assert numpy.allclose(sum_expected, agg.sum[:,u])
        detected_expected = (submat > 0).sum(axis=1)
        assert numpy.allclose(detected_expected, agg.detected[:,u])
