import scranpy
import numpy
import copy


def test_cluster_graph_multilevel():
    data = numpy.random.randn(10, 1000)
    g = scranpy.build_snn_graph(data)

    clust = scranpy.cluster_graph(g, method="multilevel")
    assert (clust.membership >= 0).all()
    assert (clust.membership < 1000).all()

    for i, lev in enumerate(clust.levels):
        assert (lev >= 0).all()
        assert (lev < 1000).all()

    # Works without weights.
    ug = copy.copy(g)
    ug.weights = None
    uclust = scranpy.cluster_graph(ug, method="multilevel")
    assert (uclust.membership >= 0).all()
    assert (uclust.membership < 1000).all()


def test_cluster_graph_leiden():
    data = numpy.random.randn(10, 1000)
    g = scranpy.build_snn_graph(data)

    clust = scranpy.cluster_graph(g, method="leiden")
    assert (clust.membership >= 0).all()
    assert (clust.membership < 1000).all()

    # Works without weights.
    ug = copy.copy(g)
    ug.weights = None
    uclust = scranpy.cluster_graph(ug, leiden_objective="cpm", method="leiden")
    assert (uclust.membership >= 0).all()
    assert (uclust.membership < 1000).all()


def test_cluster_graph_walktrap():
    data = numpy.random.randn(10, 1000)
    g = scranpy.build_snn_graph(data)

    clust = scranpy.cluster_graph(g, method="walktrap")
    assert (clust.membership >= 0).all()
    assert (clust.membership < 1000).all()

    # Works without weights.
    ug = copy.copy(g)
    ug.weights = None
    uclust = scranpy.cluster_graph(ug, method="walktrap")
    assert (uclust.membership >= 0).all()
    assert (uclust.membership < 1000).all()
