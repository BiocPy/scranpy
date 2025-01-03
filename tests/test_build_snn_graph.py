import scranpy
import numpy
import knncolle


def test_build_snn_graph():
    data = numpy.random.rand(10, 1000)
    out = scranpy.build_snn_graph(data)
    assert out.vertices == 1000
    assert len(out.edges) == 2 * len(out.weights)
    assert (out.weights > 0).all()
    assert numpy.logical_and(out.edges >= 0, out.edges < 1000).all()

    # Same results when given an index, and throwing in some parallelization.
    idx = knncolle.build_index(knncolle.AnnoyParameters(), data.T)
    out2 = scranpy.build_snn_graph(idx, num_threads=2)
    assert out.vertices == out2.vertices
    assert (out.edges == out2.edges).all()
    assert (out.weights == out2.weights).all()

    # Same results when given a matrix of indices.
    res = knncolle.find_knn(idx, num_neighbors=10, get_distance=False)
    out2 = scranpy.build_snn_graph(res, num_threads=2)
    assert out.vertices == out2.vertices
    assert (out.edges == out2.edges).all()
    assert (out.weights == out2.weights).all()

    # Conversion works.
    g = out.as_igraph()
    assert g.vcount() == 1000
    assert g.ecount() == len(out.edges)/2
    assert (g.es["weight"] == out.weights).all()
    assert not g.is_directed()
