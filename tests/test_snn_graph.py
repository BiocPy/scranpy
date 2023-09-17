import igraph as ig
from scranpy import BuildSnnGraphOptions, build_snn_graph

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def test_build_snn_graph(mock_data):
    y = mock_data.pcs
    out = build_snn_graph(y)

    assert isinstance(out, ig.Graph)

    clustering = out.community_multilevel()

    assert clustering is not None
    assert clustering.membership is not None
    assert len(clustering.membership) == y.shape[0]

    # Same results in parallel.
    outp = build_snn_graph(y, options=BuildSnnGraphOptions(num_threads=3))
    assert outp.es["weight"] == out.es["weight"]
    assert outp.es.indices == out.es.indices
