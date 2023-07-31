import igraph as ig
from scranpy.clustering import build_snn_graph

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
