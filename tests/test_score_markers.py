from biocframe import BiocFrame
from mattress import tatamize
from scranpy.marker_detection import score_markers

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def test_build_snn_graph(mock_data):
    y = mock_data.pcs
    out = tatamize(y)

    grouping = []
    for i in range(out.ncol()):
        grouping.append(i % 5)

    res = score_markers(out, grouping)

    assert res is not None
    assert "means" in res[1].columns
    assert isinstance(res[1].column("delta_detected"), BiocFrame)
