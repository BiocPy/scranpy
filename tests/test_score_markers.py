from biocframe import BiocFrame
from mattress import tatamize
from scranpy.marker_detection import score_markers
import numpy as np

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def test_score_markers(mock_data):
    y = mock_data.pcs
    out = tatamize(y)

    grouping = []
    for i in range(out.ncol()):
        grouping.append(i % 5)

    res = score_markers(out, grouping)

    assert res is not None
    assert "means" in res[1].columns
    assert isinstance(res[1].column("delta_detected"), BiocFrame)

    # Same results in parallel.
    resp = score_markers(out, grouping, num_threads=3)
    assert (res[0].column("means") == resp[0].column("means")).all()
    assert (res[1].column("cohen").column("mean") == resp[1].column("cohen").column("mean")).all()
    assert (res[2].column("auc").column("min_rank") == resp[2].column("auc").column("min_rank")).all()
    assert (res[3].column("lfc").column("min") == resp[3].column("lfc").column("min")).all()
