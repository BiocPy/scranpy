from biocframe import BiocFrame
from mattress import tatamize
from scranpy import ScoreMarkersOptions, score_markers

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def test_score_markers(mock_data):
    x = mock_data.x
    out = tatamize(x)

    grouping = []
    for i in range(out.ncol()):
        grouping.append(i % 5)

    res = score_markers(out, grouping=grouping)

    assert res is not None
    assert "means" in res["1"].columns
    assert isinstance(res["1"].column("delta_detected"), BiocFrame)

    # Works when blocks are supplied.
    resb = score_markers(
        out, grouping=grouping, options=ScoreMarkersOptions(block=mock_data.block)
    )
    assert resb is not None
    assert "detected" in resb["1"].columns
    assert isinstance(resb["1"].column("lfc"), BiocFrame)

    # Same results in parallel.
    resp = score_markers(
        out, grouping=grouping, options=ScoreMarkersOptions(num_threads=3)
    )
    assert (res["0"].column("means") == resp["0"].column("means")).all()
    assert (
        res["1"].column("cohen").column("mean") == resp["1"].column("cohen").column("mean")
    ).all()
    assert (
        res["2"].column("auc").column("min_rank")
        == resp["2"].column("auc").column("min_rank")
    ).all()
    assert (
        res["3"].column("lfc").column("min") == resp["3"].column("lfc").column("min")
    ).all()
