from scranpy.analyze import AnalyzeResults, analyze

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def test_analyze(mock_data):
    x = mock_data.x
    out = analyze(x, features=[f"{i}" for i in range(1000)])

    assert isinstance(out, AnalyzeResults)
