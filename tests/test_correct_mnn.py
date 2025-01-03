import scranpy
import numpy
import pytest


def test_correct_mnn_simple():
    numpy.random.seed(696969)
    x = numpy.asfortranarray(numpy.random.randn(10, 1000))
    block = ["A"] * 500 + ["B"] * 300 + ["C"] * 200
    x[:,[i == "B" for i in block]] += 3
    x[:,[i == "C" for i in block]] += 5

    Ameans = x[:,[i == "A" for i in block]].mean()
    Bmeans = x[:,[i == "B" for i in block]].mean()
    Cmeans = x[:,[i == "C" for i in block]].mean()
    assert Bmeans > Ameans + 2
    assert Cmeans > Bmeans + 1

    res = scranpy.correct_mnn(x, block)
    assert res.corrected.shape == (10, 1000)
    assert len(res.num_pairs) == 2
    assert res.merge_order == ["A", "B", "C"] # largest block always have the most RSS, and the most MNN pairs.

    Ameans = res.corrected[:,[i == "A" for i in block]].mean()
    Bmeans = res.corrected[:,[i == "B" for i in block]].mean()
    Cmeans = res.corrected[:,[i == "C" for i in block]].mean()
    assert abs(Bmeans - Ameans) < 0.5
    assert abs(Cmeans - Bmeans) < 0.5


def test_correct_mnn_order():
    numpy.random.seed(69696969)
    x = numpy.asfortranarray(numpy.random.randn(10, 500))
    block = ["x", "y"] * 250
    x[:,[i == "x" for i in block]] += 10

    res = scranpy.correct_mnn(x, block, order=["y", "x"])
    assert res.corrected.shape == (10, 500)
    assert len(res.num_pairs) == 1
    assert res.merge_order == ["y", "x"]

    # Actually has an effect.
    res2 = scranpy.correct_mnn(x, block, order=["x", "y"])
    assert res.corrected.shape == (10, 500)
    assert (res.corrected != res2.corrected).any()

    with pytest.raises(Exception, match="cannot find"):
        scranpy.correct_mnn(x, block, order=["z", "x"])

    with pytest.raises(Exception, match="duplicate"):
        scranpy.correct_mnn(x, block, order=["x", "x"])

    with pytest.raises(Exception, match="number of batches"):
        scranpy.correct_mnn(x, block, order=["x"])
