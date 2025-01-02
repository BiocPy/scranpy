import scranpy
import numpy
import biocutils


def test_combine_factors_single():
    numpy.random.rand(10001)
    levels = ["A", "B", "C", "D", "E"]
    x = [levels[i] for i in (numpy.random.rand(100) * len(levels)).astype(numpy.int32)]
    outcomb, outind = scranpy.combine_factors((x,))

    assert len(outcomb) == 1
    assert outcomb[0] == levels
    assert [levels[i] for i in outind] == x

    # Drops unused elements.
    x = biocutils.Factor.from_sequence(["A", "D", "F", "B"], levels=["A", "B", "C", "D", "E", "F", "G"])
    outcomb, outind = scranpy.combine_factors((x,))
    assert len(outcomb) == 1
    expected = ["A", "B", "D", "F"]
    assert outcomb[0].as_list() == expected
    assert [expected[i] for i in outind] == list(x)

    outcomb, outind = scranpy.combine_factors((x,), keep_unused=True)
    assert len(outcomb) == 1
    expected = ["A", "B", "C", "D", "E", "F", "G"]
    assert outcomb[0].as_list() == expected
    assert [expected[i] for i in outind] == list(x)


def test_combine_factors_multiple():
    numpy.random.rand(10002)
    upper_levels = ["A", "B", "C", "D", "E"]
    x = [upper_levels[i] for i in (numpy.random.rand(1000) * len(upper_levels)).astype(numpy.int32)]
    y = (numpy.random.rand(1000) * 3 + 10).astype(numpy.int32)
    lower_levels = ["x", "y", "z"]
    z = [lower_levels[i] for i in (numpy.random.rand(1000) * len(lower_levels)).astype(numpy.int32)]

    outcomb, outind = scranpy.combine_factors((x, y, z))
    assert len(outcomb) == 3
    assert len(outcomb[0]) == 45
    assert [outcomb[0][i] for i in outind] == x
    assert [outcomb[1][i] for i in outind] == list(y)
    assert [outcomb[2][i] for i in outind] == z


def test_combine_factors_multiple_unused():
    numpy.random.rand(10003)
    x = ["A","B","C","D","E"]
    y = [1,2,3,1,2]
    z = ["x", "x", "y", "y", "z"]

    # Sanity check.
    outcomb, outind = scranpy.combine_factors((x, y, z))
    assert len(outcomb) == 3
    assert len(outcomb[0]) < 45
    assert [outcomb[0][i] for i in outind] == x
    assert [outcomb[1][i] for i in outind] == list(y)
    assert [outcomb[2][i] for i in outind] == z

    outcomb, outind = scranpy.combine_factors((x, y, z), keep_unused=True)
    print((x, y, z))
    assert len(outcomb) == 3
    assert len(outcomb[0]) == 45
    assert [outcomb[0][i] for i in outind] == x
    assert [outcomb[1][i] for i in outind] == list(y)
    assert [outcomb[2][i] for i in outind] == z
