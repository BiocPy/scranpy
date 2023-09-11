import scranpy
import numpy


def test_aggregate_across_cells_simple():
    groups = ["A", "B", "C", "D", "A", "B", "C", "A"]
    x = numpy.round(numpy.random.rand(1000, 8) * 10)
    y = scranpy.aggregate_across_cells(groups)

