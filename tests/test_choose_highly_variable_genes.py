import numpy
import scranpy


def test_choose_highly_variable_genes():
    stats = numpy.random.rand(10000)
    out = scranpy.choose_highly_variable_genes(stats, top=2000)
    assert len(out) == 2000
    other = numpy.array([True] * len(stats))
    other[out] = False
    assert stats[out].min() > stats[other].max()

    out = scranpy.choose_highly_variable_genes(stats, top=numpy.inf)
    assert (out == numpy.array(range(len(stats)))).all()

    out = scranpy.choose_highly_variable_genes(stats, larger=False)
    assert len(out) == 4000
    other = numpy.array([True] * len(stats))
    other[out] = False
    assert stats[out].min() < stats[other].max()

    out = scranpy.choose_highly_variable_genes(stats, bound=0.9)
    assert len(out) < 2000
    assert stats[out].min() > 0.9

    out = scranpy.choose_highly_variable_genes(numpy.zeros(10000), keep_ties=True)
    assert (out == numpy.array(range(len(stats)))).all()
