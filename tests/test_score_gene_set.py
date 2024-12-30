import scranpy
import numpy


def test_score_gene_set():
    numpy.random.rand(1000)
    x = numpy.random.randn(1000, 100)

    res = scranpy.score_gene_set(x, range(10, 60))
    assert len(res.scores) == x.shape[1]
    assert len(res.weights) == 50
    print(res.scores)
    print(res.weights)

    # Now with blocking. 
    block = (numpy.random.rand(100) * 3).astype(numpy.int32)
    res = scranpy.score_gene_set(x, range(100, 200), block=block)
    assert len(res.scores) == x.shape[1]
    assert len(res.weights) == 100
