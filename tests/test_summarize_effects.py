import numpy
import scranpy


def test_summarize_effects():
    x = numpy.random.rand(1000, 100)
    g = (numpy.random.rand(100) * 4).astype(numpy.int32)
    summ = scranpy.score_markers(x, g)
    full = scranpy.score_markers(x, g, all_pairwise=True)

    csumm = scranpy.summarize_effects(full.cohens_d)
    assert (csumm[0].min == summ.cohens_d[0].min).all()
    assert (csumm[1].mean == summ.cohens_d[1].mean).all()
    assert (csumm[2].median == summ.cohens_d[2].median).all()
    assert (csumm[3].max == summ.cohens_d[3].max).all()
    assert (csumm[0].min_rank == summ.cohens_d[0].min_rank).all()

    asumm = scranpy.summarize_effects(full.auc, num_threads=2)
    assert (asumm[0].min == summ.auc[0].min).all()
    assert (asumm[1].mean == summ.auc[1].mean).all()
    assert (asumm[2].median == summ.auc[2].median).all()
    assert (asumm[3].max == summ.auc[3].max).all()
    assert (asumm[0].min_rank == summ.auc[0].min_rank).all()

    df = csumm[0].to_biocframe()
    assert df.shape == (1000, 5)
    assert (df.get_column("min_rank") == csumm[0].min_rank).all()
