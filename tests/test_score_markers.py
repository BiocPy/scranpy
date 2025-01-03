import scranpy
import numpy
import biocframe


def _check_summaries(summary):
    for x in summary:
        assert (x.min <= x.max).all()
        assert (x.min <= x.mean + 1e-8).all() # add some tolerance for numerical imprecision when averaging identical effects.
        assert (x.min <= x.median).all()
        assert (x.mean <= x.max + 1e-8).all()
        assert (x.median <= x.max).all()
        assert (x.min_rank < 1000).all()
        assert (x.min_rank >= 0).all()


def test_score_markers_simple():
    numpy.random.seed(42)
    x = numpy.random.rand(1000, 100)
    g = (numpy.random.rand(x.shape[1]) * 4).astype(numpy.int32)
    out = scranpy.score_markers(x, g)

    assert out.groups == [0,1,2,3]
    assert numpy.allclose(out.mean[:,0], x[:,g==0].mean(axis=1))
    assert numpy.allclose(out.detected[:,3], (x[:,g==3] > 0).mean(axis=1))

    _check_summaries(out.cohens_d)
    _check_summaries(out.auc)
    _check_summaries(out.delta_mean)
    _check_summaries(out.delta_detected)

    for aeff in out.auc:
        assert (aeff.min >= 0).all() and (aeff.min <= 1).all()
        assert (aeff.mean >= 0).all() and (aeff.mean <= 1).all()
        assert (aeff.median >= 0).all() and (aeff.median <= 1).all()
        assert (aeff.max >= 0).all() and (aeff.max <= 1).all()

    pout = scranpy.score_markers(x, g, num_threads=2)
    assert (out.mean == pout.mean).all()
    assert (out.detected == pout.detected).all()
    assert (out.cohens_d[0].mean == pout.cohens_d[0].mean).all()
    assert (out.delta_detected[1].median == pout.delta_detected[1].median).all()
    assert (out.delta_mean[2].max == pout.delta_mean[2].max).all()
    assert (out.auc[3].min_rank == pout.auc[3].min_rank).all()

    # Works without the AUC.
    aout = scranpy.score_markers(x, g, compute_auc=False)
    assert aout.auc is None
    assert not aout.cohens_d is None
    assert (aout.mean == out.mean).all()
    assert (aout.detected == out.detected).all()

    # Works without anything.
    empty = scranpy.score_markers(x, g, compute_auc=False, compute_cohens_d=False, compute_delta_detected=False, compute_delta_mean=False)
    assert empty.auc is None
    assert empty.cohens_d is None
    assert empty.delta_mean is None
    assert empty.delta_detected is None
    assert (empty.mean == out.mean).all()
    assert (empty.detected == out.detected).all()

    # This can be converted to BiocFrames.
    dfs = out.to_biocframes()
    assert len(dfs) == 4
    assert dfs.get_names().as_list() == ["0", "1", "2", "3"]
    assert dfs[0].shape[0] == x.shape[0]
    assert (dfs[0].get_column("cohens_d_median") == out.cohens_d[0].median).all()
    assert (dfs[1].get_column("auc_min_rank") == out.auc[1].min_rank).all()
    assert (dfs[2].get_column("mean") == out.mean[:,2]).all()
    assert (dfs[3].get_column("detected") == out.detected[:,3]).all()

    edfs = empty.to_biocframes(include_mean=False, include_detected=False)
    assert edfs[0].shape == (x.shape[0], 0)


def test_score_markers_blocked():
    numpy.random.seed(421)
    x = numpy.random.rand(1000, 100)
    g = (numpy.random.rand(x.shape[1]) * 4).astype(numpy.int32)
    b = (numpy.random.rand(x.shape[1]) * 3).astype(numpy.int32)
    out = scranpy.score_markers(x, g, block=b, block_weight_policy="equal")

    bkeep = (g == 2)
    assert numpy.allclose(out.mean[:,2], (
        x[:,numpy.logical_and(bkeep, b == 0)].mean(axis=1) + 
        x[:,numpy.logical_and(bkeep, b == 1)].mean(axis=1) + 
        x[:,numpy.logical_and(bkeep, b == 2)].mean(axis=1)
    )/3)
    ckeep = (g == 3)
    assert numpy.allclose(out.mean[:,3], (
        x[:,numpy.logical_and(ckeep, b == 0)].mean(axis=1) + 
        x[:,numpy.logical_and(ckeep, b == 1)].mean(axis=1) + 
        x[:,numpy.logical_and(ckeep, b == 2)].mean(axis=1)
    )/3)

    _check_summaries(out.cohens_d)
    _check_summaries(out.auc)
    _check_summaries(out.delta_mean)
    _check_summaries(out.delta_detected)

    for aeff in out.auc:
        assert (aeff.min >= 0).all() and (aeff.min <= 1).all()
        assert (aeff.mean >= 0).all() and (aeff.mean <= 1).all()
        assert (aeff.median >= 0).all() and (aeff.median <= 1).all()
        assert (aeff.max >= 0).all() and (aeff.max <= 1).all()


def test_score_markers_pairwise():
    numpy.random.seed(422)
    x = numpy.random.rand(1000, 100)
    g = (numpy.random.rand(x.shape[1]) * 4).astype(numpy.int32)
    full = scranpy.score_markers(x, g, all_pairwise=True)

    # Checking that we set the dimensions correctly.
    for g1 in range(4):
        assert (full.delta_mean[g1,g1,:] == 0).all()
        assert (full.auc[g1,g1,:] == 0).all()
        for g2 in range(g1):
            assert numpy.allclose(full.delta_mean[g1,g2,:], -full.delta_mean[g2,g1,:])
            assert numpy.allclose(full.auc[g1,g2,:], 1 - full.auc[g2,g1,:])

    assert (full.auc >= 0).all()
    assert (full.auc <= 1).all()

    # Works without AUCs.
    aout = scranpy.score_markers(x, g, all_pairwise=True, compute_auc=False)
    assert aout.auc is None
    assert (aout.mean == full.mean).all()
    assert (aout.detected == full.detected).all()

    # Works without anything.
    empty  = scranpy.score_markers(x, g, compute_auc=False, compute_cohens_d=False, compute_delta_detected=False, compute_delta_mean=False, all_pairwise=True)
    assert empty.auc is None
    assert empty.cohens_d is None
    assert empty.delta_mean is None
    assert empty.delta_detected is None
    assert (empty.mean == full.mean).all()
    assert (empty.detected == full.detected).all()

    # Works with blocking.
    b = (numpy.random.rand(x.shape[1]) * 3).astype(numpy.int32)
    bout = scranpy.score_markers(x, g, block=b, block_weight_policy="equal", all_pairwise=True)
    sbout = scranpy.score_markers(x, g, block=b, block_weight_policy="equal")
    assert (bout.mean == sbout.mean).all()
    assert (bout.detected == sbout.detected).all()
