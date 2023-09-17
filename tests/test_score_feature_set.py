from scranpy import score_feature_set, ScoreFeatureSetOptions
import numpy as np


def test_score_feature_set_simple():
    y = np.random.rand(1000, 20)
    subset = [1, 10, 100, 500]
    scores, weights = score_feature_set(y, subset)
    assert len(scores) == 20
    assert len(weights) == 4

    # Trying with scaling.
    scores2, weights2 = score_feature_set(
        y, subset, options=ScoreFeatureSetOptions(scale=True)
    )
    assert (scores2 != scores).any()

    # Trying with parallelization.
    scores2, weights2 = score_feature_set(
        y, subset, options=ScoreFeatureSetOptions(num_threads=3)
    )
    assert (scores2 == scores).all()


def test_score_feature_set_block():
    y = np.random.rand(1000, 100)
    subset = [20, 30, 40, 50, 60]
    block = ["A"] * 50 + ["B"] * 30 + ["C"] * 20

    # With blocking.
    scores, weights = score_feature_set(
        y, subset, options=ScoreFeatureSetOptions(block=block)
    )
    assert len(scores) == 100
    assert len(weights) == 5

    # Without blocking.
    scores2, weights2 = score_feature_set(y, subset)
    assert (scores != scores2).any()
