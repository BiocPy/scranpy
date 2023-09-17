import numpy as np
import scranpy


def test_downsample_by_neighbors():
    mat = np.random.rand(1000, 10)
    keep, reps = scranpy.downsample_by_neighbors(mat, k=10)
    assert len(reps) == 1000
    assert len(keep) < 1000
    assert len(set(keep)) == len(keep)  # unique
