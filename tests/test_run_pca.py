import scranpy
import numpy


def test_run_pca_simple():
    normed = numpy.random.rand(1000, 100) * 10
    pcs = scranpy.run_pca(normed, number=25)
    assert pcs.block is None

    assert (numpy.abs(pcs.components.mean(axis=1)) < 1e-8).all()
    assert pcs.components.shape[0] == 25

    assert numpy.allclose(pcs.center, normed.mean(axis=1))
    assert pcs.scale is None

    assert (pcs.variance_explained[1:] <= pcs.variance_explained[:-1]).all()
    assert numpy.allclose(pcs.variance_explained, pcs.components.var(axis=1, ddof=1))

    assert pcs.rotation.shape == (1000, 25)

    # Works with scaling.
    spcs = scranpy.run_pca(normed, scale=True)
    assert len(spcs.scale) == 1000


def test_run_pca_blocked():
    numpy.random.seed(52)
    normed = numpy.random.rand(1000, 100) * 10
    block = (numpy.random.rand(normed.shape[1]) * 3).astype(numpy.int32)
    pcs = scranpy.run_pca(normed, number=25, block=block)
    assert pcs.block == [0,1,2]

    assert (numpy.abs(pcs.components.mean(axis=1)) < 1e-8).all()
    assert pcs.components.shape[0] == 25

    assert numpy.allclose(pcs.center[0,:], normed[:,block==0].mean(axis=1))
    assert numpy.allclose(pcs.center[1,:], normed[:,block==1].mean(axis=1))
    assert numpy.allclose(pcs.center[2,:], normed[:,block==2].mean(axis=1))
    assert pcs.scale is None

    # Variance isn't so easily computed this time, so we just check it's sorted.
    assert (pcs.variance_explained[1:] <= pcs.variance_explained[:-1]).all()

    assert pcs.rotation.shape == (1000, 25)

    # Check that the PCs are different from the unblocked case.
    ref = scranpy.run_pca(normed, number=25)
    assert (ref.components != pcs.components).all()


def test_run_pca_residuals():
    normed = numpy.random.rand(1000, 100) * 10
    block = (numpy.random.rand(normed.shape[1]) * 3).astype(numpy.int32)
    pcs = scranpy.run_pca(normed, number=10, block=block, components_from_residuals=True)

    assert (numpy.abs(pcs.components.mean(axis=1)) < 1e-8).all()
    assert pcs.components.shape[0] == 10

    assert (pcs.variance_explained[1:] <= pcs.variance_explained[:-1]).all()
    ratio = pcs.components.var(axis=1) / pcs.variance_explained
    assert ratio.max() - ratio.min() < 1e-8

    assert pcs.rotation.shape == (1000, 10)

    # Make sure it's different from the other options.
    ref1 = scranpy.run_pca(normed, number=10)
    assert (ref1.components != pcs.components).all()
    ref2 = scranpy.run_pca(normed, number=10, block=block)
    assert (ref2.components != pcs.components).all()


def test_run_pca_capped():
    normed = numpy.random.rand(1000, 100) * 10
    pcs = scranpy.run_pca(normed, number=150)
    assert pcs.components.shape[0] == min(normed.shape)
    assert pcs.rotation.shape[1] == min(normed.shape)
