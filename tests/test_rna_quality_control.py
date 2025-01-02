import scranpy
import numpy
import biocutils
import pytest
import biocframe


def test_compute_rna_qc_metrics():
    y = (numpy.random.rand(2000, 1000) * 5).astype(numpy.uint32)
    sub = numpy.random.rand(y.shape[0]) <= 0.2
    qc = scranpy.compute_rna_qc_metrics(y, { "mito": sub })

    assert numpy.isclose(qc.sum, y.sum(axis=0)).all()
    assert (qc.detected == (y > 0).sum(axis=0)).all()
    assert numpy.isclose(qc.subset_proportion["mito"], y[sub,:].sum(axis=0) / qc.sum).all()
    assert qc.subset_proportion.get_names().as_list() == [ "mito" ]

    bf = qc.to_biocframe()
    assert bf.shape[0] == 1000
    assert (bf.get_column("subset_proportion_mito") == qc.subset_proportion["mito"]).all()

    bf = qc.to_biocframe(flatten=False)
    assert isinstance(bf.get_column("subset_proportion"), biocframe.BiocFrame)
    assert (bf.get_column("subset_proportion").get_column("mito") == qc.subset_proportion["mito"]).all()

    # Also works without names.
    qc = scranpy.compute_rna_qc_metrics(y, [ sub ])
    assert numpy.isclose(qc.subset_proportion[0], y[sub,:].sum(axis=0) / qc.sum).all()
    assert qc.subset_proportion.get_names() is None

    bf = qc.to_biocframe()
    assert (bf.get_column("subset_proportion_0") == qc.subset_proportion[0]).all()

    bf = qc.to_biocframe(flatten=False)
    assert isinstance(bf.get_column("subset_proportion"), biocframe.BiocFrame)
    assert (bf.get_column("subset_proportion").get_column("0") == qc.subset_proportion[0]).all()


def test_suggest_rna_qc_thresholds_simple():
    y = (numpy.random.rand(2000, 1000) * 5).astype(numpy.uint32)
    sub = numpy.random.rand(y.shape[0]) <= 0.2
    qc = scranpy.compute_rna_qc_metrics(y, { "mito": sub })
    thresholds = scranpy.suggest_rna_qc_thresholds(qc, num_mads=1.5)

    assert thresholds.sum < numpy.median(qc.sum)
    assert thresholds.detected < numpy.median(qc.detected)
    assert thresholds.subset_proportion["mito"] > numpy.median(qc.subset_proportion["mito"])
    assert thresholds.subset_proportion.get_names().as_list() == [ "mito" ]

    # Check the filter.
    expected = numpy.logical_and(qc.sum >= thresholds.sum, qc.detected >= thresholds.detected)
    expected = numpy.logical_and(expected, qc.subset_proportion["mito"] <= thresholds.subset_proportion["mito"])
    observed = scranpy.filter_rna_qc_metrics(thresholds, qc)
    assert (expected == observed).all()

    with pytest.raises(ValueError, match="cannot be supplied"):
        scranpy.filter_rna_qc_metrics(thresholds, qc, block=[1,2,3])


def test_suggest_rna_qc_thresholds_blocked():
    numpy.random.seed(42)
    y = (numpy.random.rand(2000, 1000) * 5).astype(numpy.uint32)
    sub = numpy.random.rand(y.shape[0]) <= 0.2
    qc = scranpy.compute_rna_qc_metrics(y, { "mito": sub })
    block = (numpy.random.rand(y.shape[1]) * 3).astype(numpy.int32)

    thresholds = scranpy.suggest_rna_qc_thresholds(qc, num_mads=1.5, block=block)
    assert thresholds.block == [0,1,2]
    assert thresholds.detected.names.as_list() == [ "0", "1", "2" ]
    assert thresholds.subset_proportion["mito"].names.as_list() == [ "0", "1", "2" ]

    for b in range(3):
        keep = block == b 
        assert thresholds.sum[b] < numpy.median(qc.sum[keep])
        assert thresholds.detected[b] < numpy.median(qc.detected[keep])
        assert thresholds.subset_proportion[0][b] > numpy.median(qc.subset_proportion[0][keep])

    # Check the filter.
    expected = numpy.logical_and(qc.sum >= thresholds.sum[block], qc.detected >= thresholds.detected[block])
    expected = numpy.logical_and(expected, qc.subset_proportion[0] <= thresholds.subset_proportion[0][block])
    observed = scranpy.filter_rna_qc_metrics(thresholds, qc, block=block)
    assert (expected == observed).all()

    # Same filtering with just the last block.
    last = block == 2 
    last_observed = scranpy.filter_rna_qc_metrics(
        thresholds,
        scranpy.ComputeRnaQcMetricsResults(
            qc.sum[last],
            qc.detected[last],
            biocutils.NamedList([qc.subset_proportion[0][last]], ["mito"])
        ),
        block=block[last]
    )
    assert (observed[last] == last_observed).all()

    with pytest.raises(ValueError, match="must be supplied"):
        scranpy.filter_rna_qc_metrics(thresholds, qc)

    with pytest.raises(ValueError, match="cannot find"):
        scranpy.filter_rna_qc_metrics(thresholds, qc, block=[1,2,3,4,5,6])
