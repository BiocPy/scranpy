import scranpy
import numpy
import biocutils
import biocframe


def test_compute_adt_qc_metrics():
    y = (numpy.random.rand(100, 1000) * 5).astype(numpy.uint32)
    sub = numpy.random.rand(y.shape[0]) <= 0.2
    qc = scranpy.compute_adt_qc_metrics(y, { "IgG": sub })

    assert numpy.isclose(qc.sum, y.sum(axis=0)).all()
    assert (qc.detected == (y > 0).sum(axis=0)).all()
    assert numpy.isclose(qc.subset_sum["IgG"], y[sub,:].sum(axis=0)).all()
    assert qc.subset_sum.get_names().as_list() == [ "IgG" ]

    bf = qc.to_biocframe()
    assert bf.shape[0] == 1000
    assert (bf.get_column("subset_sum_IgG") == qc.subset_sum["IgG"]).all()

    bf = qc.to_biocframe(flatten=False)
    assert isinstance(bf.get_column("subset_sum"), biocframe.BiocFrame)
    assert (bf.get_column("subset_sum").get_column("IgG") == qc.subset_sum["IgG"]).all()

    # Also works without names.
    qc = scranpy.compute_adt_qc_metrics(y, [ sub ])
    assert numpy.isclose(qc.subset_sum[0], y[sub,:].sum(axis=0)).all()
    assert qc.subset_sum.get_names() is None

    bf = qc.to_biocframe()
    assert (bf.get_column("subset_sum_0") == qc.subset_sum[0]).all()

    bf = qc.to_biocframe(flatten=False)
    assert isinstance(bf.get_column("subset_sum"), biocframe.BiocFrame)
    assert (bf.get_column("subset_sum").get_column("0") == qc.subset_sum[0]).all()


def test_suggest_adt_qc_thresholds_basic():
    y = (numpy.random.rand(100, 1000) * 5).astype(numpy.uint32)
    sub = numpy.random.rand(y.shape[0]) <= 0.2
    qc = scranpy.compute_adt_qc_metrics(y, { "IgG": sub })
    thresholds = scranpy.suggest_adt_qc_thresholds(qc, num_mads=1.5)

    assert thresholds.detected < numpy.median(qc.detected)
    assert thresholds.subset_sum["IgG"] > numpy.median(qc.subset_sum["IgG"])
    assert thresholds.subset_sum.get_names().as_list() == [ "IgG" ]

    # Check the filter.
    expected = numpy.logical_and(qc.detected >= thresholds.detected, qc.subset_sum[0] <= thresholds.subset_sum[0])
    observed = scranpy.filter_adt_qc_metrics(thresholds, qc)
    assert observed.dtype == numpy.dtype("bool")
    assert (expected == observed).all()


def test_suggest_adt_qc_thresholds_blocked():
    numpy.random.seed(42)
    y = (numpy.random.rand(100, 1000) * 5).astype(numpy.uint32)
    sub = numpy.random.rand(y.shape[0]) <= 0.2
    block = (numpy.random.rand(y.shape[1]) * 3).astype(numpy.int32)
    qc = scranpy.compute_adt_qc_metrics(y, { "IgG": sub })

    thresholds = scranpy.suggest_adt_qc_thresholds(qc, num_mads=1.5, block=block)
    assert thresholds.block == [0,1,2]
    assert thresholds.detected.names.as_list() == [ "0", "1", "2" ]
    assert thresholds.subset_sum["IgG"].names.as_list() == [ "0", "1", "2" ]

    # Check the thresholds.
    for i in range(3):
        keep = block == i
        assert thresholds.detected[i] < numpy.median(qc.detected[keep])
        assert thresholds.subset_sum["IgG"][i] > numpy.median(qc.subset_sum["IgG"][keep])

    # Check the filter.
    expected = numpy.logical_and(
        qc.detected >= [thresholds.detected[b] for b in block],
        qc.subset_sum[0] <= [thresholds.subset_sum[0][b] for b in block],
    )
    observed = scranpy.filter_adt_qc_metrics(thresholds, qc, block=block)
    assert observed.dtype == numpy.dtype("bool")
    assert (expected == observed).all()

    # Same filtering with just the last block.
    last = block == 2 
    last_observed = scranpy.filter_adt_qc_metrics(
        thresholds,
        scranpy.ComputeAdtQcMetricsResults(
            qc.sum[last],
            qc.detected[last],
            biocutils.NamedList([qc.subset_sum[0][last]], ["IgG"])
        ),
        block=block[last]
    )
    assert (observed[last] == last_observed).all()
