import scranpy
import numpy
import biocutils


def test_compute_crispr_qc_metrics():
    y = (numpy.random.rand(25, 1000) * 5).astype(numpy.uint32)
    qc = scranpy.compute_crispr_qc_metrics(y)

    assert numpy.isclose(qc.sum, y.sum(axis=0)).all()
    assert (qc.detected == (y > 0).sum(axis=0)).all()
    assert (qc.max_value == y.max(axis=0)).all()
    assert (qc.max_value == [y[m, i] for i, m in enumerate(qc.max_index)]).all()

    bf = qc.to_biocframe()
    assert bf.shape[0] == 1000
    assert (bf.get_column("sum") == qc.sum).all()


def test_suggest_crispr_qc_filters_basic():
    y = (numpy.random.rand(25, 1000) * 5).astype(numpy.uint32)
    qc = scranpy.compute_crispr_qc_metrics(y)
    thresholds = scranpy.suggest_crispr_qc_thresholds(qc, num_mads=1.5)

    expected = qc.max_value >= thresholds.max_value
    observed = scranpy.filter_crispr_qc_metrics(thresholds, qc)
    assert (expected == observed).all()


def test_suggest_crispr_qc_filters_blocked():
    y = (numpy.random.rand(25, 1000) * 5).astype(numpy.uint32)
    block = (numpy.random.rand(y.shape[1]) * 3).astype(numpy.int32)
    qc = scranpy.compute_crispr_qc_metrics(y)

    thresholds = scranpy.suggest_crispr_qc_thresholds(qc, num_mads=1.5, block=block)
    assert thresholds.block == [0,1,2]
    assert thresholds.max_value.names.as_list() == [ "0", "1", "2" ]

    expected = qc.max_value >= thresholds.max_value[block]
    observed = scranpy.filter_crispr_qc_metrics(thresholds, qc, block=block)
    assert (expected == observed).all()

    # Same filtering with just the last block.
    last = block == 2 
    last_observed = scranpy.filter_crispr_qc_metrics(
        thresholds,
        scranpy.ComputeCrisprQcMetricsResults(
            qc.sum[last],
            qc.detected[last],
            qc.max_value[last],
            qc.max_index[last],
        ),
        block=block[last]
    )
    assert (observed[last] == last_observed).all()
