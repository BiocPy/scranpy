import scranpy
import numpy
import string


LETTERS = list(string.ascii_uppercase)


def test_test_enrichment_simple():
    sets = [
        LETTERS[:10],
        LETTERS[1:10:2],
        LETTERS[9:20]
    ]

    out = scranpy.test_enrichment(LETTERS[:5], sets, LETTERS)
    # See R: phyper(c(5, 2, 0) - 1L, c(10, 5, 11), 26 - c(10, 5, 11), 5L, lower.tail=FALSE)
    assert numpy.allclose(out[0], 0.003830952)
    assert numpy.allclose(out[1], 0.235725144)
    assert numpy.allclose(out[2], 1)

    # Works with log-transformation.
    lout = scranpy.test_enrichment(LETTERS[:5], sets, LETTERS, log=True)
    assert numpy.allclose(lout, numpy.log(out))

    # Works with parallelization.
    pout = scranpy.test_enrichment(LETTERS[:5], sets, LETTERS, num_threads=2)
    assert (pout == out).all()


def test_test_enrichment_truncated():
    sets = [LETTERS[:10]]
    out = scranpy.test_enrichment(LETTERS[:5], sets, universe=list(filter(lambda x : x != "G", LETTERS)))
    # phyper(5 - 1L, 9, 25 - 9, 5, lower.tail=FALSE)
    assert numpy.allclose(out[0], 0.002371542)


def test_test_enrichment_integer_universe():
    sets = [LETTERS[:10]]
    out = scranpy.test_enrichment(LETTERS[:5], sets, universe=26)
    # phyper(5 - 1L, 10, 26 - 10, 5, lower.tail=FALSE)
    assert numpy.allclose(out[0], 0.003830952)
